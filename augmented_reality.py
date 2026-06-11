"""Realidad aumentada (apartado 2.3): robot UR5 virtual sobre la imagen.

Proyecta un robot UR5 virtual a escala real sobre la imagen de la camara
izquierda, anclado en el origen del mundo (el tablero capturado), de modo que
su TCP sigue la herramienta fisica real ("la agarra" en la imagen).

Geometria:
  - se trabaja en el frame del tablero crudo de la captura de origen (PnP),
    que es el unico geometricamente registrado con la imagen. Las correcciones
    `world_flip_z`/`world_scale` son para el visor Swift y aqui se ignoran;
  - K es la matriz de la camara rectificada (`stereo.p_left`), con distorsion
    cero porque la rectificacion ya la elimina (mismo K que uso `solvePnP`);
  - se resuelve IK de solo posicion para que el TCP del UR5 alcance la punta
    de la herramienta, y se proyectan los eslabones con `cv2.projectPoints`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app_state import BASE_DIR

try:
    import cv2 as cv
    import numpy as np
except ModuleNotFoundError:
    cv = None
    np = None

try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
except ModuleNotFoundError:
    rtb = None
    SE3 = None


WORLD_TRANSFORM_PATH = BASE_DIR / "calibration" / "world_transform.npz"
# Alcance util del UR5 (m). Si la herramienta cae mas lejos del origen, se
# proyecta el objetivo a la frontera para que la IK siempre resuelva.
UR5_MAX_REACH_M = 0.82


class AugmentedRobot:
    """Robot UR5 virtual proyectado sobre la imagen, siguiendo la herramienta.

    ``robot_scale`` dibuja el robot como maqueta (p. ej. 0.4 = 40% del tamano
    real) manteniendo la misma configuracion articular: un gemelo en miniatura
    cuyos movimientos se aprecian enteros dentro del encuadre.
    """

    def __init__(
        self,
        stereo: Any,
        world_transform_path: Path | None = None,
        robot_scale: float = 0.4,
    ) -> None:
        self.available = False
        if cv is None or np is None or rtb is None:
            return
        path = world_transform_path or WORLD_TRANSFORM_PATH
        if not path.exists():
            print(
                "AR: falta calibration/world_transform.npz. Captura el origen "
                "del mundo para registrar la escena antes de usar AR."
            )
            return

        data = np.load(path)
        # npz guarda mundo<-camara (R_world_cam, t_world en cm).
        self.r_world_cam = np.asarray(data["rotation"], dtype=np.float64)
        self.t_world_cm = np.asarray(data["translation"], dtype=np.float64).reshape(3)
        # Inversa: camara<-mundo, lo que necesita projectPoints.
        r_cam_world = self.r_world_cam.T
        t_cam_world = -r_cam_world @ self.t_world_cm
        self.r_cam_world = r_cam_world
        self.t_cam_world = t_cam_world
        self.rvec, _ = cv.Rodrigues(r_cam_world)
        self.tvec = t_cam_world.reshape(3, 1)
        # K de la camara rectificada (distorsion cero tras rectificar).
        self.camera_matrix = np.asarray(stereo.p_left, dtype=np.float64)[:3, :3]
        self.dist = np.zeros((5, 1), dtype=np.float64)
        self.fx = float(self.camera_matrix[0, 0])

        self.ur5 = rtb.models.UR5()
        self.ur5.q = self.ur5.qz
        self.robot_scale = max(0.05, min(1.0, float(robot_scale)))
        self._has_pose = False
        self._reached = False
        self._last_target_world_cm: Any = None
        self.available = True

    # -- transformaciones -------------------------------------------------
    def _camera_cm_to_world_cm(self, point_cm: Any) -> Any:
        point = np.asarray(point_cm, dtype=np.float64)
        return self.r_world_cam @ point + self.t_world_cm

    def project_world_cm(self, points_world_cm: Any) -> Any:
        """Proyecta puntos 3D del frame mundo (cm) a pixeles de la imagen."""
        pts = np.asarray(points_world_cm, dtype=np.float64).reshape(-1, 1, 3)
        projected, _ = cv.projectPoints(
            pts, self.rvec, self.tvec, self.camera_matrix, self.dist
        )
        return projected.reshape(-1, 2)

    # -- actualizacion de la pose del robot -------------------------------
    def update(self, payload: dict[str, Any] | None) -> None:
        """Resuelve IK de solo posicion para que el TCP siga la herramienta."""
        if not self.available or not payload:
            return
        position_cm = payload.get("position_cm")
        if not position_cm:
            return
        world_cm = self._camera_cm_to_world_cm(position_cm[:3])
        self._last_target_world_cm = world_cm.copy()
        target_m = world_cm / 100.0
        # Clamp al alcance del UR5 para que la IK siempre resuelva.
        reach = float(np.linalg.norm(target_m))
        within_reach = reach <= UR5_MAX_REACH_M
        if not within_reach and reach > 1e-9:
            target_m = target_m * (UR5_MAX_REACH_M / reach)

        solution = self.ur5.ikine_LM(
            SE3(*target_m), q0=self.ur5.q, mask=[1, 1, 1, 0, 0, 0]
        )
        if solution.success:
            self.ur5.q = solution.q
            self._has_pose = True
            self._reached = within_reach
        else:
            self._reached = False

    # -- dibujado ---------------------------------------------------------
    def _chain_world_cm(self) -> Any:
        # fkine_all da las poses de TODOS los frames del modelo, incluyendo
        # algunos espurios en el origen (base/tool repetidos). Nos quedamos con
        # la cadena cinematica: filtramos los frames en el origen (salvo la
        # base) y forzamos el TCP real (`fkine`) como ultimo punto.
        poses = self.ur5.fkine_all(self.ur5.q)
        chain_cm = [poses[0].t * 100.0]
        for pose in poses[1:]:
            if float(np.linalg.norm(pose.t)) < 1e-6:
                continue
            chain_cm.append(pose.t * 100.0)
        chain_cm.append(self.ur5.fkine(self.ur5.q).t * 100.0)
        return np.asarray(chain_cm)

    def _joint_pixels(self) -> Any:
        return self.project_world_cm(self._chain_world_cm())

    def _camera_depth_cm(self, points_world_cm: Any) -> Any:
        """Profundidad (cm) de puntos mundo en el frame camara (eje Z optico)."""
        pts = np.atleast_2d(np.asarray(points_world_cm, dtype=np.float64))
        cam = (self.r_cam_world @ pts.T).T + self.t_cam_world
        return cam[:, 2]

    def _draw_world_axes(self, frame: Any, length_cm: float = 12.0) -> None:
        axes_world = np.array(
            [[0, 0, 0], [length_cm, 0, 0], [0, length_cm, 0], [0, 0, length_cm]],
            dtype=np.float64,
        )
        pts = np.clip(self.project_world_cm(axes_world), -4000, 4000).astype(int)
        o, x, y, z = pts
        for end, color, label in ((x, (0, 0, 255), "X"), (y, (0, 255, 0), "Y"), (z, (255, 80, 80), "Z")):
            cv.line(frame, tuple(o), tuple(end), color, 2, cv.LINE_AA)
            cv.putText(
                frame, label, (end[0] + 4, end[1] - 4),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv.LINE_AA,
            )

    def _draw_ground(self, frame: Any, size_cm: float = 24.0, step_cm: float = 8.0) -> None:
        """Rejilla sutil sobre el plano de la mesa (z=0) alrededor del origen."""
        overlay = frame.copy()
        ticks = np.arange(-size_cm, size_cm + step_cm / 2.0, step_cm)
        for v in ticks:
            ends = np.array(
                [[v, -size_cm, 0.0], [v, size_cm, 0.0],
                 [-size_cm, v, 0.0], [size_cm, v, 0.0]],
                dtype=np.float64,
            )
            p = np.clip(self.project_world_cm(ends), -4000, 4000).astype(int)
            cv.line(overlay, tuple(p[0]), tuple(p[1]), (255, 255, 255), 1, cv.LINE_AA)
            cv.line(overlay, tuple(p[2]), tuple(p[3]), (255, 255, 255), 1, cv.LINE_AA)
        cv.addWeighted(overlay, 0.22, frame, 0.78, 0.0, frame)

    def _apply_shadow(self, frame: Any, chain_cm: Any) -> None:
        """Sombra suave del robot proyectada sobre la mesa (luz cenital)."""
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        flat = chain_cm.copy()
        flat[:, 2] = 0.0
        depths = np.maximum(self._camera_depth_cm(flat), 1.0)
        px = np.clip(self.project_world_cm(flat), -4000, 4000).astype(int)
        for i in range(len(px) - 1):
            radius_px = self.fx * 4.5 * self.robot_scale / float(depths[i + 1])
            thickness = int(np.clip(radius_px * 2.0, 2, 160))
            cv.line(mask, tuple(px[i]), tuple(px[i + 1]), 1.0, thickness, cv.LINE_AA)
        mask = cv.GaussianBlur(mask, (31, 31), 0)
        frame[:] = (frame * (1.0 - 0.45 * mask[..., None])).astype(np.uint8)

    def _draw_capsule(
        self,
        frame: Any,
        p0: Any,
        p1: Any,
        r0: float,
        r1: float,
        color: tuple[int, int, int],
        edge: tuple[int, int, int],
        spec: tuple[int, int, int],
    ) -> None:
        """Eslabon como cilindro 2.5D: tubo + bordes oscuros + brillo axial."""
        r0 = float(np.clip(r0, 2.0, 300.0))
        r1 = float(np.clip(r1, 2.0, 300.0))
        d = p1 - p0
        norm = float(np.linalg.norm(d))
        if norm < 1e-3:
            cv.circle(frame, tuple(p0.astype(int)), int(max(r0, r1)), color, -1, cv.LINE_AA)
            return
        u = np.array([-d[1], d[0]]) / norm
        poly = np.array(
            [p0 + u * r0, p1 + u * r1, p1 - u * r1, p0 - u * r0], dtype=np.int32
        )
        cv.fillPoly(frame, [poly], color, cv.LINE_AA)
        cv.circle(frame, tuple(p0.astype(int)), int(r0), color, -1, cv.LINE_AA)
        cv.circle(frame, tuple(p1.astype(int)), int(r1), color, -1, cv.LINE_AA)
        cv.line(
            frame, tuple((p0 + u * r0).astype(int)), tuple((p1 + u * r1).astype(int)),
            edge, 2, cv.LINE_AA,
        )
        cv.line(
            frame, tuple((p0 - u * r0).astype(int)), tuple((p1 - u * r1).astype(int)),
            edge, 2, cv.LINE_AA,
        )
        # Linea especular descentrada: vende el volumen del cilindro.
        spec_w = max(1, int(min(r0, r1) * 0.35))
        cv.line(
            frame,
            tuple((p0 - u * r0 * 0.45).astype(int)),
            tuple((p1 - u * r1 * 0.45).astype(int)),
            spec, spec_w, cv.LINE_AA,
        )

    def _draw_robot(self, frame: Any, chain_cm: Any, depths: Any) -> None:
        """UR5 con estetica realista: eslabones grises, juntas azul UR."""
        px = np.clip(self.project_world_cm(chain_cm), -4000, 4000)
        n = len(chain_cm)
        # Radios reales aproximados del UR5 (hombro ~4.2 cm, muneca ~2.6 cm),
        # escalados igual que la cadena para mantener las proporciones.
        radii_cm = np.linspace(4.2, 2.6, n) * self.robot_scale
        link_color, edge, spec = (168, 166, 160), (84, 82, 78), (235, 235, 230)
        joint_color, joint_rim = (185, 150, 60), (110, 85, 30)

        # Algoritmo del pintor: dibujar primero lo mas lejano a la camara.
        items: list[tuple[float, str, int]] = []
        for i in range(n - 1):
            items.append((float((depths[i] + depths[i + 1]) / 2.0), "link", i))
        for i in range(1, n - 1):
            items.append((float(depths[i]) - 0.1, "joint", i))
        items.sort(key=lambda item: -item[0])

        for _, kind, i in items:
            if kind == "link":
                r0 = self.fx * radii_cm[i] / max(float(depths[i]), 1.0)
                r1 = self.fx * radii_cm[i + 1] / max(float(depths[i + 1]), 1.0)
                self._draw_capsule(frame, px[i], px[i + 1], r0, r1, link_color, edge, spec)
            else:
                r = self.fx * (radii_cm[i] * 1.12) / max(float(depths[i]), 1.0)
                r = float(np.clip(r, 3.0, 320.0))
                center = tuple(px[i].astype(int))
                cv.circle(frame, center, int(r), joint_color, -1, cv.LINE_AA)
                cv.circle(frame, center, int(r), joint_rim, 2, cv.LINE_AA)
                highlight = (int(center[0] - r * 0.3), int(center[1] - r * 0.3))
                cv.circle(frame, highlight, max(1, int(r * 0.25)), (255, 240, 205), -1, cv.LINE_AA)

        tcp = tuple(px[-1].astype(int))
        cv.circle(frame, tcp, 9, (60, 60, 255), 2, cv.LINE_AA)
        cv.putText(
            frame, "TCP", (tcp[0] + 12, tcp[1] - 8),
            cv.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 2, cv.LINE_AA,
        )

    def _draw_tool_reticle(self, frame: Any) -> None:
        """Reticula sobre la herramienta real; se pone verde al alcanzarla."""
        if self._last_target_world_cm is None:
            return
        target_px = self.project_world_cm(self._last_target_world_cm)[0]
        # La maqueta no toca fisicamente la herramienta: el "alcanzada" indica
        # que el robot a escala real habria llegado (IK resuelta dentro del
        # alcance), y se une maqueta-herramienta con una linea guia.
        locked = self._reached
        color = (90, 220, 90) if locked else (60, 200, 255)
        tcp_px = self.project_world_cm(
            self._chain_world_cm()[-1] * self.robot_scale
        )[0]
        cv.line(
            frame,
            tuple(np.clip(tcp_px, -4000, 4000).astype(int)),
            tuple(np.clip(target_px, -4000, 4000).astype(int)),
            color, 1, cv.LINE_AA,
        )
        c = tuple(np.clip(target_px, -4000, 4000).astype(int))
        cv.circle(frame, c, 14, color, 2, cv.LINE_AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv.line(
                frame,
                (c[0] + dx * 8, c[1] + dy * 8),
                (c[0] + dx * 20, c[1] + dy * 20),
                color, 2, cv.LINE_AA,
            )
        if locked:
            cv.putText(
                frame, "HERRAMIENTA ALCANZADA", (c[0] + 24, c[1] + 24),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv.LINE_AA,
            )

    def _hud(self, frame: Any, text: str, color: tuple[int, int, int]) -> None:
        """Banda translucida con el estado, arriba a la izquierda."""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = 10, 8, min(w - 10, 560), 44
        roi = frame[y0:y1, x0:x1]
        frame[y0:y1, x0:x1] = (roi * 0.35).astype(np.uint8)
        cv.putText(
            frame, text, (x0 + 10, y1 - 12),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv.LINE_AA,
        )

    def render(self, left_frame: Any, payload: dict[str, Any] | None = None) -> Any:
        """Devuelve una copia de ``left_frame`` con el robot virtual encima."""
        if not self.available:
            return left_frame
        if payload is not None:
            self.update(payload)
        frame = left_frame.copy()
        self._draw_ground(frame)
        self._draw_world_axes(frame)

        if not self._has_pose:
            self._hud(frame, "AR: esperando pose de la herramienta...", (0, 255, 255))
            return frame

        # Maqueta: misma configuracion articular, dibujada a escala reducida
        # desde el origen para que los movimientos quepan en el encuadre.
        chain_cm = self._chain_world_cm() * self.robot_scale
        depths = self._camera_depth_cm(chain_cm)
        if float(np.min(depths)) <= 2.0:
            self._hud(frame, "AR: robot fuera del encuadre", (0, 255, 255))
            return frame

        self._apply_shadow(frame, chain_cm)
        self._draw_robot(frame, chain_cm, depths)
        self._draw_tool_reticle(frame)
        self._hud(
            frame,
            f"Realidad aumentada: UR5 virtual (escala {self.robot_scale:.0%})",
            (255, 220, 90),
        )
        return frame


def run_pose_with_ar() -> None:
    """Bucle de pose mostrando el robot UR5 virtual en AR sobre la camara."""
    if cv is None or np is None:
        print("OpenCV/NumPy no instalados.")
        return
    if rtb is None:
        print("roboticstoolbox-python no esta instalado. No se puede hacer AR.")
        return

    import time

    from app_state import load_config, save_last_pose
    from main_pose import StereoPoseProcessor, draw_pose_frame
    from vision.camera import (
        CameraSource,
        load_stereo_camera_configs,
        read_stereo_pair,
    )
    from vision.frame_debug import show_debug_frame

    config = load_config()
    try:
        robot_scale = float(config.get("ar", {}).get("robot_scale", 0.4))
    except (TypeError, ValueError):
        robot_scale = 0.4
    processor = StereoPoseProcessor(config)
    ar = AugmentedRobot(processor.stereo, robot_scale=robot_scale)
    if not ar.available:
        return

    left_config, right_config = load_stereo_camera_configs()
    left_camera = CameraSource(left_config)
    right_camera = CameraSource(right_config)
    last_saved_at = 0.0

    print("Realidad aumentada. El UR5 virtual sigue la herramienta. Pulsa q/Esc.")
    try:
        left_camera.open()
        right_camera.open()
        while True:
            left_frame, right_frame = read_stereo_pair(left_camera, right_camera)
            result = processor.process(left_frame, right_frame)

            if result.pose_payload is not None:
                now = time.monotonic()
                if now - last_saved_at >= 0.5:
                    save_last_pose(result.pose_payload)
                    last_saved_at = now

            left_debug, right_debug = draw_pose_frame(result)
            ar_frame = ar.render(result.left_rect, result.pose_payload)

            show_debug_frame(f"Pose izquierda [{left_config.index}]", left_debug)
            show_debug_frame(f"Pose derecha [{right_config.index}]", right_debug)
            show_debug_frame("Realidad aumentada (UR5 virtual)", ar_frame)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        left_camera.release()
        right_camera.release()
        cv.destroyAllWindows()
