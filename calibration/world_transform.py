"""Captura del origen fisico (frame mundo).

Coloca el tablero de calibracion sobre la mesa en la posicion exacta donde
quieres que este la base del robot virtual. Esta utilidad detecta el tablero
en la imagen rectificada de la camara izquierda, resuelve ``solvePnP`` para
obtener la pose camara<->mundo y guarda la transformacion en
``calibration/world_transform.npz``.

Convencion del frame mundo:
  - origen en la primera esquina detectada del tablero;
  - eje X a lo largo del tablero (direccion de columnas);
  - eje Y a lo largo del tablero (direccion de filas);
  - eje Z saliendo del tablero (hacia arriba si el tablero esta plano).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from app_state import BASE_DIR, load_config, save_calibration_info, save_config
from vision.camera import CameraSource, load_stereo_camera_configs
from vision.stereo import load_stereo_calibration

try:
    import cv2 as cv
    import numpy as np
except ModuleNotFoundError:
    cv = None
    np = None


WORLD_TRANSFORM_PATH = BASE_DIR / "calibration" / "world_transform.npz"


@dataclass
class WorldTransform:
    """Transformacion rigida del frame de la camara izquierda al frame mundo.

    Aplicacion: ``point_world = scale * (rotation @ point_camera + translation)``.
    Las traslaciones estan en cm. ``scale`` (default 1.0) permite comprimir
    o expandir el workspace virtual respecto al fisico, util cuando el
    alcance del robot virtual no encaja con el espacio real.
    """

    rotation: Any  # (3, 3) R_world_cam
    translation: Any  # (3,) t_world_cam en cm
    rms_reprojection: float
    captured_at: str
    scale: float = 1.0

    def transform_point_cm(
        self, point_cm: Sequence[float]
    ) -> tuple[float, float, float]:
        if np is None:
            raise RuntimeError("NumPy no esta instalado.")
        point = np.asarray(point_cm, dtype=float)
        transformed = (self.rotation @ point + self.translation) * self.scale
        return (float(transformed[0]), float(transformed[1]), float(transformed[2]))

    def transform_direction(
        self, direction: Sequence[float]
    ) -> tuple[float, float, float]:
        if np is None:
            raise RuntimeError("NumPy no esta instalado.")
        vector = np.asarray(direction, dtype=float)
        transformed = self.rotation @ vector
        return (float(transformed[0]), float(transformed[1]), float(transformed[2]))


def _rotation_x(deg: float) -> Any:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(deg: float) -> Any:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_z(deg: float) -> Any:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def load_world_transform() -> WorldTransform | None:
    """Carga ``calibration/world_transform.npz`` y aplica las correcciones.

    Los tres angulos ``calibration.world_{roll,pitch,yaw}_deg`` rotan el
    frame mundo alrededor de X, Y y Z respectivamente, sin necesidad de
    volver a capturar el origen. Se aplican en orden Rx @ Ry @ Rz.
    """
    if np is None or not WORLD_TRANSFORM_PATH.exists():
        return None
    data = np.load(WORLD_TRANSFORM_PATH)
    rotation = data["rotation"]
    translation = data["translation"]

    calibration_section = load_config().get("calibration", {})
    roll = float(calibration_section.get("world_roll_deg", 0.0))
    pitch = float(calibration_section.get("world_pitch_deg", 0.0))
    yaw = float(calibration_section.get("world_yaw_deg", 0.0))
    flip_z = bool(calibration_section.get("world_flip_z", False))

    if abs(roll) > 1e-9 or abs(pitch) > 1e-9 or abs(yaw) > 1e-9:
        correction = _rotation_x(roll) @ _rotation_y(pitch) @ _rotation_z(yaw)
        rotation = correction @ rotation
        translation = correction @ translation
    if flip_z:
        # Rotacion 180 grados alrededor del eje X (Rx_180 = diag(1,-1,-1)).
        # ANTES era diag(1,1,-1), una REFLEXION con det=-1: rompe la matriz
        # de orientacion de la herramienta (la IK del UR5 espera rotaciones
        # right-handed). Esta version es una rotacion real (det=+1): invierte
        # tanto Y como Z, asi que si Y queda al reves usa world_yaw=180.
        flip = np.diag([1.0, -1.0, -1.0])
        rotation = flip @ rotation
        translation = flip @ translation

    scale = float(calibration_section.get("world_scale", 1.0))

    return WorldTransform(
        rotation=rotation,
        translation=translation,
        rms_reprojection=float(data["rms_reprojection"]),
        captured_at=str(data["captured_at"]),
        scale=scale,
    )


def _rotate_axis_90deg(axis_key: str) -> None:
    config = load_config()
    calibration_section = config.setdefault("calibration", {})
    current = float(calibration_section.get(axis_key, 0.0))
    new_value = (current + 90.0) % 360.0
    calibration_section[axis_key] = new_value
    save_config(config)
    print(f"{axis_key}: {current:.1f} -> {new_value:.1f} grados.")


def _reset_world_rotation_corrections(config: dict[str, Any]) -> bool:
    """Elimina rotaciones manuales que no pertenecen a una captura nueva."""
    calibration_section = config.setdefault("calibration", {})
    changed = False
    for key in ("world_roll_deg", "world_pitch_deg", "world_yaw_deg"):
        current = float(calibration_section.get(key, 0.0))
        if abs(current) > 1e-9:
            changed = True
        calibration_section[key] = 0.0
    return changed


def rotate_world_roll_90deg() -> None:
    """Suma 90 grados al roll (rotacion alrededor del eje X mundo)."""
    _rotate_axis_90deg("world_roll_deg")


def rotate_world_pitch_90deg() -> None:
    """Suma 90 grados al pitch (rotacion alrededor del eje Y mundo)."""
    _rotate_axis_90deg("world_pitch_deg")


def rotate_world_yaw_90deg() -> None:
    """Suma 90 grados al yaw (rotacion alrededor del eje Z mundo)."""
    _rotate_axis_90deg("world_yaw_deg")


def toggle_world_flip_z() -> None:
    """Invierte el eje Z mundo (arriba<->abajo) sin tocar el plano horizontal."""
    config = load_config()
    calibration_section = config.setdefault("calibration", {})
    current = bool(calibration_section.get("world_flip_z", False))
    calibration_section["world_flip_z"] = not current
    save_config(config)
    print(f"world_flip_z: {current} -> {not current}.")


def configure_world_scale() -> None:
    """Pregunta el factor de escala del workspace virtual.

    Valores > 1.0 amplifican: 1 cm real -> N cm virtuales. Util cuando el
    espacio fisico es menor que el alcance util del robot.
    Valores < 1.0 comprimen el espacio virtual.
    """
    config = load_config()
    calibration_section = config.setdefault("calibration", {})
    current = float(calibration_section.get("world_scale", 1.0))

    print(
        "\nFactor de escala del workspace virtual.\n"
        "  1.0 = sin escalado (1 cm real = 1 cm virtual)\n"
        "  2.0 = el robot se mueve el doble que la herramienta\n"
        "  0.5 = el robot se mueve la mitad (comprime espacio grande)"
    )
    raw = input(f"Nuevo valor world_scale [{current}]: ").strip()
    if raw == "":
        print("Sin cambios.")
        return
    try:
        new_value = float(raw)
    except ValueError:
        print("Valor no valido. Sin cambios.")
        return
    if new_value <= 0.0:
        print("El factor debe ser positivo. Sin cambios.")
        return

    calibration_section["world_scale"] = new_value
    save_config(config)
    print(f"world_scale: {current} -> {new_value}.")


def _save_world_transform(transform: WorldTransform) -> None:
    WORLD_TRANSFORM_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        WORLD_TRANSFORM_PATH,
        rotation=transform.rotation,
        translation=transform.translation,
        rms_reprojection=transform.rms_reprojection,
        captured_at=transform.captured_at,
    )


def _build_object_points(
    pattern_size: tuple[int, int], square_size_cm: float
) -> Any:
    cols, rows = pattern_size
    points = np.zeros((cols * rows, 3), dtype=np.float32)
    points[:, :2] = np.indices((cols, rows)).T.reshape(-1, 2)
    points[:, :2] *= square_size_cm
    return points


def _draw_preview(
    frame: Any,
    pattern_size: tuple[int, int],
    corners: Any | None,
    found: bool,
    last_rms: float | None,
) -> None:
    if corners is not None:
        cv.drawChessboardCorners(frame, pattern_size, corners, found)

    status = "Tablero detectado" if found else "Buscando tablero..."
    color = (0, 255, 0) if found else (0, 0, 255)
    cv.putText(
        frame,
        status,
        (20, 32),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        status,
        (20, 32),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv.LINE_AA,
    )

    hint = "SPACE para capturar | Q/Esc para salir"
    cv.putText(
        frame,
        hint,
        (20, 64),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        hint,
        (20, 64),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
        cv.LINE_AA,
    )

    if last_rms is not None:
        cv.putText(
            frame,
            f"Ultima captura RMS: {last_rms:.2f}",
            (20, 96),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            f"Ultima captura RMS: {last_rms:.2f}",
            (20, 96),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )


def _solve_and_save(
    object_points: Any,
    corners: Any,
    camera_matrix: Any,
) -> WorldTransform:
    """Resuelve solvePnP y construye la transformacion camara->mundo."""
    dist_zero = np.zeros((5, 1), dtype=np.float64)
    success, rvec, tvec = cv.solvePnP(
        object_points,
        corners,
        camera_matrix,
        dist_zero,
        flags=cv.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP no convergio para la captura.")

    r_cam_world, _ = cv.Rodrigues(rvec)
    t_cam_world = tvec.reshape(3)

    # Reproyeccion para medir error
    projected, _ = cv.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_zero
    )
    error = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))

    # Invertir para tener mundo<-camara (el visor recibe puntos en frame
    # camara y los necesita en frame mundo).
    rotation_world_cam = r_cam_world.T
    translation_world_cm = -rotation_world_cam @ t_cam_world

    return WorldTransform(
        rotation=rotation_world_cam.astype(np.float64),
        translation=translation_world_cm.astype(np.float64),
        rms_reprojection=rms,
        captured_at=datetime.now().isoformat(timespec="seconds"),
    )


def capture_world_transform() -> None:
    """Entry point del menu: captura interactiva del origen fisico."""
    if cv is None or np is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    config = load_config()
    calibration_config = config.get("calibration", {})
    pattern_size = tuple(
        int(value) for value in calibration_config.get("chessboard_pattern_size", [7, 7])
    )
    square_size_cm = float(
        calibration_config.get("chessboard_square_size", 2.1)
    )

    stereo = load_stereo_calibration()
    camera_matrix = stereo.p_left[:3, :3].astype(np.float64)
    left_config, _ = load_stereo_camera_configs()
    left_camera = CameraSource(left_config)

    object_points = _build_object_points(pattern_size, square_size_cm)
    last_rms: float | None = None
    captured: WorldTransform | None = None

    print(
        "\nColoca el tablero plano sobre la mesa en el origen deseado y pulsa "
        "SPACE cuando este detectado."
    )
    print(
        f"Patron: {pattern_size[0]}x{pattern_size[1]} esquinas internas, "
        f"square={square_size_cm} cm."
    )

    try:
        left_camera.open()
        while True:
            raw_frame = left_camera.read()
            rectified = cv.remap(
                raw_frame, stereo.left_map_x, stereo.left_map_y, cv.INTER_LINEAR
            )
            gray = cv.cvtColor(rectified, cv.COLOR_BGR2GRAY)

            found, corners = cv.findChessboardCorners(
                gray,
                pattern_size,
                flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                corners = cv.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01),
                )

            preview = rectified.copy()
            _draw_preview(preview, pattern_size, corners, found, last_rms)
            cv.imshow("Capturar origen del mundo (izquierda rectificada)", preview)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" ") and found:
                captured = _solve_and_save(object_points, corners, camera_matrix)
                last_rms = captured.rms_reprojection
                print(
                    f"Captura ok. RMS reproyeccion: {last_rms:.3f} px. "
                    "Pulsa SPACE de nuevo para sustituir o Q/Esc para guardar y salir."
                )
    finally:
        left_camera.release()
        cv.destroyAllWindows()

    if captured is None:
        print("No se capturo ninguna pose. world_transform.npz NO modificado.")
        return

    _save_world_transform(captured)
    if _reset_world_rotation_corrections(config):
        save_config(config)
        print("Rotaciones manuales del mundo reiniciadas a 0 grados.")
    save_calibration_info(
        {
            "world_transform_path": str(
                WORLD_TRANSFORM_PATH.relative_to(BASE_DIR)
            ),
            "world_transform_rms_px": captured.rms_reprojection,
            "world_transform_captured_at": captured.captured_at,
        }
    )
    print(
        f"\nGuardado {WORLD_TRANSFORM_PATH.relative_to(BASE_DIR)} "
        f"(RMS={captured.rms_reprojection:.3f} px)."
    )
