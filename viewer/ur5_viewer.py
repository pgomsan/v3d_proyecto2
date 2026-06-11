"""Visor 3D del UR5 con Robotics Toolbox y Swift.

Dos modos:
  - ``launch_ur5_viewer``: pose fija hardcodeada para validar el visor.
  - ``run_ur5_pose_follower``: lee ``state/last_pose.json`` en bucle y mueve
    el robot para imitar la herramienta real.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import roboticstoolbox as rtb
    import spatialgeometry as sg
    import swift
    from spatialmath import SE3
except ModuleNotFoundError:
    np = None
    rtb = None
    sg = None
    swift = None
    SE3 = None


BASE_DIR = Path(__file__).resolve().parent.parent
LAST_POSE_PATH = BASE_DIR / "state" / "last_pose.json"

# Pose codo-arriba sensata para el UR5: con esta configuracion inicial el
# warm-start de la IK iterativa tiende a mantener el codo arriba en lugar de
# encontrar soluciones donde el brazo se mete bajo la mesa.
UR5_ELBOW_UP_Q = (0.0, -1.5707963267948966, 1.5707963267948966,
                  -1.5707963267948966, -1.5707963267948966, 0.0)
FLOOR_TOLERANCE_M = 0.05


def _build_scene(
    env: Any, table_height_m: float
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Anade mesa, UR5, marcadores objetivo/TCP y replicas A/B/C.

    Convencion: Z=0 mundo coincide con la superficie de la mesa real (donde
    se captura el origen del tablero). La mesa virtual se dibuja por debajo
    (Z negativa) y el UR5 se monta justo en Z=0.

    Devuelve ``(ur5, target_sphere, tcp_sphere, tool_markers)`` donde
    ``tool_markers`` es un dict con las tres esferas de colores que replican
    la herramienta real para depurar visualmente la orientacion.
    """
    table = sg.Cuboid(
        scale=[1.0, 1.0, table_height_m],
        pose=SE3(0.0, 0.0, -table_height_m / 2.0),
        color=(0.55, 0.4, 0.25, 1.0),
    )
    env.add(table)

    ur5 = rtb.models.UR5()
    ur5.base = SE3(0.0, 0.0, 0.0)
    ur5.q = list(UR5_ELBOW_UP_Q)
    env.add(ur5)

    target_sphere = sg.Sphere(
        radius=0.02,
        pose=SE3(0.0, 0.0, 0.2),
        color=(1.0, 0.0, 0.0, 1.0),
    )
    env.add(target_sphere)

    tcp_sphere = sg.Sphere(
        radius=0.016,
        pose=SE3(0.0, 0.0, 0.2),
        color=(0.0, 0.55, 1.0, 1.0),
    )
    env.add(tcp_sphere)

    # Ejes XYZ atados al TCP del UR5 para visualizar su orientacion.
    # Sin esto la esfera azul es simetrica y no se aprecia si la IK rota
    # el flange cuando la herramienta gira alrededor del eje principal.
    axis_length = 0.08
    axis_radius = 0.004
    tcp_axes = {
        "x": sg.Cylinder(
            radius=axis_radius, length=axis_length, color=(1.0, 0.1, 0.1, 1.0)
        ),
        "y": sg.Cylinder(
            radius=axis_radius, length=axis_length, color=(0.1, 1.0, 0.1, 1.0)
        ),
        "z": sg.Cylinder(
            radius=axis_radius, length=axis_length, color=(0.2, 0.4, 1.0, 1.0)
        ),
    }
    for axis in tcp_axes.values():
        env.add(axis)

    # Offsets locales de cada eje del TCP en su pose canonica (cilindro va
    # en +Z por defecto). Se aplican en cada actualizacion del TCP.
    # Replicas de las marcas reales (rosa A, verde B, amarillo C). Permiten
    # ver de un vistazo si la orientacion del amarillo esta siendo seguida.
    marker_a = sg.Sphere(
        radius=0.014, pose=SE3(0.0, 0.0, 0.2), color=(1.0, 0.4, 0.7, 1.0)
    )
    marker_b = sg.Sphere(
        radius=0.014, pose=SE3(0.0, 0.0, 0.2), color=(0.2, 0.85, 0.2, 1.0)
    )
    marker_c = sg.Sphere(
        radius=0.014, pose=SE3(0.0, 0.0, 0.2), color=(1.0, 0.95, 0.1, 1.0)
    )
    env.add(marker_a)
    env.add(marker_b)
    env.add(marker_c)
    tool_markers = {
        "a": marker_a,
        "b": marker_b,
        "c": marker_c,
        "tcp_axes": tcp_axes,
        "tcp_axes_length": axis_length,
    }

    # Vista inicial: desde delante-derecha-arriba mirando al area de trabajo
    try:
        env.set_camera_pose([1.0, -1.0, 0.8], [0.0, 0.0, 0.2])
    except Exception:
        pass

    return ur5, target_sphere, tcp_sphere, tool_markers


def _se3_from_position_direction(
    position_m: tuple[float, float, float],
    direction: tuple[float, float, float],
    aligned_axis: str = "z",
) -> Any:
    """Construye SE3 con el eje ``aligned_axis`` del TCP alineado a ``direction``.

    ``aligned_axis`` puede ser "x", "y" o "z". El eje elegido apunta a lo
    largo del bisturi real. Los otros dos se construyen ortogonales y
    estables (perpendiculares al Z mundo cuando es posible).
    """
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return SE3(*position_m)
    d = d / norm

    z_world = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, z_world)) > 0.999:
        helper = np.array([1.0, 0.0, 0.0])
        perp1 = helper - np.dot(helper, d) * d
        perp1 = perp1 / np.linalg.norm(perp1)
    else:
        perp1 = np.cross(z_world, d)
        perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(d, perp1)

    # Tras la construccion, perp2 = cross(d, perp1). Para que las columnas
    # formen un sistema diestro (det = +1), perp2 debe ir negado cuando el
    # eje alineado es Y, por la regla de la mano derecha.
    axis = aligned_axis.lower() if isinstance(aligned_axis, str) else "z"
    if axis == "x":
        R = np.column_stack([d, perp1, perp2])
    elif axis == "y":
        R = np.column_stack([perp1, d, -perp2])
    else:
        R = np.column_stack([perp1, perp2, d])
    return SE3.Rt(R, list(position_m))


def _orthonormalize_rotation(rotation: Any) -> Any:
    """Proyecta una matriz aproximada sobre SO(3)."""
    u_matrix, _, vt_matrix = np.linalg.svd(np.asarray(rotation, dtype=float))
    result = u_matrix @ vt_matrix
    if np.linalg.det(result) < 0.0:
        u_matrix[:, -1] *= -1.0
        result = u_matrix @ vt_matrix
    return result


def _nearest_equivalent_joints(target: Any, current: Any) -> Any:
    """Escoge para cada articulacion el angulo 2*pi equivalente mas cercano."""
    target_array = np.asarray(target, dtype=float)
    current_array = np.asarray(current, dtype=float)
    delta = (target_array - current_array + math.pi) % (2.0 * math.pi) - math.pi
    return current_array + delta


def _step_joints_towards(
    current: Any,
    target: Any,
    max_step_rad: float,
) -> Any:
    """Limita el cambio articular por frame para evitar saltos visuales."""
    current_array = np.asarray(current, dtype=float)
    target_array = _nearest_equivalent_joints(target, current_array)
    step = np.clip(
        target_array - current_array,
        -abs(float(max_step_rad)),
        abs(float(max_step_rad)),
    )
    return current_array + step


def _rotation_matrix_from_payload(payload: dict[str, Any]) -> Any | None:
    orientation = payload.get("orientation")
    if not isinstance(orientation, dict):
        return None
    if orientation.get("format") != "rotation_matrix":
        return None
    value = orientation.get("value")
    try:
        rotation = np.asarray(value, dtype=float).reshape(3, 3)
    except (TypeError, ValueError):
        return None
    return _orthonormalize_rotation(rotation)


def _map_tool_orientation_to_tcp(
    tool_rotation: Any,
    aligned_axis: str,
    axis_flip: bool,
) -> Any:
    """Mapea el frame de la herramienta al eje configurable del TCP."""
    x_axis = tool_rotation[:, 0]
    y_axis = tool_rotation[:, 1]
    z_axis = tool_rotation[:, 2]
    axis = aligned_axis.lower() if isinstance(aligned_axis, str) else "x"

    if axis == "y":
        tcp_rotation = np.column_stack([y_axis, x_axis, -z_axis])
        flip_rotation = np.diag([1.0, -1.0, -1.0])
    elif axis == "z":
        tcp_rotation = np.column_stack([y_axis, z_axis, x_axis])
        flip_rotation = np.diag([-1.0, 1.0, -1.0])
    else:
        tcp_rotation = np.column_stack([x_axis, y_axis, z_axis])
        flip_rotation = np.diag([-1.0, 1.0, -1.0])

    if axis_flip:
        tcp_rotation = tcp_rotation @ flip_rotation
    return _orthonormalize_rotation(tcp_rotation)


def _target_pose_from_payload(
    payload: dict[str, Any],
    world_transform: Any,
    aligned_axis: str,
    axis_flip: bool,
) -> Any | None:
    position_cm = payload.get("position_cm")
    direction = payload.get("direction")
    if not position_cm:
        return None

    position_cm = tuple(float(value) for value in position_cm[:3])
    tool_rotation = _rotation_matrix_from_payload(payload)
    if world_transform is not None:
        world_position = world_transform.transform_point_cm(position_cm)
        if tool_rotation is not None:
            tool_rotation = world_transform.rotation @ tool_rotation
    else:
        world_position = position_cm

    position_m = tuple(value / 100.0 for value in world_position)
    if tool_rotation is not None:
        tcp_rotation = _map_tool_orientation_to_tcp(
            tool_rotation, aligned_axis, axis_flip
        )
        return SE3.Rt(tcp_rotation, list(position_m))

    if not direction:
        return SE3(*position_m)
    world_direction = tuple(float(value) for value in direction[:3])
    if world_transform is not None:
        world_direction = world_transform.transform_direction(world_direction)
    if axis_flip:
        world_direction = tuple(-component for component in world_direction)
    return _se3_from_position_direction(
        position_m, world_direction, aligned_axis
    )


def _respects_floor_standalone(ur5: Any, q: Any) -> bool:
    try:
        poses = ur5.fkine_all(q)
    except Exception:
        return True
    for pose in poses:
        if float(pose.t[2]) < -FLOOR_TOLERANCE_M:
            return False
    return True


def _solve_follower_ik(ur5: Any, target_in_base: Any) -> Any | None:
    """IK con orientacion completa para ``run_ur5_pose_follower``.

    Warm-start desde el ``q`` actual; sin fallback a solo posicion para
    garantizar que la orientacion de la herramienta (vector A-B) se
    respeta siempre.
    """
    try:
        solution = ur5.ikine_LM(target_in_base, q0=ur5.q)
    except Exception:
        return None
    if not solution.success:
        return None
    q_candidate = _nearest_equivalent_joints(solution.q, ur5.q)
    if _respects_floor_standalone(ur5, q_candidate):
        return q_candidate
    return None


def launch_ur5_viewer(
    target_position_cm: tuple[float, float, float] = (40.0, 0.0, 20.0),
    table_height_cm: float = 60.0,
) -> None:
    """Abre el visor y resuelve IK hacia ``target_position_cm`` (cm).

    El UR5 se monta en Z=0 (superficie de la mesa) y la mesa se dibuja por
    debajo. Las coordenadas del target estan en frame mundo, con Z hacia
    arriba desde la superficie de la mesa real.
    """
    if rtb is None or swift is None:
        print(
            "roboticstoolbox-python no esta instalado. Activa el venv y "
            "ejecuta 'pip install -r requirements.txt'."
        )
        return

    target_position_m = tuple(value / 100.0 for value in target_position_cm)
    table_height_m = table_height_cm / 100.0

    env = swift.Swift()
    env.launch(realtime=True)
    ur5, target_sphere, tcp_sphere, _tool_markers = _build_scene(env, table_height_m)
    target_sphere.T = SE3(*target_position_m)
    if tcp_sphere is not None:
        tcp_sphere.T = SE3(*target_position_m)

    target_world = SE3(*target_position_m)
    target_in_base = ur5.base.inv() * target_world
    solution = ur5.ikine_LM(target_in_base)
    if not solution.success:
        print(
            f"IK no convergio para target {target_position_cm} cm. "
            "Prueba con una pose dentro del alcance del UR5 (~85 cm) "
            "medido desde la base del robot."
        )
        return

    ur5.q = solution.q

    tcp_pose = ur5.fkine(ur5.q)
    tcp_cm = tuple(value * 100.0 for value in tcp_pose.t)
    print(
        f"\nTarget (cm): ({target_position_cm[0]:.2f}, "
        f"{target_position_cm[1]:.2f}, {target_position_cm[2]:.2f})\n"
        f"TCP resuelto (cm): ({tcp_cm[0]:.2f}, {tcp_cm[1]:.2f}, {tcp_cm[2]:.2f})\n"
        "Cierra la pestana de Swift para volver al menu."
    )

    env.hold()


class UR5Visualizer:
    """Encapsula el visor Swift + UR5 para llamarse desde otra pipeline.

    Uso tipico::

        viz = UR5Visualizer()
        viz.launch()
        try:
            for payload in stream_of_payloads():
                viz.update_from_payload(payload)
                viz.step()
        finally:
            viz.close()
    """

    def __init__(
        self,
        table_height_cm: float = 60.0,
        world_transform: Any = None,
        smoothing_alpha: float = 1.0,
        tcp_aligned_axis: str = "z",
        tcp_axis_flip: bool = False,
        max_joint_speed_deg_s: float = 720.0,
        marker_b_local_cm: tuple[float, float, float] | None = None,
        marker_c_local_cm: tuple[float, float, float] | None = None,
    ) -> None:
        self.table_height_m = table_height_cm / 100.0
        self.dt = 1.0 / 30.0
        self.env: Any = None
        self.ur5: Any = None
        self.target_sphere: Any = None
        self.tcp_sphere: Any = None
        self.tool_markers: dict[str, Any] = {}
        # Coordenadas locales de las marcas en el frame de la herramienta
        # (A en el origen). Sirven para dibujar replicas en el visor.
        self.marker_b_local_cm = marker_b_local_cm
        self.marker_c_local_cm = marker_c_local_cm
        self.world_transform = world_transform
        # alpha en [0, 1]: 1.0 = sin suavizado (raw), valores bajos = mas suave
        # pero con mas lag. 0.15 reduce el ruido amplificado por world_scale.
        self.smoothing_alpha = max(0.0, min(1.0, smoothing_alpha))
        self.tcp_aligned_axis = tcp_aligned_axis
        self.tcp_axis_flip = tcp_axis_flip
        self.max_joint_speed_rad_s = math.radians(
            max(1.0, float(max_joint_speed_deg_s))
        )
        self._target_q: Any = None
        self._smoothed_position_cm: tuple[float, float, float] | None = None
        self._smoothed_direction: tuple[float, float, float] | None = None
        self._smoothed_rotation: Any = None

    def launch(self) -> bool:
        if rtb is None or swift is None:
            print(
                "roboticstoolbox-python no esta instalado. No se puede "
                "abrir el visor del UR5."
            )
            return False
        self.env = swift.Swift()
        self.env.launch(realtime=True)
        (
            self.ur5,
            self.target_sphere,
            self.tcp_sphere,
            self.tool_markers,
        ) = _build_scene(self.env, self.table_height_m)
        # Coloca blue + ejes sobre el flange real desde el primer frame,
        # antes de que llegue ningun payload.
        self._update_tcp_marker()
        return True

    def update_from_payload(self, payload: dict[str, Any]) -> bool:
        """Actualiza el objetivo articular a partir de ``payload``.

        El movimiento se integra despues en cada llamada a :meth:`step`, de
        modo que sigue siendo continuo aunque falten detecciones intermedias.
        """
        if self.ur5 is None:
            return False
        position_cm = payload.get("position_cm")
        direction = payload.get("direction")
        rotation = _rotation_matrix_from_payload(payload)
        if not position_cm or (not direction and rotation is None):
            return False

        position_cm = tuple(float(value) for value in position_cm[:3])
        if direction:
            direction = tuple(float(value) for value in direction[:3])

        alpha = self.smoothing_alpha
        if self._smoothed_position_cm is None or alpha >= 1.0:
            smoothed_pos = position_cm
            smoothed_dir = direction
            smoothed_rotation = rotation
        else:
            smoothed_pos = tuple(
                (1.0 - alpha) * old + alpha * new
                for old, new in zip(self._smoothed_position_cm, position_cm)
            )
            if direction and self._smoothed_direction:
                mixed = tuple(
                    (1.0 - alpha) * old + alpha * new
                    for old, new in zip(self._smoothed_direction, direction)
                )
                norm = math.sqrt(sum(component * component for component in mixed))
                smoothed_dir = (
                    tuple(component / norm for component in mixed)
                    if norm > 1e-9
                    else direction
                )
            else:
                smoothed_dir = direction

            if rotation is not None and self._smoothed_rotation is not None:
                smoothed_rotation = _orthonormalize_rotation(
                    (1.0 - alpha) * self._smoothed_rotation + alpha * rotation
                )
            else:
                smoothed_rotation = rotation

        self._smoothed_position_cm = smoothed_pos
        self._smoothed_direction = smoothed_dir
        self._smoothed_rotation = smoothed_rotation

        if self.world_transform is not None:
            world_pos = self.world_transform.transform_point_cm(smoothed_pos)
            world_dir = (
                self.world_transform.transform_direction(smoothed_dir)
                if smoothed_dir
                else None
            )
            if smoothed_rotation is not None:
                smoothed_rotation = (
                    self.world_transform.rotation @ smoothed_rotation
                )
        else:
            world_pos = smoothed_pos
            world_dir = smoothed_dir

        position_m = tuple(value / 100.0 for value in world_pos)
        # Clamp el target al workspace alcanzable del UR5. Si pides una pose
        # fuera del alcance, la IK falla y el robot se queda quieto -> rojo
        # flotando lejos del flange. Proyectando el target a la frontera, el
        # robot siempre llega a lo mas cerca posible y rojo+blue se mantienen
        # juntos.
        position_m = self._clamp_position_to_workspace(position_m)
        if smoothed_rotation is not None:
            tcp_rotation = _map_tool_orientation_to_tcp(
                smoothed_rotation,
                self.tcp_aligned_axis,
                self.tcp_axis_flip,
            )
            target_world = SE3.Rt(tcp_rotation, list(position_m))
        else:
            oriented_dir = (
                tuple(-component for component in world_dir)
                if self.tcp_axis_flip
                else world_dir
            )
            target_world = _se3_from_position_direction(
                position_m, oriented_dir, self.tcp_aligned_axis
            )
        target_in_base = self.ur5.base.inv() * target_world
        target_q = self._solve_ik_robust(target_in_base)
        if target_q is None:
            # IK no encontro nada. Mantenemos el target anterior para que el
            # robot siga viajando hacia la ultima pose buena (no lo congelamos
            # ni movemos los marcadores; asi rojo/A/B/C/blue siempre coinciden).
            return False

        # Mapeo directo: lo que ve la IK es lo que va al robot. Sin EMA en
        # espacio q ni rechazo de salto de ramas: la fluidez visual viene del
        # rate limit articular en step(), y la limpieza de poses ya la hace
        # TemporalPoseFilter aguas arriba.
        self._target_q = target_q

        self.target_sphere.T = SE3(*position_m)
        self._update_tool_markers(position_m, smoothed_rotation)
        return True

    def _update_tcp_marker(self) -> None:
        if self.ur5 is None or self.tcp_sphere is None:
            return
        tcp_pose = self.ur5.fkine(self.ur5.q)
        self.tcp_sphere.T = SE3(*tcp_pose.t)

        tcp_axes = self.tool_markers.get("tcp_axes") if self.tool_markers else None
        if not tcp_axes:
            return
        length = float(self.tool_markers.get("tcp_axes_length", 0.08))
        half = length / 2.0
        # El cilindro por defecto va en +Z centrado en el origen. Para que
        # cada eje arranque en el TCP y apunte en +X / +Y / +Z del flange,
        # se compone una rotacion local + desplazamiento de half.
        local_poses = {
            "x": SE3(half, 0.0, 0.0) * SE3.Ry(math.pi / 2.0),
            "y": SE3(0.0, half, 0.0) * SE3.Rx(-math.pi / 2.0),
            "z": SE3(0.0, 0.0, half),
        }
        for axis_key, axis_geom in tcp_axes.items():
            axis_geom.T = tcp_pose * local_poses[axis_key]

    def _update_tool_markers(
        self,
        marker_a_world_m: tuple[float, float, float],
        world_rotation: Any,
    ) -> None:
        """Coloca las esferas A/B/C en sus posiciones del frame mundo."""
        if not self.tool_markers:
            return
        marker_a = self.tool_markers.get("a")
        if marker_a is not None:
            marker_a.T = SE3(*marker_a_world_m)

        if world_rotation is None:
            return
        a_array = np.asarray(marker_a_world_m, dtype=float)
        for key, local_cm in (
            ("b", self.marker_b_local_cm),
            ("c", self.marker_c_local_cm),
        ):
            marker = self.tool_markers.get(key)
            if marker is None or local_cm is None:
                continue
            local_m = np.asarray(local_cm, dtype=float) / 100.0
            position = a_array + world_rotation @ local_m
            marker.T = SE3(*position.tolist())

    def _try_ik(
        self,
        target_in_base: Any,
        q_seed: Any,
        mask: tuple[int, int, int, int, int, int] | None = None,
        max_joint_delta_rad: float | None = None,
    ) -> Any | None:
        """Intenta una IK con la semilla dada y filtra suelo + saltos articulares.

        ``max_joint_delta_rad`` evita aceptar soluciones de otra rama IK que
        provocarian teleportes; ``None`` = sin limite (uso solo en casos donde
        la continuidad ya no se puede preservar).
        """
        kwargs: dict[str, Any] = {"q0": q_seed}
        if mask is not None:
            kwargs["mask"] = list(mask)
        try:
            solution = self.ur5.ikine_LM(target_in_base, **kwargs)
        except Exception:
            return None
        if not solution.success:
            return None
        candidate_q = _nearest_equivalent_joints(solution.q, self.ur5.q)
        if not self._respects_floor(candidate_q):
            return None
        if max_joint_delta_rad is not None:
            delta = np.max(np.abs(candidate_q - np.asarray(self.ur5.q)))
            if delta > max_joint_delta_rad:
                return None
        return candidate_q

    def _solve_ik_robust(self, target_in_base: Any) -> Any | None:
        """IK con orientacion completa: respeta posicion + orientacion (A-B).

        Warm-start con ``q`` actual para preservar continuidad. Si no
        converge devuelve ``None`` y el visor mantiene la ultima pose valida.
        Sin fallback a solo posicion: el usuario quiere que la orientacion
        de la herramienta (vector A-B) se respete siempre.
        """
        return self._try_ik(target_in_base, self.ur5.q)

    def _clamp_position_to_workspace(
        self, position_m: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Proyecta el target a un workspace alcanzable por el UR5.

        El UR5 alcanza ~85 cm. Usamos 78 cm como radio efectivo (margen para
        que la IK encuentre soluciones limpias sin forzar singularidades de
        codo extendido). Si el target esta mas lejos de la base, se proyecta
        a la frontera por la direccion base->target. Z minimo positivo para
        no pedir poses pegadas a la mesa.
        """
        if self.ur5 is None:
            return position_m
        base = np.asarray(self.ur5.base.t, dtype=float)
        target = np.asarray(position_m, dtype=float)
        delta = target - base
        distance = float(np.linalg.norm(delta))
        max_reach_m = 0.78
        if distance > max_reach_m:
            target = base + delta * (max_reach_m / distance)
        min_z_m = 0.03
        if target[2] < min_z_m:
            target[2] = min_z_m
        return (float(target[0]), float(target[1]), float(target[2]))

    def _respects_floor(self, q: Any) -> bool:
        """Devuelve True si ningun link cae bajo el suelo (Z < -tolerance)."""
        if self.ur5 is None:
            return True
        try:
            poses = self.ur5.fkine_all(q)
        except Exception:
            return True
        for pose in poses:
            if float(pose.t[2]) < -FLOOR_TOLERANCE_M:
                return False
        return True

    def move_to_named_pose(self, joint_degrees: Any) -> bool:
        """Fija una pose articular objetivo fija (en grados).

        El robot se desplazara hacia ella en las siguientes llamadas a
        :meth:`step` a velocidad limitada. Mientras no se reanude el
        seguimiento, el robot se queda en esa pose. Devuelve si se acepto.
        """
        if self.ur5 is None or joint_degrees is None:
            return False
        target_q = np.radians(np.asarray(joint_degrees, dtype=float))
        if target_q.shape != np.asarray(self.ur5.q).shape:
            print(
                "Aviso: la pose de gesto no tiene el numero de articulaciones "
                f"del UR5 ({np.asarray(self.ur5.q).shape[0]})."
            )
            return False
        self._target_q = target_q
        return True

    def step(self) -> None:
        if self.ur5 is not None and self._target_q is not None:
            max_step_rad = self.max_joint_speed_rad_s * self.dt
            next_q = _step_joints_towards(
                self.ur5.q,
                self._target_q,
                max_step_rad,
            )
            if self._respects_floor(next_q):
                self.ur5.q = next_q
        # Sincronizar siempre blue + ejes con la q ACTUAL del robot. Si lo
        # dejamos dentro del if anterior, blue queda en su pose inicial
        # cuando no hay target o la comprobacion de suelo descarta el paso,
        # y se ve la esfera separada del flange.
        if self.ur5 is not None:
            self._update_tcp_marker()
        if self.env is not None:
            self.env.step(self.dt)

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
        self.tcp_sphere = None


def cycle_tcp_aligned_axis() -> None:
    """Cicla x -> y -> z -> x para alinear el TCP con la herramienta real.

    Util cuando el bisturi virtual sale rotado 90 grados respecto al real.
    """
    from app_state import load_config, save_config

    config = load_config()
    tool_section = config.setdefault("tool", {})
    current = str(tool_section.get("tcp_aligned_axis", "z")).lower()
    next_axis = {"x": "y", "y": "z", "z": "x"}.get(current, "z")
    tool_section["tcp_aligned_axis"] = next_axis
    save_config(config)
    print(f"tcp_aligned_axis: {current} -> {next_axis}")


def toggle_tcp_axis_flip() -> None:
    """Invierte el sentido del eje TCP (cambia 'punta arriba' por 'abajo')."""
    from app_state import load_config, save_config

    config = load_config()
    tool_section = config.setdefault("tool", {})
    current = bool(tool_section.get("tcp_axis_flip", False))
    tool_section["tcp_axis_flip"] = not current
    save_config(config)
    print(f"tcp_axis_flip: {current} -> {not current}")


def run_ur5_pose_follower(
    table_height_cm: float = 60.0,
    update_rate_hz: float = 30.0,
) -> None:
    """Lee ``state/last_pose.json`` en bucle y mueve el UR5 para imitarla.

    Por ahora la pose se interpreta como si el frame de la camara izquierda
    rectificada coincidiera con el frame mundo del visor (Z arriba, origen
    en la mesa). La transformacion real camara->mesa se anadira cuando se
    capture el origen fisico con el tablero.
    """
    if rtb is None or swift is None:
        print(
            "roboticstoolbox-python no esta instalado. Activa el venv y "
            "ejecuta 'pip install -r requirements.txt'."
        )
        return

    if not LAST_POSE_PATH.exists():
        print(
            f"No existe {LAST_POSE_PATH}. Ejecuta primero la estimacion de "
            "pose o usa el modo 'pose fija' para validar el visor."
        )
        return

    from app_state import load_config
    from calibration.world_transform import load_world_transform

    config = load_config()
    tool_config = config.get("tool", {})
    tcp_aligned_axis = str(tool_config.get("tcp_aligned_axis", "x")).lower()
    tcp_axis_flip = bool(tool_config.get("tcp_axis_flip", False))
    world_transform = load_world_transform()

    table_height_m = table_height_cm / 100.0
    period_s = 1.0 / max(update_rate_hz, 1.0)

    env = swift.Swift()
    env.launch(realtime=True)
    ur5, target_sphere, tcp_sphere, _tool_markers = _build_scene(env, table_height_m)

    print(
        "\nVisor siguiendo last_pose.json. "
        "Cierra la pestana de Swift para volver al menu."
    )

    last_timestamp: str | None = None
    while True:
        try:
            with LAST_POSE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            time.sleep(period_s)
            env.step(period_s)
            continue

        timestamp = payload.get("timestamp")
        if timestamp != last_timestamp:
            position_cm = payload.get("position_cm")
            if position_cm:
                target_world = _target_pose_from_payload(
                    payload,
                    world_transform,
                    tcp_aligned_axis,
                    tcp_axis_flip,
                )
                if target_world is None:
                    last_timestamp = timestamp
                    continue
                target_in_base = ur5.base.inv() * target_world
                solved_q = _solve_follower_ik(ur5, target_in_base)
                if solved_q is not None:
                    ur5.q = solved_q
                    target_sphere.T = SE3(*target_world.t)
                    if tcp_sphere is not None:
                        tcp_pose = ur5.fkine(ur5.q)
                        tcp_sphere.T = SE3(*tcp_pose.t)
                else:
                    print(
                        f"IK no convergio para pose {position_cm} cm "
                        "(fuera de alcance o singularidad)."
                    )
            last_timestamp = timestamp

        env.step(period_s)
        time.sleep(period_s)
