from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from app_state import load_config, save_last_pose
from pose.geometry import distance_3d
from pose.rod_pose import (
    RodPoseEstimator,
    ToolMarkerTriplet3D,
    ToolParameters,
)
from pose.tracking import TemporalPoseFilter
from vision.camera import (
    CameraSource,
    load_stereo_camera_configs,
    read_stereo_pair,
)
from vision.color_markers import ColorRange, detect_marker
from vision.frame_debug import draw_marker_debug, show_debug_frame
from vision.stereo import (
    epipolar_errors_are_valid,
    epipolar_errors_px,
    load_stereo_calibration,
)

try:
    import cv2 as cv
except ModuleNotFoundError:
    cv = None


def _color_range_from_config(config: dict[str, Any], prefix: str) -> ColorRange:
    color_config = config.get("color_detection", {})
    lower = color_config.get(f"{prefix}_hsv_lower", [0, 0, 0])
    upper = color_config.get(f"{prefix}_hsv_upper", [0, 0, 0])
    return ColorRange(
        lower=tuple(int(value) for value in lower),
        upper=tuple(int(value) for value in upper),
    )


def _draw_marker_geometry(
    frame: Any,
    marker_a: Any,
    marker_b: Any,
    marker_c: Any,
) -> Any:
    if cv is None:
        return frame

    points = {}
    for name, marker in (("A", marker_a), ("B", marker_b), ("C", marker_c)):
        if marker is not None:
            points[name] = tuple(
                int(round(value)) for value in marker.center_px
            )

    for first, second, color in (
        ("A", "B", (0, 255, 255)),
        ("A", "C", (255, 255, 0)),
        ("B", "C", (255, 255, 0)),
    ):
        if first not in points or second not in points:
            continue
        start = points[first]
        end = points[second]
        cv.line(frame, start, end, color, 2)
        cv.putText(
            frame,
            f"{first}-{second}",
            ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2 - 8),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv.LINE_AA,
        )
    return frame


def _draw_tip_projection(
    frame: Any,
    tip_px: tuple[float, float] | None,
    center_px: tuple[float, float] | None = None,
) -> Any:
    """Dibuja la punta estimada sobre la imagen rectificada."""
    if cv is None or tip_px is None:
        return frame

    tip = (int(round(tip_px[0])), int(round(tip_px[1])))
    color = (255, 0, 255)
    if center_px is not None:
        center = (int(round(center_px[0])), int(round(center_px[1])))
        cv.line(frame, center, tip, color, 2)
        cv.circle(frame, center, 5, (0, 255, 255), 1)

    cv.circle(frame, tip, 11, color, 3)
    cv.drawMarker(frame, tip, color, cv.MARKER_TILTED_CROSS, 28, 3)
    cv.putText(
        frame,
        "punta",
        (tip[0] + 12, tip[1] - 12),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        frame,
        "punta",
        (tip[0] + 12, tip[1] - 12),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv.LINE_AA,
    )
    return frame


def _draw_pose_overlay(
    frame: Any,
    marker_a_3d: tuple[float, float, float] | None,
    marker_b_3d: tuple[float, float, float] | None,
    marker_c_3d: tuple[float, float, float] | None,
    expected_distances_cm: dict[str, float],
    stereo_rms: float,
    tracking_status: str,
    tracking_reason: str,
    epipolar_errors: dict[str, float],
    max_epipolar_error_px: float,
    processing_fps: float,
    center_cm: tuple[float, float, float] | None = None,
    tip_position_cm: tuple[float, float, float] | None = None,
) -> Any:
    if cv is None:
        return frame

    if tracking_status == "VALID":
        color = (0, 255, 0)
    elif tracking_status == "LOST":
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    epipolar_text = " ".join(
        f"{name.upper()}={error:.1f}"
        for name, error in sorted(epipolar_errors.items())
    )
    if not epipolar_text:
        epipolar_text = "sin correspondencias completas"

    lines = [
        f"Tracking: {tracking_status} | FPS: {processing_fps:.1f}",
        f"Epipolar: {epipolar_text} px | limite={max_epipolar_error_px:.1f}",
    ]
    if tracking_reason:
        lines.append(tracking_reason)

    if (
        marker_a_3d is not None
        and marker_b_3d is not None
        and marker_c_3d is not None
    ):
        distances = {
            "ab": distance_3d(marker_a_3d, marker_b_3d),
            "ac": distance_3d(marker_a_3d, marker_c_3d),
            "bc": distance_3d(marker_b_3d, marker_c_3d),
        }
        lines.extend(
            [
                "Distancias 3D "
                + " ".join(
                    f"{key.upper()}={distance:.2f}"
                    f"({distance - expected_distances_cm.get(key, distance):+.2f})cm"
                    for key, distance in distances.items()
                ),
                f"A=({marker_a_3d[0]:.1f},{marker_a_3d[1]:.1f},{marker_a_3d[2]:.1f})cm",
                f"B=({marker_b_3d[0]:.1f},{marker_b_3d[1]:.1f},{marker_b_3d[2]:.1f})cm",
                f"C=({marker_c_3d[0]:.1f},{marker_c_3d[1]:.1f},{marker_c_3d[2]:.1f})cm",
            ]
        )
        if center_cm is not None:
            lines.append(
                "Centro herramienta="
                f"({center_cm[0]:.1f},{center_cm[1]:.1f},{center_cm[2]:.1f})cm",
            )
        if tip_position_cm is not None:
            lines.append(
                "Punta estimada="
                f"({tip_position_cm[0]:.1f},{tip_position_cm[1]:.1f},{tip_position_cm[2]:.1f})cm",
            )
    lines.append(f"RMS estereo: {stereo_rms:.2f}")

    for row, text in enumerate(lines):
        y = 32 + row * 28
        cv.putText(
            frame,
            text,
            (20, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv.LINE_AA,
        )
        cv.putText(
            frame,
            text,
            (20, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv.LINE_AA,
        )
    return frame


def _smoothed_fps(
    previous_fps: float,
    previous_time: float,
    current_time: float,
    alpha: float,
) -> float:
    elapsed = current_time - previous_time
    if elapsed <= 0.0:
        return previous_fps
    instant_fps = 1.0 / elapsed
    if previous_fps <= 0.0:
        return instant_fps
    clamped_alpha = max(0.0, min(1.0, alpha))
    return (
        (1.0 - clamped_alpha) * previous_fps
        + clamped_alpha * instant_fps
    )


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _tool_marker_distances(config: dict[str, Any]) -> dict[str, float]:
    tool_config = config.get("tool", {})
    ab = _optional_float(tool_config.get("marker_distance_cm"))
    along = _optional_float(tool_config.get("marker_c_along_ab_cm"))
    offset = _optional_float(tool_config.get("marker_c_offset_cm"))
    if ab is None or along is None or offset is None:
        return {}
    return {
        "ab": ab,
        "ac": (along**2 + offset**2) ** 0.5,
        "bc": ((ab - along) ** 2 + offset**2) ** 0.5,
    }


def _tool_parameters_from_config(config: dict[str, Any]) -> ToolParameters:
    tool_config = config.get("tool", {})
    raw_tip_offset = tool_config.get("tip_offset_cm", [0.0, 0.0, 0.0])
    try:
        tip_offset_cm = tuple(float(value) for value in raw_tip_offset[:3])
    except (TypeError, ValueError):
        tip_offset_cm = (0.0, 0.0, 0.0)
    if len(tip_offset_cm) != 3:
        tip_offset_cm = (0.0, 0.0, 0.0)

    length_cm = _optional_float(tool_config.get("length_cm"))
    marker_distance_cm = _optional_float(tool_config.get("marker_distance_cm"))
    marker_c_along_ab_cm = _optional_float(
        tool_config.get("marker_c_along_ab_cm")
    )
    marker_c_offset_cm = _optional_float(tool_config.get("marker_c_offset_cm"))

    raw_tolerance = tool_config.get("marker_distance_tolerance_ratio", 0.50)
    try:
        tolerance_ratio = float(raw_tolerance)
    except (TypeError, ValueError):
        tolerance_ratio = 0.50

    return ToolParameters(
        tool_id=str(tool_config.get("tool_id", "tool_01")),
        tool_type=str(tool_config.get("tool_type", "tool")),
        length_cm=length_cm,
        marker_distance_cm=marker_distance_cm,
        tip_offset_cm=tip_offset_cm,
        marker_c_along_ab_cm=marker_c_along_ab_cm,
        marker_c_offset_cm=marker_c_offset_cm,
        marker_distance_tolerance_ratio=tolerance_ratio,
    )


def preview_marker_detection() -> None:
    """Previsualiza deteccion de marcas sin estimar pose.

    Deberia abrir las dos camaras, ejecutar solo `vision/color_markers.py` y
    dibujar resultados de depuracion. Util para ajustar HSV.
    """
    if cv is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    config = load_config()
    marker_a_range = _color_range_from_config(config, "marker_a")
    marker_b_range = _color_range_from_config(config, "marker_b")
    marker_c_range = _color_range_from_config(config, "marker_c")
    left_config, right_config = load_stereo_camera_configs()
    left_camera = CameraSource(left_config)
    right_camera = CameraSource(right_config)

    print("Previsualizando marcas. Pulsa q o Esc en una ventana para salir.")
    print(f"Marca A HSV: {marker_a_range.lower} -> {marker_a_range.upper}")
    print(f"Marca B HSV: {marker_b_range.lower} -> {marker_b_range.upper}")
    print(f"Marca C HSV: {marker_c_range.lower} -> {marker_c_range.upper}")

    try:
        left_camera.open()
        right_camera.open()

        while True:
            left_frame, right_frame = read_stereo_pair(left_camera, right_camera)

            left_a = detect_marker(left_frame, "A", marker_a_range)
            left_b = detect_marker(left_frame, "B", marker_b_range)
            left_c = detect_marker(left_frame, "C", marker_c_range)
            right_a = detect_marker(right_frame, "A", marker_a_range)
            right_b = detect_marker(right_frame, "B", marker_b_range)
            right_c = detect_marker(right_frame, "C", marker_c_range)

            left_debug = draw_marker_debug(left_frame, [left_a, left_b, left_c])
            right_debug = draw_marker_debug(
                right_frame, [right_a, right_b, right_c]
            )
            _draw_marker_geometry(left_debug, left_a, left_b, left_c)
            _draw_marker_geometry(right_debug, right_a, right_b, right_c)

            show_debug_frame(f"Marcas izquierda [{left_config.index}]", left_debug)
            show_debug_frame(f"Marcas derecha [{right_config.index}]", right_debug)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        left_camera.release()
        right_camera.release()
        cv.destroyAllWindows()


@dataclass
class PoseFrameResult:
    """Resultado del procesado de un par estereo (solo datos, sin GUI).

    Reune todo lo que el dibujado, el visor y la persistencia necesitan, de
    modo que el calculo (caro: rectificacion, deteccion, triangulacion) se
    pueda ejecutar en un hilo aparte del que pinta y muestra ventanas.
    """

    left_rect: Any
    right_rect: Any
    left_markers: tuple[Any, Any, Any]
    right_markers: tuple[Any, Any, Any]
    tracking_status: str
    tracking_reason: str
    epipolar_errors: dict[str, float]
    max_epipolar_error_px: float
    stereo_rms: float
    expected_distances_cm: dict[str, float]
    processing_fps: float
    tool_pixels: list[tuple[float, float]]
    marker_a_3d: tuple[float, float, float] | None = None
    marker_b_3d: tuple[float, float, float] | None = None
    marker_c_3d: tuple[float, float, float] | None = None
    pose_payload: dict[str, Any] | None = None
    center_3d: tuple[float, float, float] | None = None
    tip_3d: tuple[float, float, float] | None = None
    left_center_px: tuple[float, float] | None = None
    left_tip_px: tuple[float, float] | None = None
    right_center_px: tuple[float, float] | None = None
    right_tip_px: tuple[float, float] | None = None


class StereoPoseProcessor:
    """Procesa pares estereo a pose 3D. Solo calculo, sin ventanas ni disco.

    Encapsula los parametros de configuracion, la calibracion estereo, el
    estimador de pose y el filtro temporal. Lo comparten el bucle secuencial
    (`run_pose_estimation`) y la app multihilo (`main_app3d`), de modo que la
    logica de vision vive en un unico sitio.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.marker_a_range = _color_range_from_config(config, "marker_a")
        self.marker_b_range = _color_range_from_config(config, "marker_b")
        self.marker_c_range = _color_range_from_config(config, "marker_c")
        self.expected_distances_cm = _tool_marker_distances(config)
        tracking_config = config.get("tracking", {})

        def _as_float(key: str, default: float) -> float:
            try:
                return float(tracking_config.get(key, default))
            except (TypeError, ValueError):
                return default

        def _as_int(key: str, default: int) -> int:
            try:
                return int(tracking_config.get(key, default))
            except (TypeError, ValueError):
                return default

        self.max_epipolar_error_px = _as_float("max_epipolar_error_px", 25.0)
        pose_smoothing_alpha = _as_float("pose_smoothing_alpha", 0.15)
        max_position_jump_cm = _as_float("max_position_jump_cm", 5.0)
        max_orientation_jump_deg = _as_float("max_orientation_jump_deg", 35.0)
        temporal_reacquire_frames = _as_int("temporal_reacquire_frames", 3)

        self.pose_estimator = RodPoseEstimator(_tool_parameters_from_config(config))
        self.temporal_filter = TemporalPoseFilter(
            smoothing_alpha=pose_smoothing_alpha,
            max_position_jump_cm=max_position_jump_cm,
            max_orientation_jump_deg=max_orientation_jump_deg,
            reacquire_frames=temporal_reacquire_frames,
        )
        self.stereo = load_stereo_calibration()

    def process(
        self,
        left_frame: Any,
        right_frame: Any,
        processing_fps: float = 0.0,
    ) -> PoseFrameResult:
        """Procesa un par estereo y devuelve un :class:`PoseFrameResult`."""
        stereo = self.stereo
        left_rect, right_rect = stereo.rectify_pair(left_frame, right_frame)

        left_a = detect_marker(left_rect, "A", self.marker_a_range)
        left_b = detect_marker(left_rect, "B", self.marker_b_range)
        left_c = detect_marker(left_rect, "C", self.marker_c_range)
        right_a = detect_marker(right_rect, "A", self.marker_a_range)
        right_b = detect_marker(right_rect, "B", self.marker_b_range)
        right_c = detect_marker(right_rect, "C", self.marker_c_range)

        detections = {
            "a": (left_a, right_a),
            "b": (left_b, right_b),
            "c": (left_c, right_c),
        }
        correspondences = {
            name: (left.center_px, right.center_px)
            for name, (left, right) in detections.items()
            if left is not None and right is not None
        }
        epipolar_errors = epipolar_errors_px(correspondences)
        all_markers_detected = len(correspondences) == len(detections)
        epipolar_valid = epipolar_errors_are_valid(
            epipolar_errors, ("a", "b", "c"), self.max_epipolar_error_px
        )

        result = PoseFrameResult(
            left_rect=left_rect,
            right_rect=right_rect,
            left_markers=(left_a, left_b, left_c),
            right_markers=(right_a, right_b, right_c),
            tracking_status="LOST",
            tracking_reason="",
            epipolar_errors=epipolar_errors,
            max_epipolar_error_px=self.max_epipolar_error_px,
            stereo_rms=stereo.rms,
            expected_distances_cm=self.expected_distances_cm,
            processing_fps=processing_fps,
            tool_pixels=[
                detection.center_px
                for detection in (left_a, left_b, left_c)
                if detection is not None
            ],
        )

        missing_detections = [
            f"{name.upper()}-{side}"
            for name, (left, right) in detections.items()
            for side, detection in (("L", left), ("R", right))
            if detection is None
        ]
        result.tracking_reason = "Faltan: " + ", ".join(missing_detections)

        if all_markers_detected and not epipolar_valid:
            result.tracking_status = "REJECTED_EPIPOLAR"
            rejected = [
                f"{name.upper()}={error:.1f}px"
                for name, error in sorted(epipolar_errors.items())
                if error > self.max_epipolar_error_px
            ]
            result.tracking_reason = "Fuera de limite: " + ", ".join(rejected)
        elif all_markers_detected:
            try:
                result.marker_a_3d = stereo.triangulate_point(
                    left_a.center_px, right_a.center_px
                )
                result.marker_b_3d = stereo.triangulate_point(
                    left_b.center_px, right_b.center_px
                )
                result.marker_c_3d = stereo.triangulate_point(
                    left_c.center_px, right_c.center_px
                )
            except ValueError as exc:
                result.tracking_status = "REJECTED_TRIANGULATION"
                result.tracking_reason = str(exc)

        if (
            result.marker_a_3d is not None
            and result.marker_b_3d is not None
            and result.marker_c_3d is not None
        ):
            confidence = min(
                left_a.confidence,
                right_a.confidence,
                left_b.confidence,
                right_b.confidence,
                left_c.confidence,
                right_c.confidence,
            )
            marker_triplet_3d = ToolMarkerTriplet3D(
                marker_a_cm=result.marker_a_3d,
                marker_b_cm=result.marker_b_3d,
                marker_c_cm=result.marker_c_3d,
                confidence=confidence,
                marker_pixels={
                    "a_left_px": list(left_a.center_px),
                    "a_right_px": list(right_a.center_px),
                    "b_left_px": list(left_b.center_px),
                    "b_right_px": list(right_b.center_px),
                    "c_left_px": list(left_c.center_px),
                    "c_right_px": list(right_c.center_px),
                },
            )
            tool_pose = self.pose_estimator.estimate_from_markers(marker_triplet_3d)
            if tool_pose is not None:
                candidate_payload = self.pose_estimator.build_payload(
                    tool_pose, marker_triplet_3d
                )
                candidate_payload["stereo_rms"] = stereo.rms
                candidate_payload["epipolar_errors_px"] = epipolar_errors
                candidate_payload["max_epipolar_error_px"] = self.max_epipolar_error_px
                candidate_payload["processing_fps"] = processing_fps
                temporal_result = self.temporal_filter.filter(candidate_payload)
                if temporal_result.accepted:
                    result.tracking_status = "VALID"
                    result.tracking_reason = temporal_result.reason
                    payload = temporal_result.payload
                    assert payload is not None
                    payload["tracking_status"] = "VALID"
                    result.pose_payload = payload
                    result.center_3d = tuple(payload["markers"]["center_3d_cm"])
                    # La "punta" del overlay se dibuja sobre la marca B (verde)
                    # para que se vea el sentido del vector A->B. Solo visual.
                    result.tip_3d = tuple(payload["markers"]["b_3d_cm"])
                    result.left_center_px = stereo.project_left_point(result.center_3d)
                    result.left_tip_px = stereo.project_left_point(result.tip_3d)
                    result.right_center_px = stereo.project_right_point(result.center_3d)
                    result.right_tip_px = stereo.project_right_point(result.tip_3d)
                else:
                    result.tracking_status = "REJECTED_TEMPORAL"
                    result.tracking_reason = temporal_result.reason
            else:
                result.tracking_status = "REJECTED_GEOMETRY"
                result.tracking_reason = "Las distancias AB/AC/BC no coinciden"

        return result


def draw_pose_frame(result: PoseFrameResult) -> tuple[Any, Any]:
    """Dibuja las ventanas de depuracion izquierda y derecha de un resultado.

    Devuelve ``(left_debug, right_debug)`` listos para mostrar. No abre
    ventanas: el que llama decide si los muestra (hilo principal).
    """
    left_a, left_b, left_c = result.left_markers
    right_a, right_b, right_c = result.right_markers
    left_debug = draw_marker_debug(result.left_rect, [left_a, left_b, left_c])
    right_debug = draw_marker_debug(result.right_rect, [right_a, right_b, right_c])
    _draw_marker_geometry(left_debug, left_a, left_b, left_c)
    _draw_marker_geometry(right_debug, right_a, right_b, right_c)
    _draw_tip_projection(left_debug, result.left_tip_px, result.left_center_px)
    _draw_tip_projection(right_debug, result.right_tip_px, result.right_center_px)
    for debug in (left_debug, right_debug):
        _draw_pose_overlay(
            debug,
            result.marker_a_3d,
            result.marker_b_3d,
            result.marker_c_3d,
            result.expected_distances_cm,
            result.stereo_rms,
            result.tracking_status,
            result.tracking_reason,
            result.epipolar_errors,
            result.max_epipolar_error_px,
            result.processing_fps,
            center_cm=result.center_3d,
            tip_position_cm=result.tip_3d,
        )
    return left_debug, right_debug


def run_pose_estimation(
    on_payload: Callable[[dict[str, Any]], None] | None = None,
    on_tick: Callable[[], None] | None = None,
    on_frame: Callable[[Any, list[tuple[float, float]]], None] | None = None,
) -> None:
    """Bucle principal de estimacion de pose.

    Argumentos opcionales:
      - ``on_payload`` se llama con el payload cada vez que hay pose nueva,
        antes de guardarlo en disco. Util para enchufar un visor que tenga
        que reaccionar en vivo.
      - ``on_tick`` se llama una vez por iteracion del bucle (haya pose o no).
        Util para refrescar el visor aunque la deteccion falle un frame.
      - ``on_frame`` se llama una vez por iteracion con el frame izquierdo de
        depuracion (tras dibujar la pose) y la lista de pixeles de los
        marcadores detectados en esa imagen. Permite detectar gestos sobre la
        misma camara, ignorando la mano que sostiene la herramienta.
    """
    if cv is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    config = load_config()
    tracking_config = config.get("tracking", {})
    try:
        fps_smoothing_alpha = float(
            tracking_config.get("fps_smoothing_alpha", 0.15)
        )
    except (TypeError, ValueError):
        fps_smoothing_alpha = 0.15

    processor = StereoPoseProcessor(config)
    stereo = processor.stereo
    left_config, right_config = load_stereo_camera_configs()
    left_camera = CameraSource(left_config)
    right_camera = CameraSource(right_config)
    last_saved_at = 0.0
    previous_frame_at = time.monotonic()
    processing_fps = 0.0

    print("Estimando pose 3D. Pulsa q o Esc en una ventana para salir.")
    print(f"RMS estereo cargado: {stereo.rms:.3f}")
    print(f"Limite epipolar: {processor.max_epipolar_error_px:.1f} px")
    if stereo.rms > 5.0:
        print("Aviso: RMS estereo alto. La triangulacion puede ser poco fiable.")

    try:
        left_camera.open()
        right_camera.open()

        while True:
            left_frame, right_frame = read_stereo_pair(left_camera, right_camera)
            current_frame_at = time.monotonic()
            processing_fps = _smoothed_fps(
                processing_fps,
                previous_frame_at,
                current_frame_at,
                fps_smoothing_alpha,
            )
            previous_frame_at = current_frame_at

            result = processor.process(left_frame, right_frame, processing_fps)

            if result.pose_payload is not None and on_payload is not None:
                on_payload(result.pose_payload)
            if (
                result.pose_payload is not None
                and current_frame_at - last_saved_at >= 0.5
            ):
                save_last_pose(result.pose_payload)
                last_saved_at = current_frame_at

            left_debug, right_debug = draw_pose_frame(result)

            if on_frame is not None:
                on_frame(left_debug, result.tool_pixels)

            show_debug_frame(f"Pose izquierda rectificada [{left_config.index}]", left_debug)
            show_debug_frame(f"Pose derecha rectificada [{right_config.index}]", right_debug)

            if on_tick is not None:
                on_tick()

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        left_camera.release()
        right_camera.release()
        cv.destroyAllWindows()


def _build_ur5_visualizer(config: dict[str, Any]) -> Any:
    """Construye el visor UR5 a partir de la configuracion y el origen.

    Carga la transformada de mundo, lee los parametros de suavizado, eje TCP
    y geometria de marcas, e instancia ``UR5Visualizer`` sin lanzarlo todavia.
    Lo comparten el modo solo-visor y el modo pose + gestos.
    """
    from calibration.world_transform import load_world_transform
    from viewer.ur5_viewer import UR5Visualizer

    world_transform = load_world_transform()
    if world_transform is None:
        print(
            "Aviso: no se ha capturado el origen del mundo. Las poses se "
            "interpretan en frame camara izquierda (puede no encajar con la "
            "mesa virtual). Captura el origen desde el menu para corregirlo."
        )
    else:
        print(
            f"Origen del mundo cargado (RMS={world_transform.rms_reprojection:.2f} "
            f"px, capturado {world_transform.captured_at})."
        )

    tool_config = config.get("tool", {})
    try:
        smoothing_alpha = float(tool_config.get("smoothing_alpha", 1.0))
    except (TypeError, ValueError):
        smoothing_alpha = 1.0
    tracking_config = config.get("tracking", {})
    try:
        max_joint_speed_deg_s = float(
            tracking_config.get("max_joint_speed_deg_s", 360.0)
        )
    except (TypeError, ValueError):
        max_joint_speed_deg_s = 360.0

    tcp_aligned_axis = str(tool_config.get("tcp_aligned_axis", "z")).lower()
    tcp_axis_flip = bool(tool_config.get("tcp_axis_flip", False))

    ab = _optional_float(tool_config.get("marker_distance_cm"))
    c_along = _optional_float(tool_config.get("marker_c_along_ab_cm"))
    c_offset = _optional_float(tool_config.get("marker_c_offset_cm"))
    marker_b_local_cm = (float(ab), 0.0, 0.0) if ab is not None else None
    marker_c_local_cm = (
        (float(c_along), float(c_offset), 0.0)
        if c_along is not None and c_offset is not None
        else None
    )

    return UR5Visualizer(
        world_transform=world_transform,
        smoothing_alpha=smoothing_alpha,
        tcp_aligned_axis=tcp_aligned_axis,
        tcp_axis_flip=tcp_axis_flip,
        max_joint_speed_deg_s=max_joint_speed_deg_s,
        marker_b_local_cm=marker_b_local_cm,
        marker_c_local_cm=marker_c_local_cm,
    )


def run_pose_with_ur5_viewer() -> None:
    """Pipeline de pose + visor UR5 en vivo en el mismo proceso.

    Abre las dos camaras, ejecuta deteccion y triangulacion, y a la vez
    actualiza el robot virtual cada vez que hay pose nueva. Salir con q/Esc
    en cualquiera de las ventanas de OpenCV.
    """
    config = load_config()
    visualizer = _build_ur5_visualizer(config)
    if not visualizer.launch():
        return

    try:
        run_pose_estimation(
            on_payload=visualizer.update_from_payload,
            on_tick=visualizer.step,
        )
    finally:
        visualizer.close()


def run_pose_and_gestures() -> None:
    """Pipeline de pose + visor UR5 + gestos en el mismo bucle.

    Reutiliza la camara izquierda del pose para detectar gestos de la mano y
    controlar el seguimiento del robot:
      - mano abierta (`stop`) o dos dedos (`pause`): el robot congela la
        ultima pose;
      - puno cerrado (`continue`): el robot vuelve a seguir la herramienta.
    Salir con q/Esc en una de las ventanas de OpenCV.
    """
    from gestures.gesture_classifier import GestureController

    config = load_config()
    gestures_config = config.get("gestures", {})
    mapping = gestures_config.get("command_mapping", {})
    try:
        exclusion_radius_ratio = float(
            gestures_config.get("exclusion_radius_ratio", 0.18)
        )
    except (TypeError, ValueError):
        exclusion_radius_ratio = 0.18
    try:
        confirm_frames = int(gestures_config.get("confirm_frames", 6))
    except (TypeError, ValueError):
        confirm_frames = 6
    # Poses articulares fijas (en grados) que un comando puede disparar.
    poses = gestures_config.get("poses", {})

    visualizer = _build_ur5_visualizer(config)
    if not visualizer.launch():
        return

    def on_command(command: str) -> None:
        # Si el comando confirmado corresponde a una pose fija, el robot va
        # alli y se queda hasta que se reanude el seguimiento (puno cerrado).
        pose = poses.get(command)
        if pose is not None:
            visualizer.move_to_named_pose(pose)

    controller = GestureController(
        mapping,
        exclusion_radius_ratio=exclusion_radius_ratio,
        confirm_frames=confirm_frames,
        on_command=on_command,
    )

    def on_payload(payload: dict[str, Any]) -> None:
        # Solo movemos el robot mientras el gesto activo lo permita; si esta
        # pausado, el visor mantiene la ultima pose por si solo.
        if controller.active:
            visualizer.update_from_payload(payload)

    try:
        run_pose_estimation(
            on_payload=on_payload,
            on_tick=visualizer.step,
            on_frame=controller.process_frame,
        )
    finally:
        controller.close()
        visualizer.close()
