from __future__ import annotations

import time
from typing import Any

from app_state import load_config, save_last_pose
from pose.geometry import distance_3d
from pose.rod_pose import (
    RodPoseEstimator,
    ToolMarkerPair3D,
    ToolParameters,
)
from vision.camera import CameraSource, load_stereo_camera_configs
from vision.color_markers import ColorRange, detect_marker
from vision.frame_debug import draw_marker_debug, show_debug_frame
from vision.stereo import load_stereo_calibration

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


def _draw_marker_line(frame: Any, marker_a: Any, marker_b: Any) -> Any:
    if cv is None or marker_a is None or marker_b is None:
        return frame

    a = tuple(int(round(value)) for value in marker_a.center_px)
    b = tuple(int(round(value)) for value in marker_b.center_px)
    cv.line(frame, a, b, (0, 255, 255), 2)
    cv.putText(
        frame,
        "A-B",
        ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2 - 12),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
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
    expected_distance_cm: float | None,
    stereo_rms: float,
    center_cm: tuple[float, float, float] | None = None,
    tip_position_cm: tuple[float, float, float] | None = None,
) -> Any:
    if cv is None:
        return frame

    if marker_a_3d is None or marker_b_3d is None:
        lines = [
            "Pose 3D: marcas incompletas",
            f"RMS estereo: {stereo_rms:.2f}",
        ]
        color = (0, 0, 255)
    else:
        distance_cm = distance_3d(marker_a_3d, marker_b_3d)
        error_text = ""
        if expected_distance_cm is not None:
            error_text = f" error={distance_cm - expected_distance_cm:+.2f}cm"
        lines = [
            f"Distancia 3D A-B: {distance_cm:.2f}cm{error_text}",
            f"A=({marker_a_3d[0]:.1f},{marker_a_3d[1]:.1f},{marker_a_3d[2]:.1f})cm",
            f"B=({marker_b_3d[0]:.1f},{marker_b_3d[1]:.1f},{marker_b_3d[2]:.1f})cm",
            f"RMS estereo: {stereo_rms:.2f}",
        ]
        if center_cm is not None:
            lines.insert(
                1,
                "Centro herramienta="
                f"({center_cm[0]:.1f},{center_cm[1]:.1f},{center_cm[2]:.1f})cm",
            )
        if tip_position_cm is not None:
            lines.insert(
                2,
                "Punta estimada="
                f"({tip_position_cm[0]:.1f},{tip_position_cm[1]:.1f},{tip_position_cm[2]:.1f})cm",
            )
        color = (0, 255, 255)

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


def _tool_marker_distance(config: dict[str, Any]) -> float | None:
    raw_distance = config.get("tool", {}).get("marker_distance_cm")
    try:
        return None if raw_distance is None else float(raw_distance)
    except (TypeError, ValueError):
        return None


def _tool_parameters_from_config(config: dict[str, Any]) -> ToolParameters:
    tool_config = config.get("tool", {})
    raw_tip_offset = tool_config.get("tip_offset_cm", [0.0, 0.0, 0.0])
    try:
        tip_offset_cm = tuple(float(value) for value in raw_tip_offset[:3])
    except (TypeError, ValueError):
        tip_offset_cm = (0.0, 0.0, 0.0)
    if len(tip_offset_cm) != 3:
        tip_offset_cm = (0.0, 0.0, 0.0)

    raw_length = tool_config.get("length_cm")
    raw_marker_distance = tool_config.get("marker_distance_cm")
    try:
        length_cm = None if raw_length is None else float(raw_length)
    except (TypeError, ValueError):
        length_cm = None
    try:
        marker_distance_cm = (
            None if raw_marker_distance is None else float(raw_marker_distance)
        )
    except (TypeError, ValueError):
        marker_distance_cm = None

    return ToolParameters(
        tool_id=str(tool_config.get("tool_id", "tool_01")),
        tool_type=str(tool_config.get("tool_type", "tool")),
        length_cm=length_cm,
        marker_distance_cm=marker_distance_cm,
        tip_offset_cm=tip_offset_cm,
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
    left_config, right_config = load_stereo_camera_configs()
    left_camera = CameraSource(left_config)
    right_camera = CameraSource(right_config)

    print("Previsualizando marcas. Pulsa q o Esc en una ventana para salir.")
    print(f"Marca A HSV: {marker_a_range.lower} -> {marker_a_range.upper}")
    print(f"Marca B HSV: {marker_b_range.lower} -> {marker_b_range.upper}")

    try:
        left_camera.open()
        right_camera.open()

        while True:
            left_frame = left_camera.read()
            right_frame = right_camera.read()

            left_a = detect_marker(left_frame, "A", marker_a_range)
            left_b = detect_marker(left_frame, "B", marker_b_range)
            right_a = detect_marker(right_frame, "A", marker_a_range)
            right_b = detect_marker(right_frame, "B", marker_b_range)

            left_debug = draw_marker_debug(left_frame, [left_a, left_b])
            right_debug = draw_marker_debug(right_frame, [right_a, right_b])
            _draw_marker_line(left_debug, left_a, left_b)
            _draw_marker_line(right_debug, right_a, right_b)

            show_debug_frame(f"Marcas izquierda [{left_config.index}]", left_debug)
            show_debug_frame(f"Marcas derecha [{right_config.index}]", right_debug)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        left_camera.release()
        right_camera.release()
        cv.destroyAllWindows()


def run_pose_estimation() -> None:
    """Bucle principal de estimacion de pose.

    Implementacion recomendada:
    - abrir camaras izquierda y derecha;
    - leer frames sincronizados de forma aproximada;
    - corregir/rectificar si ya tienes calibracion;
    - detectar marcas;
    - reconstruir posicion 3D con tu metodo;
    - guardar pose actual y log.
    """
    if cv is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    config = load_config()
    marker_a_range = _color_range_from_config(config, "marker_a")
    marker_b_range = _color_range_from_config(config, "marker_b")
    expected_distance_cm = _tool_marker_distance(config)
    tool_parameters = _tool_parameters_from_config(config)
    pose_estimator = RodPoseEstimator(tool_parameters)
    stereo = load_stereo_calibration()
    left_config, right_config = load_stereo_camera_configs()
    left_camera = CameraSource(left_config)
    right_camera = CameraSource(right_config)
    last_saved_at = 0.0

    print("Estimando pose 3D. Pulsa q o Esc en una ventana para salir.")
    print(f"RMS estereo cargado: {stereo.rms:.3f}")
    if stereo.rms > 5.0:
        print("Aviso: RMS estereo alto. La triangulacion puede ser poco fiable.")

    try:
        left_camera.open()
        right_camera.open()

        while True:
            left_frame = left_camera.read()
            right_frame = right_camera.read()
            left_rect, right_rect = stereo.rectify_pair(left_frame, right_frame)

            left_a = detect_marker(left_rect, "A", marker_a_range)
            left_b = detect_marker(left_rect, "B", marker_b_range)
            right_a = detect_marker(right_rect, "A", marker_a_range)
            right_b = detect_marker(right_rect, "B", marker_b_range)

            marker_a_3d: tuple[float, float, float] | None = None
            marker_b_3d: tuple[float, float, float] | None = None
            pose_payload: dict[str, Any] | None = None
            center_3d: tuple[float, float, float] | None = None
            tip_3d: tuple[float, float, float] | None = None
            left_center_px: tuple[float, float] | None = None
            left_tip_px: tuple[float, float] | None = None
            right_center_px: tuple[float, float] | None = None
            right_tip_px: tuple[float, float] | None = None
            if left_a is not None and right_a is not None:
                marker_a_3d = stereo.triangulate_point(
                    left_a.center_px, right_a.center_px
                )
            if left_b is not None and right_b is not None:
                marker_b_3d = stereo.triangulate_point(
                    left_b.center_px, right_b.center_px
                )

            if marker_a_3d is not None and marker_b_3d is not None:
                confidence = min(
                    left_a.confidence,
                    right_a.confidence,
                    left_b.confidence,
                    right_b.confidence,
                )
                marker_pair_3d = ToolMarkerPair3D(
                    marker_a_cm=marker_a_3d,
                    marker_b_cm=marker_b_3d,
                    confidence=confidence,
                    marker_pixels={
                        "a_left_px": list(left_a.center_px),
                        "a_right_px": list(right_a.center_px),
                        "b_left_px": list(left_b.center_px),
                        "b_right_px": list(right_b.center_px),
                    },
                )
                tool_pose = pose_estimator.estimate_from_markers(marker_pair_3d)
                if tool_pose is not None:
                    pose_payload = pose_estimator.build_payload(
                        tool_pose, marker_pair_3d
                    )
                    pose_payload["stereo_rms"] = stereo.rms
                    center_3d = tuple(pose_payload["markers"]["center_3d_cm"])
                    tip_3d = tuple(pose_payload["tip_position_cm"])
                    left_center_px = stereo.project_left_point(center_3d)
                    left_tip_px = stereo.project_left_point(tip_3d)
                    right_center_px = stereo.project_right_point(center_3d)
                    right_tip_px = stereo.project_right_point(tip_3d)
                now = time.monotonic()
                if pose_payload is not None and now - last_saved_at >= 0.5:
                    save_last_pose(pose_payload)
                    last_saved_at = now

            left_debug = draw_marker_debug(left_rect, [left_a, left_b])
            right_debug = draw_marker_debug(right_rect, [right_a, right_b])
            _draw_marker_line(left_debug, left_a, left_b)
            _draw_marker_line(right_debug, right_a, right_b)
            _draw_tip_projection(left_debug, left_tip_px, left_center_px)
            _draw_tip_projection(right_debug, right_tip_px, right_center_px)
            _draw_pose_overlay(
                left_debug,
                marker_a_3d,
                marker_b_3d,
                expected_distance_cm,
                stereo.rms,
                center_3d,
                tip_3d,
            )

            show_debug_frame(f"Pose izquierda rectificada [{left_config.index}]", left_debug)
            show_debug_frame(f"Pose derecha rectificada [{right_config.index}]", right_debug)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        left_camera.release()
        right_camera.release()
        cv.destroyAllWindows()


def run_pose_and_gestures() -> None:
    """Ejecuta pose y gestos en el mismo bucle.

    Esta funcion deberia producir un payload conjunto para RoboDK:
    pose de herramienta + comando de gesto activo.
    """
    pass
