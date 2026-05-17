from __future__ import annotations

from typing import Any

try:
    import cv2 as cv
except ModuleNotFoundError:
    cv = None


MARKER_COLORS = {
    "A": (0, 255, 0),
    "B": (255, 0, 255),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "orange": (0, 165, 255),
    "green": (0, 255, 0),
    "pink": (255, 0, 255),
}


def draw_marker_debug(frame: Any, detections: list[Any]) -> Any:
    """Dibuja centros, areas y etiquetas de marcas sobre un frame.

    Esta funcion deberia recibir detecciones ya calculadas. No debe detectar
    colores por su cuenta; solo visualizar resultados.
    """
    if cv is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    output = frame.copy()
    for detection in detections:
        if detection is None:
            continue
        x, y = detection.center_px
        center = (int(round(x)), int(round(y)))
        color = MARKER_COLORS.get(detection.name, (0, 255, 255))
        label = (
            f"{detection.name} "
            f"{detection.confidence:.2f} "
            f"{int(round(detection.area_px))}px"
        )
        cv.circle(output, center, 7, color, 2)
        cv.drawMarker(output, center, color, cv.MARKER_CROSS, 18, 2)
        cv.putText(
            output,
            label,
            (center[0] + 10, center[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv.LINE_AA,
        )
    return output


def draw_pose_debug(frame: Any, pose: Any) -> Any:
    """Dibuja la pose estimada de la herramienta.

    Implementacion recomendada:
    - dibujar linea entre marcas;
    - dibujar punta util si ya aplicas `tip_offset_cm`;
    - mostrar posicion, direccion y confianza en texto.
    """
    if cv is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    output = frame.copy()
    if pose is None:
        cv.putText(
            output,
            "Pose no detectada",
            (20, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        return output

    text = f"Pose conf={getattr(pose, 'confidence', 0.0):.2f}"
    position = getattr(pose, "position_cm", None)
    direction = getattr(pose, "direction", None)
    if position is not None:
        text += f" pos=({position[0]:.1f},{position[1]:.1f},{position[2]:.1f})cm"
    if direction is not None:
        text += f" dir=({direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f})"

    cv.putText(
        output,
        text,
        (20, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv.LINE_AA,
    )
    return output


def draw_gesture_debug(frame: Any, gesture: Any) -> Any:
    """Dibuja el gesto y comando activo sobre el frame."""
    if cv is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    output = frame.copy()
    if gesture is None:
        text = "Gesto: none"
    else:
        text = (
            f"Gesto: {getattr(gesture, 'gesture', 'unknown')} "
            f"cmd={getattr(gesture, 'command', 'none')} "
            f"conf={getattr(gesture, 'confidence', 0.0):.2f}"
        )
    cv.putText(
        output,
        text,
        (20, 35),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv.LINE_AA,
    )
    return output


def show_debug_frame(window_name: str, frame: Any) -> None:
    """Muestra una ventana OpenCV de depuracion.

    Mantener esta funcion fina permite cambiar despues el modo de visualizacion
    sin tocar los algoritmos de vision.
    """
    if cv is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")
    cv.imshow(window_name, frame)
