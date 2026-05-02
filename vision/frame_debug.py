from __future__ import annotations

from typing import Any


def draw_marker_debug(frame: Any, detections: list[Any]) -> Any:
    """Dibuja centros, areas y etiquetas de marcas sobre un frame.

    Esta funcion deberia recibir detecciones ya calculadas. No debe detectar
    colores por su cuenta; solo visualizar resultados.
    """
    pass


def draw_pose_debug(frame: Any, pose: Any) -> Any:
    """Dibuja la pose estimada de la herramienta.

    Implementacion recomendada:
    - dibujar linea entre marcas;
    - dibujar punta util si ya aplicas `tip_offset_cm`;
    - mostrar posicion, direccion y confianza en texto.
    """
    pass


def draw_gesture_debug(frame: Any, gesture: Any) -> Any:
    """Dibuja el gesto y comando activo sobre el frame."""
    pass


def show_debug_frame(window_name: str, frame: Any) -> None:
    """Muestra una ventana OpenCV de depuracion.

    Mantener esta funcion fina permite cambiar despues el modo de visualizacion
    sin tocar los algoritmos de vision.
    """
    pass
