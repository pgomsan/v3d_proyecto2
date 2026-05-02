from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HandLandmarks:
    handedness: str
    points: list[tuple[float, float, float]]
    confidence: float


class HandDetector:
    """Detector de manos.

    La implementacion prevista es MediaPipe Hands, pero se deja aislada para
    poder cambiar de backend sin tocar el clasificador de gestos.
    """

    def __init__(self) -> None:
        """Inicializa el backend de deteccion de manos."""
        pass

    def detect(self, frame: Any) -> list[HandLandmarks]:
        """Devuelve landmarks normalizados de las manos visibles.

        Implementacion recomendada:
        - convertir BGR a RGB si usas MediaPipe;
        - ejecutar el detector;
        - convertir landmarks a `HandLandmarks`;
        - devolver lista vacia si no hay manos.
        """
        pass

    def close(self) -> None:
        """Libera recursos del detector si el backend lo requiere."""
        pass


def preview_hand_detection() -> None:
    """Previsualiza landmarks de mano sobre la camara."""
    pass
