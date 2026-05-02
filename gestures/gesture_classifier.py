from __future__ import annotations

from dataclasses import dataclass

from gestures.hand_detector import HandLandmarks


@dataclass
class GestureResult:
    gesture: str
    command: str
    confidence: float


def classify_gesture(hand: HandLandmarks) -> GestureResult:
    """Clasifica una mano a partir de sus landmarks.

    Implementacion recomendada inicial:
    - calcular que dedos estan extendidos;
    - mapear patrones sencillos a `open_hand`, `closed_fist`, `two_fingers`;
    - devolver `unknown` si no hay confianza suficiente.
    """
    pass


def gesture_to_command(gesture: str, command_mapping: dict[str, str]) -> str:
    """Convierte un gesto en comando de alto nivel para RoboDK."""
    pass


def preview_gesture_detection() -> None:
    """Muestra camara + gesto clasificado para depuracion."""
    pass


def run_gesture_detection() -> None:
    """Ejecuta deteccion de gestos y guarda ultimo comando.

    Implementacion recomendada:
    - abrir camara principal o ambas si quieres comparar;
    - detectar mano;
    - clasificar gesto;
    - guardar `state/last_gesture.json`;
    - anadir linea a `state/logs/gestures.jsonl`.
    """
    pass
