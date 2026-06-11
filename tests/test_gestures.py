from __future__ import annotations

import unittest

import numpy as np

from gestures.gesture_classifier import (
    GestureController,
    _hand_centroid_px,
    classify_gesture,
    fingers_extended,
    gesture_to_command,
)
from gestures.hand_detector import HandLandmarks

_MAPPING = {
    "open_hand": "stop",
    "closed_fist": "continue",
    "two_fingers": "pause",
    "unknown": "none",
}


class _FakeDetector:
    """Detector falso: devuelve manos predefinidas, sin camara ni MediaPipe."""

    def __init__(self, hands: list[HandLandmarks]) -> None:
        self.hands = hands

    def detect(self, frame: object) -> list[HandLandmarks]:
        return self.hands

    def close(self) -> None:
        pass


def _hand(points: list[tuple[float, float, float]], confidence: float = 0.9) -> HandLandmarks:
    return HandLandmarks(handedness="right", points=points, confidence=confidence)


def _build_points(extended: dict[str, bool]) -> list[tuple[float, float, float]]:
    """Construye 21 landmarks sinteticos con la mano apuntando hacia arriba.

    La muneca esta abajo (y alto) y los dedos hacia arriba (y bajo). Un dedo
    extendido tiene la punta mas lejos de la muneca que su articulacion PIP.
    """
    wrist = (0.5, 1.0, 0.0)
    points: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * 21
    points[0] = wrist

    # x base de cada dedo para separarlos horizontalmente.
    bases = {"thumb": 0.2, "index": 0.4, "middle": 0.5, "ring": 0.6, "pinky": 0.7}
    # (mcp, pip, dip, tip) indices por dedo.
    cadenas = {
        "thumb": (1, 2, 3, 4),
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }
    for nombre, (mcp, pip, dip, tip) in cadenas.items():
        x = bases[nombre]
        points[mcp] = (x, 0.8, 0.0)
        points[pip] = (x, 0.65, 0.0)
        if extended.get(nombre, False):
            # Punta lejos de la muneca (arriba).
            points[dip] = (x, 0.5, 0.0)
            points[tip] = (x, 0.4, 0.0)
        else:
            # Punta plegada hacia la muneca (mas cerca que el PIP).
            points[dip] = (x, 0.72, 0.0)
            points[tip] = (x, 0.78, 0.0)
    return points


class GestureClassifierTests(unittest.TestCase):
    def test_open_hand(self) -> None:
        points = _build_points(
            {"index": True, "middle": True, "ring": True, "pinky": True}
        )
        self.assertEqual(classify_gesture(_hand(points)).gesture, "open_hand")

    def test_closed_fist(self) -> None:
        points = _build_points(
            {"index": False, "middle": False, "ring": False, "pinky": False}
        )
        self.assertEqual(classify_gesture(_hand(points)).gesture, "closed_fist")

    def test_two_fingers(self) -> None:
        points = _build_points(
            {"index": True, "middle": True, "ring": False, "pinky": False}
        )
        self.assertEqual(classify_gesture(_hand(points)).gesture, "two_fingers")

    def test_unknown_for_non_contiguous_three(self) -> None:
        # Tres dedos pero no indice+corazon+anular: patron no reconocido.
        points = _build_points(
            {"index": True, "middle": False, "ring": True, "pinky": True}
        )
        self.assertEqual(classify_gesture(_hand(points)).gesture, "unknown")

    def test_unknown_for_pinky_only(self) -> None:
        # Dos dedos extendidos pero no son indice+corazon.
        points = _build_points(
            {"index": False, "middle": True, "ring": False, "pinky": True}
        )
        self.assertEqual(classify_gesture(_hand(points)).gesture, "unknown")

    def test_fingers_extended_detection(self) -> None:
        points = _build_points(
            {"index": True, "middle": False, "ring": False, "pinky": False}
        )
        estado = fingers_extended(points)
        self.assertTrue(estado["index"])
        self.assertFalse(estado["middle"])

    def test_empty_landmarks_is_unknown(self) -> None:
        result = classify_gesture(_hand([]))
        self.assertEqual(result.gesture, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_gesture_to_command_mapping(self) -> None:
        mapping = {
            "open_hand": "stop",
            "closed_fist": "continue",
            "two_fingers": "pause",
            "unknown": "none",
        }
        self.assertEqual(gesture_to_command("open_hand", mapping), "stop")
        self.assertEqual(gesture_to_command("closed_fist", mapping), "continue")
        self.assertEqual(gesture_to_command("two_fingers", mapping), "pause")
        self.assertEqual(gesture_to_command("algo_raro", mapping), "none")


class GestureControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.open_points = _build_points(
            {"index": True, "middle": True, "ring": True, "pinky": True}
        )
        self.fist_points = _build_points(
            {"index": False, "middle": False, "ring": False, "pinky": False}
        )
        self.centroid = _hand_centroid_px(self.open_points, 640, 480)

    def test_ignora_la_mano_que_sostiene_la_herramienta(self) -> None:
        # Unica mano visible, pero esta justo sobre la herramienta: se ignora.
        detector = _FakeDetector([_hand(self.open_points)])
        controller = GestureController(
            _MAPPING, detector=detector, persist=False, confirm_frames=2
        )
        for _ in range(5):
            controller.process_frame(self.frame, tool_pixels=[self.centroid])
        # Nunca se aplico el 'stop': el robot sigue activo.
        self.assertTrue(controller.active)

    def test_mano_libre_pausa_tras_confirmacion(self) -> None:
        detector = _FakeDetector([_hand(self.open_points)])
        controller = GestureController(
            _MAPPING, detector=detector, persist=False, confirm_frames=3
        )
        # Herramienta lejos del centroide de la mano (fuera de la zona muerta).
        tool_lejos = [(self.centroid[0] + 400.0, self.centroid[1])]
        controller.process_frame(self.frame, tool_pixels=tool_lejos)
        controller.process_frame(self.frame, tool_pixels=tool_lejos)
        self.assertTrue(controller.active)  # aun sin confirmar
        controller.process_frame(self.frame, tool_pixels=tool_lejos)
        self.assertFalse(controller.active)  # 3er frame confirma 'stop'

    def test_three_fingers_es_un_gesto(self) -> None:
        points = _build_points(
            {"index": True, "middle": True, "ring": True, "pinky": False}
        )
        self.assertEqual(classify_gesture(_hand(points)).gesture, "three_fingers")

    def test_comando_de_pose_dispara_callback_y_para_seguimiento(self) -> None:
        mapping = dict(_MAPPING)
        mapping["three_fingers"] = "home"
        three_points = _build_points(
            {"index": True, "middle": True, "ring": True, "pinky": False}
        )
        detector = _FakeDetector([_hand(three_points)])
        recibidos: list[str] = []
        controller = GestureController(
            mapping,
            detector=detector,
            persist=False,
            confirm_frames=2,
            on_command=recibidos.append,
        )
        tool_lejos = [(self.centroid[0] + 400.0, self.centroid[1])]
        controller.process_frame(self.frame, tool_pixels=tool_lejos)
        controller.process_frame(self.frame, tool_pixels=tool_lejos)
        self.assertEqual(recibidos, ["home"])
        self.assertFalse(controller.active)  # ir a pose detiene el seguimiento

    def test_puno_reanuda_tras_pausa(self) -> None:
        detector = _FakeDetector([_hand(self.fist_points)])
        controller = GestureController(
            _MAPPING, detector=detector, persist=False, confirm_frames=2
        )
        controller.active = False
        controller.last_command = "stop"
        tool_lejos = [(self.centroid[0] + 400.0, self.centroid[1])]
        controller.process_frame(self.frame, tool_pixels=tool_lejos)
        controller.process_frame(self.frame, tool_pixels=tool_lejos)
        self.assertTrue(controller.active)
        self.assertEqual(controller.last_command, "continue")


if __name__ == "__main__":
    unittest.main()
