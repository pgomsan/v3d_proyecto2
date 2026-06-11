from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ModuleNotFoundError:
    mp = None
    mp_python = None
    mp_vision = None


# El build de MediaPipe 0.10.x solo trae la Tasks API, que necesita un modelo
# `.task`. Se descarga una vez a esta ruta (no se versiona) y, si falta, se
# intenta bajar automaticamente desde el repositorio oficial de modelos.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "gestures" / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model(model: Path) -> bool:
    """Garantiza que el modelo existe; lo descarga si falta. Devuelve si esta."""
    if model.exists():
        return True
    try:
        import urllib.request

        print(f"Descargando modelo de manos en {model} ...")
        model.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model)
        print("Modelo descargado.")
        return True
    except Exception as exc:  # noqa: BLE001 - red opcional, no debe romper
        print(f"Aviso: no se pudo descargar el modelo de manos ({exc}).")
        return False


@dataclass
class HandLandmarks:
    handedness: str
    points: list[tuple[float, float, float]]
    confidence: float


class HandDetector:
    """Detector de manos basado en MediaPipe Hand Landmarker (Tasks API).

    Se mantiene aislado del clasificador para poder cambiar de backend sin
    tocar la logica de gestos. Si MediaPipe o el modelo `.task` no estan
    disponibles, `detect` devuelve una lista vacia en lugar de fallar.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_path: Path | str | None = None,
    ) -> None:
        """Inicializa el backend de deteccion de manos."""
        self._landmarker: Any | None = None
        model = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if mp_vision is None or not _ensure_model(model):
            return

        base_options = mp_python.BaseOptions(model_asset_path=str(model))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: Any) -> list[HandLandmarks]:
        """Devuelve landmarks normalizados de las manos visibles.

        - convierte BGR a RGB (MediaPipe espera RGB);
        - ejecuta el Hand Landmarker;
        - convierte cada mano a `HandLandmarks` (21 puntos normalizados);
        - devuelve lista vacia si no hay manos o el backend no esta disponible.
        """
        if self._landmarker is None or cv2 is None or mp is None or frame is None:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return []

        hands: list[HandLandmarks] = []
        handedness_list = result.handedness or []
        for idx, landmarks in enumerate(result.hand_landmarks):
            points = [(lm.x, lm.y, lm.z) for lm in landmarks]
            handedness = "unknown"
            confidence = 1.0
            if idx < len(handedness_list) and handedness_list[idx]:
                category = handedness_list[idx][0]
                handedness = category.category_name.lower()
                confidence = float(category.score)
            hands.append(
                HandLandmarks(
                    handedness=handedness,
                    points=points,
                    confidence=confidence,
                )
            )
        return hands

    def close(self) -> None:
        """Libera recursos del detector."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def preview_hand_detection() -> None:
    """Previsualiza landmarks de mano sobre la camara (delegado en gestos)."""
    from gestures.gesture_classifier import preview_gesture_detection

    preview_gesture_detection()
