from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app_state import load_config

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


@dataclass
class CameraConfig:
    index: int
    name: str = "camera"
    width: int | None = None
    height: int | None = None
    fps: int | None = None


class CameraSource:
    """Wrapper minimo de una camara OpenCV.

    Cuando completes el proyecto, este objeto deberia ser el unico sitio donde
    se abre `cv2.VideoCapture`. Asi el resto del codigo trabaja con una API
    estable y no repite configuracion de ancho, alto, fps o liberacion.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.capture: Any | None = None

    def open(self) -> None:
        """Abre la camara y aplica parametros basicos de captura.

        Mas adelante puedes ampliar aqui:
        - backend concreto de OpenCV si lo necesitas;
        - comprobacion de resolucion real aceptada por la camara;
        - calentamiento de varios frames antes de empezar a procesar.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

        self.capture = cv2.VideoCapture(self.config.index)
        if self.config.width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if self.config.fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, self.config.fps)

        if not self.capture.isOpened():
            self.release()
            raise RuntimeError(f"No se pudo abrir la camara {self.config.name} en indice {self.config.index}.")

    def read(self) -> Any:
        """Devuelve un frame de la camara.

        Para la parte de vision, aqui no conviene hacer segmentacion ni
        calibracion: solo leer y devolver imagen. La correccion de distorsion,
        rectificacion estereo y deteccion de marcas deberian vivir en otros
        modulos.
        """
        if self.capture is None:
            self.open()

        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"No se pudo leer frame de la camara {self.config.name}.")
        return frame

    def release(self) -> None:
        """Libera la camara de forma segura."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def find_available_cameras(max_index: int = 10) -> list[int]:
    """Busca indices de camara que OpenCV puede abrir.

    Esta funcion solo comprueba disponibilidad. No decide cual es izquierda o
    derecha; esa decision se guarda en `state/config.json` desde el menu.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    available: list[int] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    available.append(index)
        finally:
            capture.release()
    return available


def _camera_config_from_dict(name: str, payload: dict[str, Any]) -> CameraConfig:
    return CameraConfig(
        index=int(payload.get("index", 0)),
        name=str(payload.get("name", name)),
        width=payload.get("width"),
        height=payload.get("height"),
        fps=payload.get("fps"),
    )


def load_stereo_camera_configs() -> tuple[CameraConfig, CameraConfig]:
    """Carga la pareja izquierda/derecha desde la configuracion persistente."""
    config = load_config()
    cameras = config.get("cameras", {})
    default_camera = config.get("camera", {})

    left_payload = {
        "index": config.get("camera_index", 0),
        "name": "left",
        **default_camera,
        **cameras.get("left", {}),
    }
    right_payload = {
        "index": 1,
        "name": "right",
        **default_camera,
        **cameras.get("right", {}),
    }
    return _camera_config_from_dict("left", left_payload), _camera_config_from_dict("right", right_payload)


def preview_cameras() -> None:
    """Muestra una previsualizacion simple de las dos camaras.

    Esto es deliberadamente basico. Cuando trabajes la vision, aqui solo
    deberias visualizar frames crudos o depuracion ligera. La deteccion de
    marcas, rectificacion estereo y calculo 3D deberian llamarse desde
    `main_pose.py`.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    left_config, right_config = load_stereo_camera_configs()
    left_camera = CameraSource(left_config)
    right_camera = CameraSource(right_config)

    try:
        if left_config.index == right_config.index:
            print("Aviso: izquierda y derecha usan el mismo indice. Para estereo deberian ser camaras distintas.")

        left_camera.open()
        right_camera.open()
        print("Previsualizando camaras. Pulsa q o Esc en una ventana de OpenCV para salir.")

        while True:
            left_frame = left_camera.read()
            right_frame = right_camera.read()

            cv2.imshow(f"Camara izquierda [{left_config.index}]", left_frame)
            cv2.imshow(f"Camara derecha [{right_config.index}]", right_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        left_camera.release()
        right_camera.release()
        cv2.destroyAllWindows()


def preview_camera() -> None:
    """Alias usado por el menu para mantener un nombre corto."""
    preview_cameras()
