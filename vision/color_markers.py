from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ColorRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass
class MarkerDetection:
    name: str
    center_px: tuple[float, float]
    area_px: float
    confidence: float


@dataclass
class MarkerPair:
    marker_a: MarkerDetection
    marker_b: MarkerDetection


def threshold_color(frame: Any, color_range: ColorRange) -> Any:
    """Devuelve una mascara binaria para un rango HSV.

    Implementacion recomendada:
    - convertir BGR a HSV con `cv2.cvtColor`;
    - aplicar `cv2.inRange`;
    - limpiar ruido con apertura/cierre morfologico;
    - devolver la mascara para depuracion y deteccion de contornos.
    """
    pass


def detect_marker(frame: Any, name: str, color_range: ColorRange) -> MarkerDetection | None:
    """Detecta una marca de color y devuelve su centro.

    Implementacion recomendada:
    - llamar a `threshold_color`;
    - buscar contornos;
    - filtrar por area minima y circularidad;
    - escoger el contorno mas fiable;
    - calcular centroide con momentos;
    - devolver `MarkerDetection` con centro, area y confianza.
    """
    pass


def detect_marker_pair(frame: Any, marker_a_range: ColorRange, marker_b_range: ColorRange) -> MarkerPair | None:
    """Detecta las dos marcas que definen la herramienta.

    Implementacion recomendada:
    - detectar marca A y marca B por separado;
    - si falta una, devolver `None`;
    - validar que la distancia entre marcas en pixeles tiene sentido;
    - devolver `MarkerPair` para que `pose/rod_pose.py` estime la pose.
    """
    pass


def tune_hsv_ranges() -> None:
    """Interfaz de ajuste manual de rangos HSV.

    Implementacion recomendada:
    - abrir camara de referencia;
    - crear trackbars OpenCV para H, S y V;
    - visualizar frame original y mascara;
    - al confirmar, guardar los rangos en `state/config.json`.
    """
    pass
