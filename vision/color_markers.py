from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2 as cv
    import numpy as np
except ModuleNotFoundError:
    cv = None
    np = None


MIN_MARKER_AREA_PX = 120.0
MAX_MARKER_AREA_RATIO = 0.03
MORPH_KERNEL_SIZE = 5


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


@dataclass
class MarkerTriplet:
    marker_a: MarkerDetection
    marker_b: MarkerDetection
    marker_c: MarkerDetection


def threshold_color(frame: Any, color_range: ColorRange) -> Any:
    """Devuelve una mascara binaria para un rango HSV.

    Implementacion recomendada:
    - convertir BGR a HSV con `cv2.cvtColor`;
    - aplicar `cv2.inRange`;
    - limpiar ruido con apertura/cierre morfologico;
    - devolver la mascara para depuracion y deteccion de contornos.
    """
    if cv is None or np is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower = np.array(color_range.lower, dtype=np.uint8)
    upper = np.array(color_range.upper, dtype=np.uint8)

    if lower[0] <= upper[0]:
        mask = cv.inRange(hsv, lower, upper)
    else:
        # El rojo rodea el cero de OpenCV HSV: [170..179] U [0..10].
        lower_high = np.array([lower[0], lower[1], lower[2]], dtype=np.uint8)
        upper_high = np.array([179, upper[1], upper[2]], dtype=np.uint8)
        lower_low = np.array([0, lower[1], lower[2]], dtype=np.uint8)
        upper_low = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)
        mask = cv.bitwise_or(
            cv.inRange(hsv, lower_high, upper_high),
            cv.inRange(hsv, lower_low, upper_low),
        )

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    return mask


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
    if cv is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    mask = threshold_color(frame, color_range)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    frame_area = float(frame.shape[0] * frame.shape[1])
    max_marker_area = max(MIN_MARKER_AREA_PX, frame_area * MAX_MARKER_AREA_RATIO)
    candidates = [
        contour
        for contour in contours
        if MIN_MARKER_AREA_PX <= cv.contourArea(contour) <= max_marker_area
    ]
    if not candidates:
        return None

    contour = max(candidates, key=cv.contourArea)
    area = float(cv.contourArea(contour))
    moments = cv.moments(contour)
    if moments["m00"] != 0.0:
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    else:
        rect = cv.minAreaRect(contour)
        center = (float(rect[0][0]), float(rect[0][1]))

    area_score = min(1.0, area / max(MIN_MARKER_AREA_PX * 20.0, 1.0))
    size_score = min(1.0, area / max(frame_area * 0.015, 1.0))
    confidence = max(0.05, min(1.0, 0.65 * area_score + 0.35 * size_score))

    return MarkerDetection(
        name=name,
        center_px=(float(center[0]), float(center[1])),
        area_px=area,
        confidence=confidence,
    )


def detect_marker_pair(frame: Any, marker_a_range: ColorRange, marker_b_range: ColorRange) -> MarkerPair | None:
    """Detecta las dos marcas que definen la herramienta.

    Implementacion recomendada:
    - detectar marca A y marca B por separado;
    - si falta una, devolver `None`;
    - validar que la distancia entre marcas en pixeles tiene sentido;
    - devolver `MarkerPair` para que `pose/rod_pose.py` estime la pose.
    """
    marker_a = detect_marker(frame, "A", marker_a_range)
    marker_b = detect_marker(frame, "B", marker_b_range)
    if marker_a is None or marker_b is None:
        return None

    dx = marker_a.center_px[0] - marker_b.center_px[0]
    dy = marker_a.center_px[1] - marker_b.center_px[1]
    if (dx * dx + dy * dy) ** 0.5 < 5.0:
        return None

    return MarkerPair(marker_a=marker_a, marker_b=marker_b)


def detect_marker_triplet(
    frame: Any,
    marker_a_range: ColorRange,
    marker_b_range: ColorRange,
    marker_c_range: ColorRange,
) -> MarkerTriplet | None:
    """Detecta A, B y C, necesarias para recuperar la orientacion 3D completa."""
    marker_a = detect_marker(frame, "A", marker_a_range)
    marker_b = detect_marker(frame, "B", marker_b_range)
    marker_c = detect_marker(frame, "C", marker_c_range)
    if marker_a is None or marker_b is None or marker_c is None:
        return None

    centers = [marker_a.center_px, marker_b.center_px, marker_c.center_px]
    for index, first in enumerate(centers):
        for second in centers[index + 1 :]:
            dx = first[0] - second[0]
            dy = first[1] - second[1]
            if (dx * dx + dy * dy) ** 0.5 < 5.0:
                return None

    return MarkerTriplet(
        marker_a=marker_a,
        marker_b=marker_b,
        marker_c=marker_c,
    )


def tune_hsv_ranges() -> None:
    """Interfaz de ajuste manual de rangos HSV.

    Implementacion recomendada:
    - abrir camara de referencia;
    - crear trackbars OpenCV para H, S y V;
    - visualizar frame original y mascara;
    - al confirmar, guardar los rangos en `state/config.json`.
    """
    raise NotImplementedError(
        "El ajuste interactivo HSV queda pendiente; usa state/config.json para "
        "editar los rangos iniciales."
    )
