from __future__ import annotations

import math
from typing import Sequence


Vector3 = tuple[float, float, float]
Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def distance_2d(a: Point2, b: Point2) -> float:
    """Distancia euclidea entre dos puntos 2D.

    Implementala como funcion pura, sin OpenCV, para que sea facil de testear.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_3d(a: Point3, b: Point3) -> float:
    """Distancia euclidea entre dos puntos 3D."""
    return math.sqrt(
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
    )


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    """Normaliza un vector.

    Deberia controlar el caso de norma cero para no dividir por cero. Puedes
    lanzar `ValueError` si el vector no tiene direccion valida.
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        raise ValueError("No se puede normalizar un vector de norma cero.")
    return tuple(float(component) / norm for component in vector)


def midpoint_2d(a: Point2, b: Point2) -> Point2:
    """Punto medio entre dos puntos 2D."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def midpoint_3d(a: Point3, b: Point3) -> Point3:
    """Punto medio entre dos puntos 3D."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def direction_from_points(a: Point3, b: Point3) -> Vector3:
    """Vector unitario desde `a` hasta `b`.

    Esta direccion sera la orientacion principal del bisturi/herramienta.
    """
    direction = normalize_vector((b[0] - a[0], b[1] - a[1], b[2] - a[2]))
    return (direction[0], direction[1], direction[2])


def build_orientation_from_direction(direction: Vector3) -> object:
    """Construye una representacion de orientacion a partir de una direccion.

    Decision pendiente:
    - matriz 3x3;
    - angulos de Euler;
    - quaternion;
    - pose 4x4.

    Para RoboDK suele ser comodo acabar con matriz o pose 4x4, pero puedes
    empezar exportando solo el vector direccion.
    """
    return {
        "format": "direction_vector",
        "value": list(normalize_vector(direction)),
    }
