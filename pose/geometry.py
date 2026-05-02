from __future__ import annotations

from typing import Sequence


Vector3 = tuple[float, float, float]
Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def distance_2d(a: Point2, b: Point2) -> float:
    """Distancia euclidea entre dos puntos 2D.

    Implementala como funcion pura, sin OpenCV, para que sea facil de testear.
    """
    pass


def distance_3d(a: Point3, b: Point3) -> float:
    """Distancia euclidea entre dos puntos 3D."""
    pass


def normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    """Normaliza un vector.

    Deberia controlar el caso de norma cero para no dividir por cero. Puedes
    lanzar `ValueError` si el vector no tiene direccion valida.
    """
    pass


def midpoint_2d(a: Point2, b: Point2) -> Point2:
    """Punto medio entre dos puntos 2D."""
    pass


def midpoint_3d(a: Point3, b: Point3) -> Point3:
    """Punto medio entre dos puntos 3D."""
    pass


def direction_from_points(a: Point3, b: Point3) -> Vector3:
    """Vector unitario desde `a` hasta `b`.

    Esta direccion sera la orientacion principal del bisturi/herramienta.
    """
    pass


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
    pass
