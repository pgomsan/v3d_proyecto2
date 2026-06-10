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


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    """Producto escalar de dos vectores de la misma dimension."""
    return float(sum(first * second for first, second in zip(a, b)))


def cross_product(a: Vector3, b: Vector3) -> Vector3:
    """Producto vectorial 3D."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def rotation_matrix_from_markers(
    marker_a: Point3,
    marker_b: Point3,
    marker_c: Point3,
) -> tuple[Vector3, Vector3, Vector3]:
    """Construye el frame herramienta a partir de tres marcas no colineales.

    La matriz se devuelve por filas y sus columnas son:
      - X: direccion A -> B;
      - Y: componente de A -> C perpendicular a X;
      - Z: X x Y.
    """
    x_axis = direction_from_points(marker_a, marker_b)
    ac = (
        marker_c[0] - marker_a[0],
        marker_c[1] - marker_a[1],
        marker_c[2] - marker_a[2],
    )
    projection = dot_product(ac, x_axis)
    y_raw = (
        ac[0] - projection * x_axis[0],
        ac[1] - projection * x_axis[1],
        ac[2] - projection * x_axis[2],
    )
    y_normalized = normalize_vector(y_raw)
    y_axis = (y_normalized[0], y_normalized[1], y_normalized[2])
    z_normalized = normalize_vector(cross_product(x_axis, y_axis))
    z_axis = (z_normalized[0], z_normalized[1], z_normalized[2])
    y_corrected = cross_product(z_axis, x_axis)

    return (
        (x_axis[0], y_corrected[0], z_axis[0]),
        (x_axis[1], y_corrected[1], z_axis[1]),
        (x_axis[2], y_corrected[2], z_axis[2]),
    )


def build_orientation_from_markers(
    marker_a: Point3,
    marker_b: Point3,
    marker_c: Point3,
) -> object:
    """Exporta la orientacion completa de la herramienta como matriz 3x3."""
    return {
        "format": "rotation_matrix",
        "value": [
            list(row)
            for row in rotation_matrix_from_markers(marker_a, marker_b, marker_c)
        ],
        "axes": {
            "x": "marker_a_to_marker_b",
            "y": "marker_a_to_marker_c_orthogonalized",
            "z": "x_cross_y",
        },
    }


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
