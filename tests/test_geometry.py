from __future__ import annotations

import unittest

from pose.geometry import (
    build_orientation_from_markers,
    build_orientation_from_direction,
    cross_product,
    direction_from_points,
    distance_2d,
    distance_3d,
    midpoint_2d,
    midpoint_3d,
    normalize_vector,
    rotation_matrix_from_markers,
)


class GeometryTests(unittest.TestCase):
    def test_distance_and_midpoint(self) -> None:
        self.assertEqual(distance_2d((0.0, 0.0), (3.0, 4.0)), 5.0)
        self.assertEqual(distance_3d((0.0, 0.0, 0.0), (2.0, 3.0, 6.0)), 7.0)
        self.assertEqual(midpoint_2d((2.0, 4.0), (4.0, 8.0)), (3.0, 6.0))
        self.assertEqual(
            midpoint_3d((2.0, 4.0, 6.0), (4.0, 8.0, 10.0)),
            (3.0, 6.0, 8.0),
        )

    def test_direction_from_points(self) -> None:
        direction = direction_from_points((0.0, 0.0, 0.0), (0.0, 3.0, 4.0))

        self.assertAlmostEqual(direction[0], 0.0)
        self.assertAlmostEqual(direction[1], 0.6)
        self.assertAlmostEqual(direction[2], 0.8)

    def test_zero_vector_cannot_be_normalized(self) -> None:
        with self.assertRaises(ValueError):
            normalize_vector((0.0, 0.0, 0.0))

    def test_orientation_exports_direction_vector(self) -> None:
        orientation = build_orientation_from_direction((0.0, 0.0, 2.0))

        self.assertEqual(orientation["format"], "direction_vector")
        self.assertEqual(orientation["value"], [0.0, 0.0, 1.0])

    def test_rotation_matrix_from_three_markers(self) -> None:
        rotation = rotation_matrix_from_markers(
            (1.0, 2.0, 3.0),
            (16.0, 2.0, 3.0),
            (6.0, 7.0, 3.0),
        )

        self.assertEqual(
            rotation,
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )
        orientation = build_orientation_from_markers(
            (1.0, 2.0, 3.0),
            (16.0, 2.0, 3.0),
            (6.0, 7.0, 3.0),
        )
        self.assertEqual(orientation["format"], "rotation_matrix")
        self.assertEqual(orientation["value"], [list(row) for row in rotation])

    def test_cross_product(self) -> None:
        self.assertEqual(
            cross_product((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            (0.0, 0.0, 1.0),
        )


if __name__ == "__main__":
    unittest.main()
