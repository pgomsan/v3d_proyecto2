from __future__ import annotations

import unittest

from main_pose import _smoothed_fps


class TrackingTests(unittest.TestCase):
    def test_smoothed_fps_initializes_and_filters(self) -> None:
        initial = _smoothed_fps(0.0, 10.0, 10.05, 0.15)
        filtered = _smoothed_fps(initial, 10.05, 10.15, 0.5)

        self.assertAlmostEqual(initial, 20.0)
        self.assertAlmostEqual(filtered, 15.0)

    def test_smoothed_fps_ignores_non_positive_elapsed_time(self) -> None:
        self.assertEqual(_smoothed_fps(12.0, 5.0, 5.0, 0.5), 12.0)


if __name__ == "__main__":
    unittest.main()
