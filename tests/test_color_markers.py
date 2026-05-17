from __future__ import annotations

import unittest

try:
    import cv2 as cv
    import numpy as np
except ModuleNotFoundError:
    cv = None
    np = None

from vision.color_markers import ColorRange, detect_marker_pair, threshold_color


@unittest.skipIf(cv is None or np is None, "OpenCV o NumPy no estan instalados")
class ColorMarkerTests(unittest.TestCase):
    def test_detects_green_and_pink_markers(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.rectangle(frame, (45, 70), (105, 120), (80, 180, 40), -1)
        cv.rectangle(frame, (220, 65), (280, 125), (180, 0, 220), -1)

        green_range = ColorRange(lower=(55, 100, 60), upper=(88, 255, 255))
        pink_range = ColorRange(lower=(145, 60, 80), upper=(179, 255, 255))

        marker_pair = detect_marker_pair(frame, green_range, pink_range)

        self.assertIsNotNone(marker_pair)
        assert marker_pair is not None
        self.assertAlmostEqual(marker_pair.marker_a.center_px[0], 75.0, delta=2.0)
        self.assertAlmostEqual(marker_pair.marker_a.center_px[1], 95.0, delta=2.0)
        self.assertAlmostEqual(marker_pair.marker_b.center_px[0], 250.0, delta=2.0)
        self.assertAlmostEqual(marker_pair.marker_b.center_px[1], 95.0, delta=2.0)

    def test_threshold_color_detects_green_range(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[10:30, 10:30] = (80, 180, 40)
        green_range = ColorRange(lower=(55, 100, 60), upper=(88, 255, 255))

        mask = threshold_color(frame, green_range)

        self.assertGreater(int(mask.sum()), 0)

    def test_ignores_green_regions_too_large_to_be_a_marker(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.rectangle(frame, (20, 20), (300, 260), (80, 180, 40), -1)
        cv.rectangle(frame, (420, 210), (470, 250), (80, 180, 40), -1)
        cv.rectangle(frame, (520, 200), (570, 250), (180, 0, 220), -1)
        green_range = ColorRange(lower=(55, 100, 60), upper=(88, 255, 255))
        pink_range = ColorRange(lower=(145, 60, 80), upper=(179, 255, 255))

        marker_pair = detect_marker_pair(frame, green_range, pink_range)

        self.assertIsNotNone(marker_pair)
        assert marker_pair is not None
        self.assertAlmostEqual(marker_pair.marker_a.center_px[0], 445.0, delta=2.0)
        self.assertAlmostEqual(marker_pair.marker_a.center_px[1], 230.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
