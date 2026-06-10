from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from vision.stereo import (
    StereoCalibration,
    epipolar_error_px,
    epipolar_errors_are_valid,
    epipolar_errors_px,
)


@unittest.skipIf(np is None, "NumPy no esta instalado")
class StereoTests(unittest.TestCase):
    def test_epipolar_validation(self) -> None:
        errors = epipolar_errors_px(
            {
                "a": ((10.0, 20.0), (30.0, 22.0)),
                "b": ((40.0, 50.0), (60.0, 53.5)),
                "c": ((70.0, 80.0), (90.0, 84.5)),
            }
        )

        self.assertEqual(epipolar_error_px((0.0, 3.0), (5.0, 7.0)), 4.0)
        self.assertEqual(errors, {"a": 2.0, "b": 3.5, "c": 4.5})
        self.assertFalse(
            epipolar_errors_are_valid(errors, ("a", "b", "c"), 4.0)
        )
        self.assertTrue(
            epipolar_errors_are_valid(errors, ("a", "b"), 4.0)
        )
        self.assertFalse(
            epipolar_errors_are_valid({"a": 1.0}, ("a", "b"), 4.0)
        )

    def test_project_left_point(self) -> None:
        projection = np.array(
            [
                [100.0, 0.0, 50.0, 0.0],
                [0.0, 100.0, 30.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        stereo = StereoCalibration(
            p_left=projection,
            p_right=projection,
            left_map_x=None,
            left_map_y=None,
            right_map_x=None,
            right_map_y=None,
            image_size=(100, 100),
            rms=0.0,
        )

        point = stereo.project_left_point((2.0, 4.0, 10.0))

        self.assertEqual(point, (70.0, 70.0))


if __name__ == "__main__":
    unittest.main()
