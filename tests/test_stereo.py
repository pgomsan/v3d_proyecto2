from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from vision.stereo import StereoCalibration


@unittest.skipIf(np is None, "NumPy no esta instalado")
class StereoTests(unittest.TestCase):
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
