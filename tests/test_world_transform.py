from __future__ import annotations

import unittest

from calibration.world_transform import _reset_world_rotation_corrections


class WorldTransformTests(unittest.TestCase):
    def test_reset_world_rotation_corrections(self) -> None:
        config = {
            "calibration": {
                "world_roll_deg": 90.0,
                "world_pitch_deg": 180.0,
                "world_yaw_deg": 270.0,
                "world_flip_z": True,
                "world_scale": 3.0,
            }
        }

        changed = _reset_world_rotation_corrections(config)

        self.assertTrue(changed)
        self.assertEqual(config["calibration"]["world_roll_deg"], 0.0)
        self.assertEqual(config["calibration"]["world_pitch_deg"], 0.0)
        self.assertEqual(config["calibration"]["world_yaw_deg"], 0.0)
        self.assertTrue(config["calibration"]["world_flip_z"])
        self.assertEqual(config["calibration"]["world_scale"], 3.0)


if __name__ == "__main__":
    unittest.main()
