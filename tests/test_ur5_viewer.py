from __future__ import annotations

import math
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from viewer.ur5_viewer import _nearest_equivalent_joints, _step_joints_towards


@unittest.skipIf(np is None, "NumPy no esta instalado")
class UR5ViewerTests(unittest.TestCase):
    def test_nearest_equivalent_joints_avoids_full_turn(self) -> None:
        current = np.array([math.pi - 0.1])
        target = np.array([-math.pi + 0.1])

        nearest = _nearest_equivalent_joints(target, current)

        self.assertAlmostEqual(float(nearest[0] - current[0]), 0.2)

    def test_joint_step_is_limited(self) -> None:
        current = np.array([0.0, 0.0])
        target = np.array([1.0, -1.0])

        stepped = _step_joints_towards(current, target, 0.1)

        np.testing.assert_allclose(stepped, [0.1, -0.1])

    def test_repeated_steps_reach_joint_target(self) -> None:
        current = np.array([0.0, 0.0])
        target = np.array([0.3, -0.3])

        for _ in range(3):
            current = _step_joints_towards(current, target, 0.1)

        np.testing.assert_allclose(current, target)


if __name__ == "__main__":
    unittest.main()
