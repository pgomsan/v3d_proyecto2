from __future__ import annotations

import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from viewer.ur5_viewer import _step_joints_towards


@unittest.skipIf(np is None, "NumPy no esta instalado")
class ViewerSceneTests(unittest.TestCase):
    def test_step_towards_preserves_monotonic_progress(self) -> None:
        current = np.array([0.0, 0.0, 0.0])
        target = np.array([0.5, -0.5, 0.25])

        next_q = _step_joints_towards(current, target, 0.2)
        next_q2 = _step_joints_towards(next_q, target, 0.2)

        self.assertTrue(np.all(np.abs(target - next_q2) < np.abs(target - current)))


if __name__ == "__main__":
    unittest.main()
