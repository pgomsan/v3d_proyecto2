from __future__ import annotations

import unittest

from pose.tracking import TemporalPoseFilter


def _payload(
    position: tuple[float, float, float],
    rotation: list[list[float]] | None = None,
) -> dict:
    return {
        "position_cm": list(position),
        "tip_position_cm": list(position),
        "direction": [1.0, 0.0, 0.0],
        "orientation": {
            "format": "rotation_matrix",
            "value": rotation
            or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    }


class TemporalPoseFilterTests(unittest.TestCase):
    def test_rejects_isolated_position_jump(self) -> None:
        filter_ = TemporalPoseFilter(
            smoothing_alpha=1.0,
            max_position_jump_cm=5.0,
            max_orientation_jump_deg=35.0,
            reacquire_frames=3,
        )

        self.assertTrue(filter_.filter(_payload((0.0, 0.0, 0.0))).accepted)
        jumped = filter_.filter(_payload((20.0, 0.0, 0.0)))

        self.assertFalse(jumped.accepted)
        self.assertIsNone(jumped.payload)
        self.assertIn("descartado", jumped.reason)

    def test_reacquires_after_consistent_jumps(self) -> None:
        filter_ = TemporalPoseFilter(
            smoothing_alpha=1.0,
            max_position_jump_cm=5.0,
            max_orientation_jump_deg=35.0,
            reacquire_frames=3,
        )
        filter_.filter(_payload((0.0, 0.0, 0.0)))

        first = filter_.filter(_payload((20.0, 0.0, 0.0)))
        second = filter_.filter(_payload((21.0, 0.0, 0.0)))
        third = filter_.filter(_payload((22.0, 0.0, 0.0)))

        self.assertFalse(first.accepted)
        self.assertFalse(second.accepted)
        self.assertTrue(third.accepted)
        self.assertIn("reset", third.reason)

    def test_smooths_accepted_position(self) -> None:
        filter_ = TemporalPoseFilter(
            smoothing_alpha=0.25,
            max_position_jump_cm=20.0,
        )
        filter_.filter(_payload((0.0, 0.0, 0.0)))

        result = filter_.filter(_payload((4.0, 0.0, 0.0)))

        self.assertTrue(result.accepted)
        assert result.payload is not None
        self.assertEqual(result.payload["position_cm"], [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
