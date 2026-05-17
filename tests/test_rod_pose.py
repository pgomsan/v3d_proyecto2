from __future__ import annotations

import unittest

from pose.rod_pose import RodPoseEstimator, ToolMarkerPair3D, ToolParameters


class RodPoseTests(unittest.TestCase):
    def test_pose_position_is_midpoint_between_markers(self) -> None:
        estimator = RodPoseEstimator(
            ToolParameters(
                tool_id="boli_01",
                tool_type="boligrafo",
                length_cm=None,
                marker_distance_cm=8.0,
                tip_offset_cm=(0.0, 0.0, 0.0),
            )
        )
        markers = ToolMarkerPair3D(
            marker_a_cm=(1.0, 2.0, 3.0),
            marker_b_cm=(9.0, 2.0, 3.0),
            confidence=0.8,
        )

        pose = estimator.estimate_from_markers(markers)

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose.position_cm, (5.0, 2.0, 3.0))
        self.assertEqual(pose.marker_center_cm, (5.0, 2.0, 3.0))
        self.assertEqual(pose.position_reference, "tool_tip")
        self.assertEqual(pose.direction, (1.0, 0.0, 0.0))
        self.assertEqual(pose.confidence, 0.8)

    def test_payload_contains_tip_position_and_marker_distance(self) -> None:
        estimator = RodPoseEstimator(
            ToolParameters(
                tool_id="boli_01",
                tool_type="boligrafo",
                length_cm=None,
                marker_distance_cm=8.0,
                tip_offset_cm=(4.0, 0.0, 0.0),
            )
        )
        markers = ToolMarkerPair3D(
            marker_a_cm=(1.0, 2.0, 3.0),
            marker_b_cm=(9.0, 2.0, 3.0),
            confidence=0.8,
            marker_pixels={"a_left_px": [10.0, 20.0]},
        )
        pose = estimator.estimate_from_markers(markers)
        assert pose is not None

        payload = estimator.build_payload(pose, markers)

        self.assertEqual(payload["position_reference"], "tool_tip")
        self.assertEqual(payload["position_cm"], [9.0, 2.0, 3.0])
        self.assertEqual(payload["tip_position_cm"], [9.0, 2.0, 3.0])
        self.assertEqual(payload["marker_distance_cm"], 8.0)
        self.assertEqual(payload["markers"]["center_3d_cm"], [5.0, 2.0, 3.0])
        self.assertEqual(payload["markers"]["a_left_px"], [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()
