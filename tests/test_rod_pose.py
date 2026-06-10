from __future__ import annotations

import unittest

from pose.rod_pose import RodPoseEstimator, ToolMarkerTriplet3D, ToolParameters


class RodPoseTests(unittest.TestCase):
    def test_pose_position_is_marker_a_tool_tip(self) -> None:
        estimator = RodPoseEstimator(
            ToolParameters(
                tool_id="boli_01",
                tool_type="boligrafo",
                length_cm=None,
                marker_distance_cm=15.0,
                tip_offset_cm=(0.0, 0.0, 0.0),
                marker_c_along_ab_cm=5.0,
                marker_c_offset_cm=5.0,
            )
        )
        markers = ToolMarkerTriplet3D(
            marker_a_cm=(1.0, 2.0, 3.0),
            marker_b_cm=(16.0, 2.0, 3.0),
            marker_c_cm=(6.0, 7.0, 3.0),
            confidence=0.8,
        )

        pose = estimator.estimate_from_markers(markers)

        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertEqual(pose.position_cm, (1.0, 2.0, 3.0))
        self.assertEqual(pose.marker_center_cm, (8.5, 2.0, 3.0))
        self.assertEqual(pose.position_reference, "marker_a_tool_tip")
        self.assertEqual(pose.direction, (1.0, 0.0, 0.0))
        self.assertEqual(pose.orientation["format"], "rotation_matrix")
        self.assertEqual(
            pose.orientation["value"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(pose.confidence, 0.8)

    def test_payload_contains_three_markers_and_distances(self) -> None:
        estimator = RodPoseEstimator(
            ToolParameters(
                tool_id="boli_01",
                tool_type="boligrafo",
                length_cm=None,
                marker_distance_cm=15.0,
                tip_offset_cm=(0.0, 0.0, 0.0),
                marker_c_along_ab_cm=5.0,
                marker_c_offset_cm=5.0,
            )
        )
        markers = ToolMarkerTriplet3D(
            marker_a_cm=(1.0, 2.0, 3.0),
            marker_b_cm=(16.0, 2.0, 3.0),
            marker_c_cm=(6.0, 7.0, 3.0),
            confidence=0.8,
            marker_pixels={
                "a_left_px": [10.0, 20.0],
                "c_right_px": [30.0, 40.0],
            },
        )
        pose = estimator.estimate_from_markers(markers)
        assert pose is not None

        payload = estimator.build_payload(pose, markers)

        self.assertEqual(payload["position_reference"], "marker_a_tool_tip")
        self.assertEqual(payload["position_cm"], [1.0, 2.0, 3.0])
        self.assertEqual(payload["tip_position_cm"], [1.0, 2.0, 3.0])
        self.assertEqual(payload["marker_distance_cm"], 15.0)
        self.assertAlmostEqual(payload["marker_distances_cm"]["ac"], 50.0**0.5)
        self.assertAlmostEqual(payload["marker_distances_cm"]["bc"], 125.0**0.5)
        self.assertEqual(payload["markers"]["center_3d_cm"], [8.5, 2.0, 3.0])
        self.assertEqual(payload["markers"]["c_3d_cm"], [6.0, 7.0, 3.0])
        self.assertEqual(payload["markers"]["a_left_px"], [10.0, 20.0])
        self.assertEqual(payload["markers"]["c_right_px"], [30.0, 40.0])

    def test_rejects_triplet_with_invalid_c_geometry(self) -> None:
        estimator = RodPoseEstimator(
            ToolParameters(
                tool_id="boli_01",
                tool_type="boligrafo",
                length_cm=None,
                marker_distance_cm=15.0,
                tip_offset_cm=(0.0, 0.0, 0.0),
                marker_c_along_ab_cm=5.0,
                marker_c_offset_cm=5.0,
                marker_distance_tolerance_ratio=0.1,
            )
        )
        markers = ToolMarkerTriplet3D(
            marker_a_cm=(0.0, 0.0, 0.0),
            marker_b_cm=(15.0, 0.0, 0.0),
            marker_c_cm=(5.0, 10.0, 0.0),
            confidence=1.0,
        )

        self.assertIsNone(estimator.estimate_from_markers(markers))


if __name__ == "__main__":
    unittest.main()
