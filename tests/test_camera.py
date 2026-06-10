from __future__ import annotations

import unittest

from vision.camera import read_stereo_pair


class _FakeCamera:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def grab(self) -> None:
        self.calls.append(f"{self.name}.grab")

    def retrieve(self) -> str:
        self.calls.append(f"{self.name}.retrieve")
        return f"{self.name}_frame"


class CameraTests(unittest.TestCase):
    def test_stereo_pair_grabs_both_before_retrieving(self) -> None:
        calls: list[str] = []
        left = _FakeCamera("left", calls)
        right = _FakeCamera("right", calls)

        frames = read_stereo_pair(left, right)

        self.assertEqual(frames, ("left_frame", "right_frame"))
        self.assertEqual(
            calls,
            [
                "left.grab",
                "right.grab",
                "left.retrieve",
                "right.retrieve",
            ],
        )


if __name__ == "__main__":
    unittest.main()
