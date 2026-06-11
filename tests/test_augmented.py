from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2  # noqa: F401
    import roboticstoolbox  # noqa: F401

    _DEPS = True
except ModuleNotFoundError:
    _DEPS = False

if _DEPS:
    import cv2

    from augmented_reality import UR5_MAX_REACH_M, AugmentedRobot


class _StubStereo:
    """Stub con solo p_left (matriz de proyeccion 3x4 de la camara rectificada)."""

    def __init__(self, camera_matrix: np.ndarray) -> None:
        projection = np.zeros((3, 4), dtype=float)
        projection[:3, :3] = camera_matrix
        self.p_left = projection


@unittest.skipUnless(_DEPS, "Requiere cv2 y roboticstoolbox")
class AugmentedRealityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        # mundo<-camara: rotacion + traslacion (cm) arbitrarias pero validas.
        self.R = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0]
        self.t = np.array([5.0, -3.0, 40.0])
        self._tmp = tempfile.TemporaryDirectory()
        self.npz = Path(self._tmp.name) / "world_transform.npz"
        np.savez(
            self.npz,
            rotation=self.R,
            translation=self.t,
            rms_reprojection=0.5,
            captured_at="test",
        )
        self.ar = AugmentedRobot(_StubStereo(self.K), self.npz)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_disponible(self) -> None:
        self.assertTrue(self.ar.available)

    def test_roundtrip_mundo_camara_proyecta_consistente(self) -> None:
        # Un punto en frame camara (cm) -> mundo -> proyectado debe coincidir
        # con la proyeccion directa K @ Pc (el inverso PnP cancela la ida).
        point_cam_cm = np.array([2.0, 1.0, 30.0])
        world_cm = self.ar._camera_cm_to_world_cm(point_cam_cm)
        pixel = self.ar.project_world_cm(world_cm)[0]
        expected = self.K @ point_cam_cm
        expected = expected[:2] / expected[2]
        np.testing.assert_allclose(pixel, expected, atol=1e-3)

    def test_tcp_sigue_la_herramienta(self) -> None:
        # Herramienta dentro del alcance: el TCP proyectado cae sobre ella.
        position_cm = [20.0, 10.0, 30.0]
        self.ar.update({"position_cm": position_cm})
        self.assertTrue(self.ar._has_pose)
        world_cm = self.ar._camera_cm_to_world_cm(position_cm)
        self.assertLess(float(np.linalg.norm(world_cm / 100.0)), UR5_MAX_REACH_M)
        target_px = self.ar.project_world_cm(world_cm)[0]
        tcp_px = self.ar._joint_pixels()[-1]
        self.assertLess(float(np.linalg.norm(tcp_px - target_px)), 8.0)

    def test_render_smoke(self) -> None:
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Sin pose: solo rejilla + ejes + aviso.
        out = self.ar.render(frame)
        self.assertEqual(out.shape, frame.shape)
        # Con pose: robot completo con sombra y reticula.
        out = self.ar.render(frame, {"position_cm": [20.0, 10.0, 30.0]})
        self.assertEqual(out.shape, frame.shape)
        self.assertTrue(self.ar._has_pose)
        # El frame original no se modifica (render trabaja sobre copia).
        self.assertTrue((frame == 120).all())

    def test_sin_world_transform_no_disponible(self) -> None:
        ar = AugmentedRobot(_StubStereo(self.K), Path(self._tmp.name) / "no_existe.npz")
        self.assertFalse(ar.available)


if __name__ == "__main__":
    unittest.main()
