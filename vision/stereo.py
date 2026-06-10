from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app_state import BASE_DIR, load_config

try:
    import cv2 as cv
    import numpy as np
except ModuleNotFoundError:
    cv = None
    np = None


Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def epipolar_error_px(left_px: Point2, right_px: Point2) -> float:
    """Error vertical entre correspondencias de imagenes rectificadas."""
    return abs(float(left_px[1]) - float(right_px[1]))


def epipolar_errors_px(
    correspondences: dict[str, tuple[Point2, Point2]],
) -> dict[str, float]:
    """Calcula el error epipolar de cada correspondencia identificada."""
    return {
        name: epipolar_error_px(left_px, right_px)
        for name, (left_px, right_px) in correspondences.items()
    }


def epipolar_errors_are_valid(
    errors_px: dict[str, float],
    required_markers: tuple[str, ...],
    max_error_px: float,
) -> bool:
    """Valida que existan todos los errores y no superen el umbral."""
    return all(
        marker in errors_px and errors_px[marker] <= max_error_px
        for marker in required_markers
    )


@dataclass
class StereoCalibration:
    p_left: Any
    p_right: Any
    left_map_x: Any
    left_map_y: Any
    right_map_x: Any
    right_map_y: Any
    image_size: tuple[int, int]
    rms: float

    def rectify_pair(self, left_frame: Any, right_frame: Any) -> tuple[Any, Any]:
        """Rectifica un par de frames usando los mapas de calibracion estereo."""
        if cv is None:
            raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

        left_rect = cv.remap(left_frame, self.left_map_x, self.left_map_y, cv.INTER_LINEAR)
        right_rect = cv.remap(
            right_frame, self.right_map_x, self.right_map_y, cv.INTER_LINEAR
        )
        return left_rect, right_rect

    def triangulate_point(self, left_px: Point2, right_px: Point2) -> Point3:
        """Triangula un punto rectificado visto por ambas camaras."""
        if cv is None or np is None:
            raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

        left_points = np.array([[left_px]], dtype=np.float64)
        right_points = np.array([[right_px]], dtype=np.float64)
        homogeneous = cv.triangulatePoints(
            self.p_left,
            self.p_right,
            left_points.reshape(2, 1),
            right_points.reshape(2, 1),
        )
        if abs(float(homogeneous[3, 0])) < 1e-12:
            raise ValueError("Triangulacion invalida: coordenada homogenea cero.")

        point = homogeneous[:3, 0] / homogeneous[3, 0]
        return (float(point[0]), float(point[1]), float(point[2]))

    def project_left_point(self, point_cm: Point3) -> Point2:
        """Proyecta un punto 3D sobre la imagen izquierda rectificada."""
        return _project_point(self.p_left, point_cm)

    def project_right_point(self, point_cm: Point3) -> Point2:
        """Proyecta un punto 3D sobre la imagen derecha rectificada."""
        return _project_point(self.p_right, point_cm)


def _project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def _project_point(projection_matrix: Any, point_cm: Point3) -> Point2:
    if np is None:
        raise RuntimeError("NumPy no esta instalado.")

    homogeneous = np.array(
        [point_cm[0], point_cm[1], point_cm[2], 1.0], dtype=np.float64
    )
    projected = projection_matrix @ homogeneous
    if abs(float(projected[2])) < 1e-12:
        raise ValueError("Proyeccion invalida: profundidad cero.")
    return (float(projected[0] / projected[2]), float(projected[1] / projected[2]))


def load_stereo_calibration() -> StereoCalibration:
    """Carga `calibration/stereo.npz` y prepara mapas de rectificacion."""
    if cv is None or np is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    config = load_config()
    calibration_config = config.get("calibration", {})
    stereo_path = _project_path(
        calibration_config.get("stereo_calibration_path", "calibration/stereo.npz")
    )
    if not stereo_path.exists():
        raise FileNotFoundError(
            f"No existe {stereo_path.relative_to(BASE_DIR)}. Ejecuta primero la calibracion."
        )

    data = np.load(stereo_path)
    image_size = tuple(int(value) for value in data["image_size"])
    k_left = data["K_left"]
    dist_left = data["dist_left"]
    k_right = data["K_right"]
    dist_right = data["dist_right"]
    r_left_rect = data["R_left_rect"]
    r_right_rect = data["R_right_rect"]
    p_left = data["P_left"]
    p_right = data["P_right"]

    left_map_x, left_map_y = cv.initUndistortRectifyMap(
        k_left, dist_left, r_left_rect, p_left, image_size, cv.CV_32FC1
    )
    right_map_x, right_map_y = cv.initUndistortRectifyMap(
        k_right, dist_right, r_right_rect, p_right, image_size, cv.CV_32FC1
    )

    return StereoCalibration(
        p_left=p_left,
        p_right=p_right,
        left_map_x=left_map_x,
        left_map_y=left_map_y,
        right_map_x=right_map_x,
        right_map_y=right_map_y,
        image_size=image_size,
        rms=float(data["rms"]),
    )
