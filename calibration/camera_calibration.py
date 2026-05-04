from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app_state import load_config, save_calibration_info

try:
    import cv2 as cv
except ModuleNotFoundError:
    cv = None


BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = BASE_DIR / "data" / "calibration_images"
IMAGE_NAME_RE = re.compile(r"^(left|right)_(\d+)\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)
# OpenCV usa esquinas internas: un tablero de 8x8 cuadrados tiene 7x7 esquinas.
DEFAULT_PATTERN_SIZE = (7, 7)
DEFAULT_SQUARE_SIZE = 1.0


@dataclass
class CalibrationDataset:
    pattern_size: tuple[int, int]
    square_size: float
    object_points: list[np.ndarray]
    left_points: list[np.ndarray]
    right_points: list[np.ndarray]
    image_size: tuple[int, int]
    total_pairs: int
    skipped_pairs: list[str]
    reordered_pairs: list[int]


def _project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def _candidate_pattern_sizes(config: dict[str, Any]) -> list[tuple[int, int]]:
    calibration = config.get("calibration", {})
    raw = calibration.get("chessboard_pattern_size", DEFAULT_PATTERN_SIZE)
    try:
        configured = (int(raw[0]), int(raw[1]))
    except (TypeError, ValueError, IndexError):
        configured = DEFAULT_PATTERN_SIZE

    candidates = []
    for pattern_size in [configured, (configured[1], configured[0]), (9, 6), (6, 9)]:
        if pattern_size not in candidates:
            candidates.append(pattern_size)
    return candidates


def _square_size(config: dict[str, Any]) -> float:
    calibration = config.get("calibration", {})
    try:
        return float(calibration.get("chessboard_square_size", DEFAULT_SQUARE_SIZE))
    except (TypeError, ValueError):
        return DEFAULT_SQUARE_SIZE


def _make_object_points(
    pattern_size: tuple[int, int], square_size: float
) -> np.ndarray:
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0 : pattern_size[0], 0 : pattern_size[1]
    ].T.reshape(-1, 2)
    return objp * square_size


def _indexed_images(side: str) -> dict[int, Path]:
    images: dict[int, Path] = {}
    if not IMAGE_DIR.exists():
        return images

    for path in IMAGE_DIR.iterdir():
        match = IMAGE_NAME_RE.match(path.name)
        if match is None or match.group(1).lower() != side:
            continue
        images[int(match.group(2))] = path
    return images


def _image_pairs() -> list[tuple[int, Path, Path]]:
    left_images = _indexed_images("left")
    right_images = _indexed_images("right")
    indices = sorted(set(left_images) & set(right_images))
    return [(index, left_images[index], right_images[index]) for index in indices]


def _get_corners(image: np.ndarray, pattern_size: tuple[int, int]):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv.findChessboardCorners(gray, pattern_size, flags)

    if not found:
        return False, None

    criteria = (
        cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS,
        30,
        0.01,
    )
    refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def _align_corner_order(
    reference_corners: np.ndarray, candidate_corners: np.ndarray
) -> tuple[np.ndarray, bool]:
    reference_axis = reference_corners[-1, 0] - reference_corners[0, 0]
    candidate_axis = candidate_corners[-1, 0] - candidate_corners[0, 0]
    if float(np.dot(reference_axis, candidate_axis)) < 0.0:
        return candidate_corners[::-1].copy(), True
    return candidate_corners, False


def _collect_dataset(
    pattern_size: tuple[int, int], square_size: float
) -> CalibrationDataset:
    pairs = _image_pairs()
    if not pairs:
        raise FileNotFoundError(
            f"No hay pares left_XX/right_XX en {IMAGE_DIR}. "
            "Primero ejecuta 'Capturar imagenes de calibracion'."
        )

    object_template = _make_object_points(pattern_size, square_size)
    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    skipped_pairs: list[str] = []
    reordered_pairs: list[int] = []
    image_size: tuple[int, int] | None = None

    for index, left_path, right_path in pairs:
        left_image = cv.imread(str(left_path))
        right_image = cv.imread(str(right_path))
        if left_image is None or right_image is None:
            skipped_pairs.append(f"{index:02d}: imagen no legible")
            continue

        current_size = (left_image.shape[1], left_image.shape[0])
        if image_size is None:
            image_size = current_size

        if current_size != image_size:
            skipped_pairs.append(f"{index:02d}: tamano izquierdo distinto")
            continue
        if (right_image.shape[1], right_image.shape[0]) != image_size:
            skipped_pairs.append(f"{index:02d}: tamano derecho distinto")
            continue

        left_found, left_corners = _get_corners(left_image, pattern_size)
        right_found, right_corners = _get_corners(right_image, pattern_size)
        if not left_found or not right_found:
            skipped_pairs.append(f"{index:02d}: tablero no detectado en ambas camaras")
            continue

        right_corners, reordered = _align_corner_order(left_corners, right_corners)
        if reordered:
            reordered_pairs.append(index)

        object_points.append(object_template.copy())
        left_points.append(left_corners)
        right_points.append(right_corners)

    if image_size is None:
        image_size = (0, 0)

    return CalibrationDataset(
        pattern_size=pattern_size,
        square_size=square_size,
        object_points=object_points,
        left_points=left_points,
        right_points=right_points,
        image_size=image_size,
        total_pairs=len(pairs),
        skipped_pairs=skipped_pairs,
        reordered_pairs=reordered_pairs,
    )


def _best_dataset(config: dict[str, Any]) -> CalibrationDataset:
    square_size = _square_size(config)
    datasets = [
        _collect_dataset(pattern_size, square_size)
        for pattern_size in _candidate_pattern_sizes(config)
    ]
    return max(datasets, key=lambda dataset: len(dataset.object_points))


def _mean_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    total_error = 0.0
    for index, objp in enumerate(object_points):
        projected, _ = cv.projectPoints(
            objp, rvecs[index], tvecs[index], camera_matrix, distortion
        )
        error = cv.norm(image_points[index], projected, cv.NORM_L2) / len(projected)
        total_error += float(error)
    return total_error / len(object_points)


def calibrate_camera() -> None:
    """Calibra ambas camaras y la relacion estereo usando pares de tablero."""
    if cv is None:
        raise RuntimeError("OpenCV no esta instalado. Instala opencv-contrib-python.")

    config = load_config()
    calibration = config.get("calibration", {})
    dataset = _best_dataset(config)
    valid_pairs = len(dataset.object_points)
    if valid_pairs < 3:
        raise RuntimeError(
            "No hay suficientes pares validos para calibrar. "
            f"Detectados {valid_pairs} de {dataset.total_pairs}. "
            "Captura mas imagenes donde el tablero se vea completo en ambas camaras."
        )

    print(
        "Calibrando con "
        f"{valid_pairs}/{dataset.total_pairs} pares validos, "
        f"tablero {dataset.pattern_size[0]}x{dataset.pattern_size[1]}, "
        f"tamano de imagen {dataset.image_size}."
    )

    left_rms, k_left, dist_left, rvecs_left, tvecs_left = cv.calibrateCamera(
        dataset.object_points,
        dataset.left_points,
        dataset.image_size,
        None,
        None,
    )
    right_rms, k_right, dist_right, rvecs_right, tvecs_right = cv.calibrateCamera(
        dataset.object_points,
        dataset.right_points,
        dataset.image_size,
        None,
        None,
    )

    criteria = (cv.TERM_CRITERIA_MAX_ITER | cv.TERM_CRITERIA_EPS, 100, 1e-5)
    flags = cv.CALIB_FIX_INTRINSIC
    stereo_rms, k_left_st, dist_left_st, k_right_st, dist_right_st, r_st, t_st, e_st, f_st = (
        cv.stereoCalibrate(
            dataset.object_points,
            dataset.left_points,
            dataset.right_points,
            k_left,
            dist_left,
            k_right,
            dist_right,
            dataset.image_size,
            criteria=criteria,
            flags=flags,
        )
    )

    p_left = k_left_st @ np.hstack((np.eye(3), np.zeros((3, 1))))
    p_right = k_right_st @ np.hstack((r_st, t_st))

    k_left_path = _project_path(
        calibration.get("camera_matrix_left_path", "calibration/K_left.npy")
    )
    dist_left_path = _project_path(
        calibration.get("distortion_left_path", "calibration/dist_left.npy")
    )
    k_right_path = _project_path(
        calibration.get("camera_matrix_right_path", "calibration/K_right.npy")
    )
    dist_right_path = _project_path(
        calibration.get("distortion_right_path", "calibration/dist_right.npy")
    )
    stereo_path = _project_path(
        calibration.get("stereo_calibration_path", "calibration/stereo.npz")
    )

    for path in [k_left_path, dist_left_path, k_right_path, dist_right_path, stereo_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    np.save(k_left_path, k_left_st)
    np.save(dist_left_path, dist_left_st)
    np.save(k_right_path, k_right_st)
    np.save(dist_right_path, dist_right_st)
    np.savez(
        stereo_path,
        rms=stereo_rms,
        R=r_st,
        T=t_st,
        E=e_st,
        F=f_st,
        P_left=p_left,
        P_right=p_right,
        K_left=k_left_st,
        dist_left=dist_left_st,
        K_right=k_right_st,
        dist_right=dist_right_st,
        image_size=np.array(dataset.image_size),
        pattern_size=np.array(dataset.pattern_size),
        square_size=np.array(dataset.square_size),
    )

    left_mean_error = _mean_reprojection_error(
        dataset.object_points, dataset.left_points, rvecs_left, tvecs_left, k_left, dist_left
    )
    right_mean_error = _mean_reprojection_error(
        dataset.object_points,
        dataset.right_points,
        rvecs_right,
        tvecs_right,
        k_right,
        dist_right,
    )

    save_calibration_info(
        {
            "method": "individual_intrinsics_plus_stereo_calibrate",
            "calibration_images_dir": str(IMAGE_DIR.relative_to(BASE_DIR)),
            "valid_pairs": valid_pairs,
            "total_pairs": dataset.total_pairs,
            "skipped_pairs": dataset.skipped_pairs,
            "reordered_pairs": dataset.reordered_pairs,
            "pattern_size": list(dataset.pattern_size),
            "square_size": dataset.square_size,
            "image_size": list(dataset.image_size),
            "left_rms": float(left_rms),
            "right_rms": float(right_rms),
            "left_mean_reprojection_error": float(left_mean_error),
            "right_mean_reprojection_error": float(right_mean_error),
            "stereo_rms": float(stereo_rms),
            "camera_matrix_left_path": str(k_left_path.relative_to(BASE_DIR)),
            "distortion_left_path": str(dist_left_path.relative_to(BASE_DIR)),
            "camera_matrix_right_path": str(k_right_path.relative_to(BASE_DIR)),
            "distortion_right_path": str(dist_right_path.relative_to(BASE_DIR)),
            "stereo_calibration_path": str(stereo_path.relative_to(BASE_DIR)),
        }
    )

    print("Calibracion individual y estereo completada.")
    print(f"RMS izquierda: {left_rms:.6f}")
    print(f"RMS derecha: {right_rms:.6f}")
    print(f"RMS estereo: {stereo_rms:.6f}")
    print(f"Guardado: {stereo_path.relative_to(BASE_DIR)}")
