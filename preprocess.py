"""
preprocess.py - Multi-sensor preprocessing and multi-scale representations for LUNA-FUSE.

Handles:
- Preserving scientific original image while generating matching representation.
- Dynamic range normalization (percentile clipping for high-bitrate lunar sensors).
- Grayscale extraction.
- Sensor-specific tuning for OHRC, TMC-2, and IIRS hyperspectral data.
- Multi-scale pyramid generation for cross-resolution scale bridging.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from product_loader import ImageRecord


def robust_percentile_norm(img_f: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Normalize floating/integer array to uint8 [0, 255] using robust percentile clipping."""
    # Filter out NaNs and infinities if any
    valid_mask = np.isfinite(img_f)
    if not np.any(valid_mask):
        return np.zeros(img_f.shape[:2], dtype=np.uint8)

    v_min = float(np.percentile(img_f[valid_mask], p_low))
    v_max = float(np.percentile(img_f[valid_mask], p_high))

    if v_max <= v_min:
        v_min = float(np.min(img_f[valid_mask]))
        v_max = float(np.max(img_f[valid_mask]))

    if v_max > v_min:
        clipped = np.clip(img_f, v_min, v_max)
        norm = ((clipped - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
    else:
        norm = np.zeros_like(img_f, dtype=np.uint8)

    return norm


def extract_iirs_structural_proxy(img_array: np.ndarray) -> np.ndarray:
    """
    Handle IIRS hyperspectral cube data without treating it as ordinary RGB.
    If 3D (bands x H x W or H x W x bands), extracts a high-SNR structural continuum
    proxy via robust spectral band aggregation.
    """
    if img_array.ndim == 2:
        return img_array.astype(np.float32)

    # 3D Hyperspectral cube
    if img_array.ndim == 3:
        # Determine if band axis is 0 or 2
        if img_array.shape[0] < img_array.shape[1] and img_array.shape[0] < img_array.shape[2]:
            # Bands first: (C, H, W)
            # Use middle 50% of bands (highest SNR continuum away from sensor edge cutoff)
            num_bands = img_array.shape[0]
            b_start = num_bands // 4
            b_end = 3 * num_bands // 4
            subset = img_array[b_start:b_end, :, :].astype(np.float32)
            structural = np.nanmedian(subset, axis=0)
            return structural
        else:
            # Bands last: (H, W, C)
            num_bands = img_array.shape[2]
            if num_bands in [3, 4]:
                # Conventional 3 or 4 channel raster
                return cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            b_start = num_bands // 4
            b_end = 3 * num_bands // 4
            subset = img_array[:, :, b_start:b_end].astype(np.float32)
            structural = np.nanmedian(subset, axis=2)
            return structural

    return img_array.squeeze().astype(np.float32)


def to_matching_grayscale(raw_image: np.ndarray, sensor: str) -> np.ndarray:
    """Convert any raw image array to an 8-bit normalized grayscale base."""
    sensor_clean = sensor.upper()

    if "IIRS" in sensor_clean:
        proxy_2d = extract_iirs_structural_proxy(raw_image)
        return robust_percentile_norm(proxy_2d, p_low=2.0, p_high=98.0)

    # 2D or standard multichannel image
    if raw_image.ndim == 3:
        if raw_image.shape[2] == 3:
            gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        elif raw_image.shape[2] == 4:
            gray = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = np.mean(raw_image, axis=2)
    else:
        gray = raw_image

    return robust_percentile_norm(gray.astype(np.float32), p_low=1.0, p_high=99.0)


def preprocess_for_matching(gray_base: np.ndarray, sensor: str) -> np.ndarray:
    """Apply sensor-tuned enhancement (CLAHE, crater edge enhancement, shadow mitigation)."""
    sensor_clean = sensor.upper()

    if "OHRC" in sensor_clean:
        # High resolution, strong shadows: enhance local contrast while suppressing specular noise
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        proc = clahe.apply(gray_base)
        proc = cv2.GaussianBlur(proc, (3, 3), 0.5)
        return proc

    elif "TMC" in sensor_clean:
        # Medium resolution stereo/ortho
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        proc = clahe.apply(gray_base)
        proc = cv2.GaussianBlur(proc, (3, 3), 0.5)
        return proc

    elif "IIRS" in sensor_clean:
        # Lower spatial resolution hyperspectral: enhance morphological features and crater rims
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6, 6))
        proc = clahe.apply(gray_base)
        proc = cv2.bilateralFilter(proc, d=5, sigmaColor=35, sigmaSpace=35)
        return proc

    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray_base)


def preprocess_record(record: ImageRecord) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process ImageRecord into:
    1. gray_raw: Clean scientific 8-bit normalized grayscale image (for visual inspection/fusion).
    2. proc: Feature-enhanced matching image (for keypoint & descriptor extraction).
    Does NOT modify record.image.
    """
    gray_raw = to_matching_grayscale(record.image, record.sensor)
    proc = preprocess_for_matching(gray_raw, record.sensor)
    return gray_raw, proc


def generate_scale_pyramid(img: np.ndarray, scales: List[float]) -> List[Tuple[float, np.ndarray]]:
    """
    Generate scale pyramid for cross-sensor resolution bridging.
    Returns list of (scale_factor, resized_image).
    """
    pyramid = []
    h, w = img.shape[:2]
    for s in scales:
        if s == 1.0:
            pyramid.append((1.0, img))
        else:
            nw = max(16, int(w * s))
            nh = max(16, int(h * s))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            pyramid.append((s, resized))
    return pyramid
