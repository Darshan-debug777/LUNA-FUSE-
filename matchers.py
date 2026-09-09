"""
matchers.py - The Four Matching Engines of LUNA-FUSE with controlled CPU execution policy:
1. SIFT (multi-scale capable)
2. SuperPoint (single optimal scale bridge, image-size protected)
3. LoFTR (single optimal scale bridge, dimension-aligned)
4. LightGlue (single optimal scale bridge, image-size protected)

Key Features:
- Execution time tracking per matcher.
- Image-size protection (bounds maximum dimension for CPU inference).
- torch.inference_mode() for minimal memory and maximum CPU speed.
- Configurable timeout protection.
- Precise coordinate mapping back to original scientific image coordinates.
"""

import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

# Ensure trusted Mozilla CA certificates for PyTorch Hub downloads
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CPU vs CUDA dimension bounds for deep models
CPU_MAX_DIM = 512
CUDA_MAX_DIM = 1024
DEFAULT_TIMEOUT_SEC = 60.0

from registration import verify_and_estimate_geometry, align_image
from verification import compute_confidence


@dataclass
class MatcherResult:
    """Standardized result object returned by all matching engines."""
    algorithm: str
    status: str  # "PASS", "FAILED", or "TIMEOUT"
    keypoints_reference: np.ndarray
    keypoints_target: np.ndarray
    correspondences: int
    inliers: int
    inlier_ratio: float
    rmse: Optional[float]
    spatial_coverage: float
    homography: Optional[np.ndarray]
    confidence: str
    registered_image: Optional[np.ndarray]
    elapsed_time: float = 0.0
    reason: Optional[str] = None


def get_primary_scale_pair(sensor_ref: str, sensor_tgt: str) -> Tuple[float, float]:
    """
    Select the single most effective common working scale pair for deep models.
    Avoids running heavy neural networks repeatedly across redundant scales.
    """
    sr = sensor_ref.upper()
    st = sensor_tgt.upper()

    if "OHRC" in sr and "IIRS" in st:
        return (0.5, 1.0)
    elif "OHRC" in sr and "TMC" in st:
        return (0.5, 1.0)
    elif "TMC" in sr and "IIRS" in st:
        return (0.5, 1.0)
    else:
        return (1.0, 1.0)


def get_sift_scale_pairs(sensor_ref: str, sensor_tgt: str) -> List[Tuple[float, float]]:
    """SIFT is lightweight on CPU, so it can evaluate candidate scale search pairs."""
    sr = sensor_ref.upper()
    st = sensor_tgt.upper()

    if "OHRC" in sr and "IIRS" in st:
        return [(0.5, 1.0), (0.25, 1.0), (1.0, 1.0)]
    elif "OHRC" in sr and "TMC" in st:
        return [(0.5, 1.0), (0.25, 1.0), (1.0, 1.0)]
    else:
        return [(1.0, 1.0), (0.5, 0.5), (0.5, 1.0)]


# -------------------------------------------------------------------------
# 1. SIFT Matching Engine
# -------------------------------------------------------------------------
def match_sift_core(ref_gray: np.ndarray, tgt_gray: np.ndarray, ratio: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
    """OpenCV SIFT with Lowe's ratio test."""
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.015, edgeThreshold=15)
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(tgt_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) < 4:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


# -------------------------------------------------------------------------
# 2. SuperPoint Matching Engine
# -------------------------------------------------------------------------
_superpoint_model = None

def get_superpoint_model():
    global _superpoint_model
    if _superpoint_model is None:
        from lightglue import SuperPoint
        # Max keypoints tuned for CPU responsiveness and high accuracy
        max_kpts = 512 if DEVICE == "cpu" else 1024
        _superpoint_model = SuperPoint(max_num_keypoints=max_kpts).eval().to(DEVICE)
    return _superpoint_model


def match_superpoint_core(ref_gray: np.ndarray, tgt_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """SuperPoint feature extraction with Mutual Nearest Neighbor matching."""
    sp = get_superpoint_model()

    h1, w1 = ref_gray.shape[:2]
    h2, w2 = tgt_gray.shape[:2]

    # Bound image size for CPU execution
    max_dim = CPU_MAX_DIM if DEVICE == "cpu" else CUDA_MAX_DIM
    scale1 = min(1.0, max_dim / max(h1, w1))
    scale2 = min(1.0, max_dim / max(h2, w2))

    r1 = cv2.resize(ref_gray, (int(w1 * scale1), int(h1 * scale1))) if scale1 < 1.0 else ref_gray
    r2 = cv2.resize(tgt_gray, (int(w2 * scale2), int(h2 * scale2))) if scale2 < 1.0 else tgt_gray

    t1 = torch.from_numpy(r1).float()[None, None].to(DEVICE) / 255.0
    t2 = torch.from_numpy(r2).float()[None, None].to(DEVICE) / 255.0

    with torch.inference_mode():
        feats1 = sp.extract(t1)
        feats2 = sp.extract(t2)

    desc1 = feats1["descriptors"][0]
    desc2 = feats2["descriptors"][0]

    if len(desc1) < 4 or len(desc2) < 4:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    # Cosine similarity matrix & Mutual Nearest Neighbor check
    sim = torch.einsum("nd,md->nm", desc1, desc2)
    nn12 = sim.argmax(dim=1)
    nn21 = sim.argmax(dim=0)
    ids1 = torch.arange(len(desc1), device=DEVICE)
    mutual = (nn21[nn12] == ids1)

    pts1 = feats1["keypoints"][0][ids1[mutual]].cpu().numpy() / scale1
    pts2 = feats2["keypoints"][0][nn12[mutual]].cpu().numpy() / scale2
    return pts1.astype(np.float32), pts2.astype(np.float32)


# -------------------------------------------------------------------------
# 3. LoFTR Matching Engine
# -------------------------------------------------------------------------
_loftr_model = None

def get_loftr_model():
    global _loftr_model
    if _loftr_model is None:
        import kornia.feature as KF
        _loftr_model = KF.LoFTR(pretrained="outdoor").eval().to(DEVICE)
    return _loftr_model


def match_loftr_core(ref_gray: np.ndarray, tgt_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LoFTR dense transformer matching."""
    loftr = get_loftr_model()

    h1, w1 = ref_gray.shape[:2]
    h2, w2 = tgt_gray.shape[:2]

    # LoFTR requires dimensions to be divisible by 8
    target_dim = 480 if DEVICE == "cpu" else 640
    scale1 = min(1.0, target_dim / max(h1, w1))
    scale2 = min(1.0, target_dim / max(h2, w2))

    nw1 = max(64, int(round(w1 * scale1 / 8.0) * 8))
    nh1 = max(64, int(round(h1 * scale1 / 8.0) * 8))
    nw2 = max(64, int(round(w2 * scale2 / 8.0) * 8))
    nh2 = max(64, int(round(h2 * scale2 / 8.0) * 8))

    r1 = cv2.resize(ref_gray, (nw1, nh1))
    r2 = cv2.resize(tgt_gray, (nw2, nh2))

    t1 = torch.from_numpy(r1).float()[None, None].to(DEVICE) / 255.0
    t2 = torch.from_numpy(r2).float()[None, None].to(DEVICE) / 255.0

    input_dict = {"image0": t1, "image1": t2}
    with torch.inference_mode():
        corresp = loftr(input_dict)

    pts1 = corresp["keypoints0"].cpu().numpy()
    pts2 = corresp["keypoints1"].cpu().numpy()

    if len(pts1) < 4:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    # Scale back to original input coordinates
    sx1 = w1 / float(nw1)
    sy1 = h1 / float(nh1)
    sx2 = w2 / float(nw2)
    sy2 = h2 / float(nh2)

    pts1[:, 0] *= sx1
    pts1[:, 1] *= sy1
    pts2[:, 0] *= sx2
    pts2[:, 1] *= sy2

    return pts1.astype(np.float32), pts2.astype(np.float32)


# -------------------------------------------------------------------------
# 4. LightGlue Matching Engine
# -------------------------------------------------------------------------
_lightglue_model = None

def get_lightglue_model():
    global _lightglue_model
    if _lightglue_model is None:
        from lightglue import LightGlue
        _lightglue_model = LightGlue(features="superpoint", depth_confidence=0.9, width_confidence=0.95).eval().to(DEVICE)
    return _lightglue_model


def match_lightglue_core(ref_gray: np.ndarray, tgt_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """LightGlue graph neural network matcher using SuperPoint features."""
    sp = get_superpoint_model()
    lg = get_lightglue_model()

    h1, w1 = ref_gray.shape[:2]
    h2, w2 = tgt_gray.shape[:2]

    max_dim = CPU_MAX_DIM if DEVICE == "cpu" else CUDA_MAX_DIM
    scale1 = min(1.0, max_dim / max(h1, w1))
    scale2 = min(1.0, max_dim / max(h2, w2))

    r1 = cv2.resize(ref_gray, (int(w1 * scale1), int(h1 * scale1))) if scale1 < 1.0 else ref_gray
    r2 = cv2.resize(tgt_gray, (int(w2 * scale2), int(h2 * scale2))) if scale2 < 1.0 else tgt_gray

    t1 = torch.from_numpy(r1).float()[None, None].to(DEVICE) / 255.0
    t2 = torch.from_numpy(r2).float()[None, None].to(DEVICE) / 255.0

    with torch.inference_mode():
        feats1 = sp.extract(t1)
        feats2 = sp.extract(t2)
        matches01 = lg({"image0": feats1, "image1": feats2})

    matches = matches01["matches"][0].cpu().numpy()
    kpts1 = feats1["keypoints"][0].cpu().numpy()
    kpts2 = feats2["keypoints"][0].cpu().numpy()

    if len(matches) < 4:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    pts1 = kpts1[matches[:, 0]] / scale1
    pts2 = kpts2[matches[:, 1]] / scale2
    return pts1.astype(np.float32), pts2.astype(np.float32)


# -------------------------------------------------------------------------
# Execution Framework with Timing & Policy Control
# -------------------------------------------------------------------------
def run_matcher_worker(
    name: str,
    matcher_core_fn,
    ref_proc: np.ndarray,
    tgt_proc: np.ndarray,
    ref_raw: np.ndarray,
    tgt_raw: np.ndarray,
    sensor_ref: str,
    sensor_tgt: str
) -> MatcherResult:
    """Worker function that runs a single matcher with the appropriate scale policy."""
    ref_shape = ref_raw.shape[:2]
    tgt_shape = tgt_raw.shape[:2]

    # Computationally controlled policy:
    # - SIFT may test multiple scale pairs
    # - Deep models (SuperPoint, LoFTR, LightGlue) run once at the primary scale pair
    if name == "SIFT":
        scale_pairs = get_sift_scale_pairs(sensor_ref, sensor_tgt)
    else:
        scale_pairs = [get_primary_scale_pair(sensor_ref, sensor_tgt)]

    best_pts1 = np.zeros((0, 2), dtype=np.float32)
    best_pts2 = np.zeros((0, 2), dtype=np.float32)
    best_geom = {"verified": False, "inliers": 0, "inlier_ratio": 0.0, "rmse": None, "spatial_coverage": 0.0, "H": None}
    best_score = -1e9

    for s_ref, s_tgt in scale_pairs:
        # Resize to candidate scale
        if s_ref != 1.0:
            h, w = ref_proc.shape[:2]
            r_ref = cv2.resize(ref_proc, (max(32, int(w * s_ref)), max(32, int(h * s_ref))))
        else:
            r_ref = ref_proc

        if s_tgt != 1.0:
            h, w = tgt_proc.shape[:2]
            r_tgt = cv2.resize(tgt_proc, (max(32, int(w * s_tgt)), max(32, int(h * s_tgt))))
        else:
            r_tgt = tgt_proc

        try:
            pts1_scaled, pts2_scaled = matcher_core_fn(r_ref, r_tgt)
        except Exception:
            continue

        if len(pts1_scaled) < 4:
            continue

        # Map keypoints back to native coordinates
        pts1 = (pts1_scaled / s_ref).astype(np.float32)
        pts2 = (pts2_scaled / s_tgt).astype(np.float32)

        # Geometric verification
        geom = verify_and_estimate_geometry(pts1, pts2, ref_shape, tgt_shape)

        if geom["verified"]:
            score = (geom["inlier_ratio"] * 100.0) + (geom["spatial_coverage"] * 80.0) - ((geom["rmse"] or 20.0) * 5.0) + (geom["inliers"] ** 0.5 * 5.0)
        else:
            score = float(geom["inliers"])

        if score > best_score:
            best_score = score
            best_pts1 = pts1
            best_pts2 = pts2
            best_geom = geom

    num_matches = len(best_pts1)
    verified = best_geom.get("verified", False)
    inliers = best_geom.get("inliers", 0)
    inlier_ratio = best_geom.get("inlier_ratio", 0.0)
    rmse = best_geom.get("rmse")
    coverage = best_geom.get("spatial_coverage", 0.0)
    H = best_geom.get("H")
    reason = best_geom.get("reason")

    confidence = compute_confidence(inliers, inlier_ratio, rmse, coverage, verified)
    status = "PASS" if verified and confidence in ("HIGH", "MEDIUM", "LOW") else "FAILED"

    aligned = None
    if H is not None:
        aligned = align_image(tgt_raw, H, ref_shape)

    return MatcherResult(
        algorithm=name,
        status=status,
        keypoints_reference=best_pts1,
        keypoints_target=best_pts2,
        correspondences=num_matches,
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        rmse=rmse,
        spatial_coverage=coverage,
        homography=H,
        confidence=confidence,
        registered_image=aligned,
        reason=reason if status == "FAILED" else None,
    )


def execute_matcher_with_timeout(
    name: str,
    matcher_core_fn,
    ref_proc: np.ndarray,
    tgt_proc: np.ndarray,
    ref_raw: np.ndarray,
    tgt_raw: np.ndarray,
    sensor_ref: str,
    sensor_tgt: str,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
) -> MatcherResult:
    """Execute a matcher with timing and timeout protection."""
    t0 = time.perf_counter()
    result_box = [None]
    exception_box = [None]

    def target():
        try:
            result_box[0] = run_matcher_worker(
                name, matcher_core_fn, ref_proc, tgt_proc, ref_raw, tgt_raw, sensor_ref, sensor_tgt
            )
        except Exception as e:
            exception_box[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_sec)
    elapsed = time.perf_counter() - t0

    if thread.is_alive():
        # Exceeded timeout
        return MatcherResult(
            algorithm=name,
            status="TIMEOUT",
            keypoints_reference=np.zeros((0, 2), dtype=np.float32),
            keypoints_target=np.zeros((0, 2), dtype=np.float32),
            correspondences=0,
            inliers=0,
            inlier_ratio=0.0,
            rmse=None,
            spatial_coverage=0.0,
            homography=None,
            confidence="FAILED",
            registered_image=None,
            elapsed_time=elapsed,
            reason=f"Exceeded timeout threshold of {timeout_sec:.1f}s",
        )

    if exception_box[0] is not None:
        err = exception_box[0]
        return MatcherResult(
            algorithm=name,
            status="FAILED",
            keypoints_reference=np.zeros((0, 2), dtype=np.float32),
            keypoints_target=np.zeros((0, 2), dtype=np.float32),
            correspondences=0,
            inliers=0,
            inlier_ratio=0.0,
            rmse=None,
            spatial_coverage=0.0,
            homography=None,
            confidence="FAILED",
            registered_image=None,
            elapsed_time=elapsed,
            reason=f"Runtime error: {type(err).__name__} ({str(err)})",
        )

    res = result_box[0]
    if res is not None:
        res.elapsed_time = elapsed
        return res

    return MatcherResult(
        algorithm=name,
        status="FAILED",
        keypoints_reference=np.zeros((0, 2), dtype=np.float32),
        keypoints_target=np.zeros((0, 2), dtype=np.float32),
        correspondences=0,
        inliers=0,
        inlier_ratio=0.0,
        rmse=None,
        spatial_coverage=0.0,
        homography=None,
        confidence="FAILED",
        registered_image=None,
        elapsed_time=elapsed,
        reason="Unknown execution failure",
    )


def run_all_four_matchers(
    ref_proc: np.ndarray,
    tgt_proc: np.ndarray,
    ref_raw: np.ndarray,
    tgt_raw: np.ndarray,
    sensor_ref: str,
    sensor_tgt: str,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    on_matcher_complete=None
) -> List[MatcherResult]:
    """
    Run all FOUR matching engines independently:
    1. SIFT
    2. SuperPoint
    3. LoFTR
    4. LightGlue
    """
    matchers = [
        ("SIFT", match_sift_core),
        ("SuperPoint", match_superpoint_core),
        ("LoFTR", match_loftr_core),
        ("LightGlue", match_lightglue_core),
    ]

    results = []
    for name, fn in matchers:
        res = execute_matcher_with_timeout(
            name, fn, ref_proc, tgt_proc, ref_raw, tgt_raw, sensor_ref, sensor_tgt, timeout_sec=timeout_sec
        )
        results.append(res)
        if on_matcher_complete:
            on_matcher_complete(res)

    return results
