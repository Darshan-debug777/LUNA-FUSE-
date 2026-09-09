"""
registration.py - Geometric verification, RANSAC estimation, sub-pixel refinement,
spatial distribution analysis, and alignment for LUNA-FUSE.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2


def is_valid_homography(H: np.ndarray, ref_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> bool:
    """
    Strictly check whether homography H represents a physically plausible transformation.
    Rejects singular, inverted, folded, or extreme-distortion transformations.
    """
    if H is None or H.shape != (3, 3):
        return False

    # Check for NaNs or Infs
    if not np.all(np.isfinite(H)):
        return False

    # Condition number check
    cond = np.linalg.cond(H)
    if cond > 1e6 or np.isnan(cond):
        return False

    det = np.linalg.det(H)
    if abs(det) < 1e-8:
        return False

    # Check mapping of the four target image corners
    th, tw = target_shape[:2]
    rh, rw = ref_shape[:2]
    corners = np.array([
        [0, 0],
        [tw, 0],
        [tw, th],
        [0, th]
    ], dtype=np.float32).reshape(-1, 1, 2)

    try:
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return False

    # Ensure projected polygon is convex and non-degenerate
    # Cross products of consecutive edge vectors must maintain consistent sign
    edges = [projected[(i + 1) % 4] - projected[i] for i in range(4)]
    cross_z = [edges[i][0] * edges[(i + 1) % 4][1] - edges[i][1] * edges[(i + 1) % 4][0] for i in range(4)]
    if not (all(c > 0 for c in cross_z) or all(c < 0 for c in cross_z)):
        return False  # Self-intersecting / flipped polygon

    # Area check: projected area must not be near zero or astronomical
    proj_area = 0.5 * abs(
        (projected[0, 0] * projected[1, 1] + projected[1, 0] * projected[2, 1] +
         projected[2, 0] * projected[3, 1] + projected[3, 0] * projected[0, 1]) -
        (projected[1, 0] * projected[0, 1] + projected[2, 0] * projected[1, 1] +
         projected[3, 0] * projected[2, 1] + projected[0, 0] * projected[3, 1])
    )
    ref_area = rw * rh
    if proj_area < 0.005 * ref_area or proj_area > 150.0 * ref_area:
        return False

    return True


def compute_spatial_coverage(pts: np.ndarray, img_shape: Tuple[int, int], grid_divs: int = 4) -> float:
    """
    Measure how well distributed inlier correspondences are across the lunar terrain.
    Divides the image into a grid (e.g., 4x4 = 16 cells) and calculates the percentage
    of cells containing valid correspondences, modulated by convex hull coverage.
    Returns value between 0.0 and 1.0.
    """
    if len(pts) < 3:
        return 0.0

    h, w = img_shape[:2]
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Grid occupancy
    cols = np.clip((xs / max(w, 1) * grid_divs).astype(int), 0, grid_divs - 1)
    rows = np.clip((ys / max(h, 1) * grid_divs).astype(int), 0, grid_divs - 1)
    occupied_cells = len(set(zip(cols, rows)))
    grid_score = occupied_cells / float(grid_divs * grid_divs)

    # Convex hull area ratio
    try:
        hull = cv2.convexHull(pts.astype(np.float32))
        hull_area = cv2.contourArea(hull)
        area_score = min(1.0, hull_area / (w * h * 0.6))
    except Exception:
        area_score = 0.0

    coverage = 0.6 * grid_score + 0.4 * area_score
    return float(np.clip(coverage, 0.0, 1.0))


def compute_reprojection_rmse(H: np.ndarray, pts_ref: np.ndarray, pts_target: np.ndarray, inlier_mask: np.ndarray) -> Optional[float]:
    """Calculate RMSE of reprojection on inlier correspondences."""
    if H is None or inlier_mask is None or inlier_mask.sum() == 0:
        return None

    target_in = pts_target[inlier_mask].reshape(-1, 1, 2).astype(np.float32)
    ref_in = pts_ref[inlier_mask].reshape(-1, 2).astype(np.float32)

    try:
        projected = cv2.perspectiveTransform(target_in, H).reshape(-1, 2)
        errors = np.linalg.norm(projected - ref_in, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        return rmse
    except Exception:
        return None


def refine_homography_subpixel(H_init: np.ndarray, pts_ref: np.ndarray, pts_target: np.ndarray, inlier_mask: np.ndarray) -> np.ndarray:
    """Sub-pixel iterative refinement of homography on inlier set."""
    if H_init is None or inlier_mask is None or inlier_mask.sum() < 6:
        return H_init

    ref_in = pts_ref[inlier_mask]
    tgt_in = pts_target[inlier_mask]

    try:
        # Re-estimate with tight RANSAC threshold for sub-pixel accuracy
        H_refined, _ = cv2.findHomography(tgt_in, ref_in, cv2.RANSAC, ransacReprojThreshold=2.0)
        if H_refined is not None:
            return H_refined
    except Exception:
        pass
    return H_init


def verify_and_estimate_geometry(
    pts_ref: np.ndarray,
    pts_target: np.ndarray,
    ref_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    ransac_thresh: float = 4.0
) -> Dict[str, Any]:
    """
    Run full geometric verification with RANSAC, stability checks,
    sub-pixel refinement, RMSE, and spatial coverage.
    """
    num_matches = len(pts_ref)
    empty_result = {
        "verified": False,
        "H": None,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "rmse": None,
        "spatial_coverage": 0.0,
        "inlier_mask": np.zeros(num_matches, dtype=bool),
        "reason": "Insufficient matches (< 4)",
    }

    if num_matches < 4:
        return empty_result

    # First RANSAC pass (target -> ref)
    H, mask = cv2.findHomography(pts_target, pts_ref, cv2.RANSAC, ransac_thresh)
    if H is None or mask is None:
        empty_result["reason"] = "RANSAC failed to estimate initial homography"
        return empty_result

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(inlier_mask.sum())
    inlier_ratio = float(inliers / num_matches) if num_matches > 0 else 0.0

    # Minimum inliers required for scientific lunar registration
    if inliers < 8:
        empty_result["inlier_mask"] = inlier_mask
        empty_result["inliers"] = inliers
        empty_result["inlier_ratio"] = inlier_ratio
        empty_result["reason"] = f"Too few inliers ({inliers} < 8)"
        return empty_result

    # Plausibility check
    if not is_valid_homography(H, ref_shape, target_shape):
        empty_result["inlier_mask"] = inlier_mask
        empty_result["inliers"] = inliers
        empty_result["inlier_ratio"] = inlier_ratio
        empty_result["reason"] = "Homography is degenerate, unstable, or geometrically inverted"
        return empty_result

    # Sub-pixel refinement
    H_ref = refine_homography_subpixel(H, pts_ref, pts_target, inlier_mask)
    if is_valid_homography(H_ref, ref_shape, target_shape):
        H = H_ref

    # Compute final verified RMSE and spatial coverage
    rmse = compute_reprojection_rmse(H, pts_ref, pts_target, inlier_mask)
    coverage = compute_spatial_coverage(pts_ref[inlier_mask], ref_shape)

    # Recheck criteria: reject if coverage is tiny (< 10%) or RMSE is unacceptable (> 15 px)
    if rmse is not None and rmse > 15.0:
        empty_result["inlier_mask"] = inlier_mask
        empty_result["inliers"] = inliers
        empty_result["inlier_ratio"] = inlier_ratio
        empty_result["rmse"] = rmse
        empty_result["spatial_coverage"] = coverage
        empty_result["reason"] = f"Reprojection RMSE too high ({rmse:.2f} px > 15 px)"
        return empty_result

    if coverage < 0.10:
        empty_result["inlier_mask"] = inlier_mask
        empty_result["inliers"] = inliers
        empty_result["inlier_ratio"] = inlier_ratio
        empty_result["rmse"] = rmse
        empty_result["spatial_coverage"] = coverage
        empty_result["reason"] = f"Correspondences clustered in tiny region (coverage {coverage*100:.1f}% < 10%)"
        return empty_result

    return {
        "verified": True,
        "H": H,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "spatial_coverage": coverage,
        "inlier_mask": inlier_mask,
        "reason": None,
    }


def align_image(target_gray: np.ndarray, H: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    """Warp target image into reference coordinate frame using homography H."""
    rh, rw = ref_shape[:2]
    if H is None:
        return np.zeros((rh, rw), dtype=np.uint8)
    warped = cv2.warpPerspective(target_gray, H, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def blend_overlay(ref_gray: np.ndarray, aligned_target: np.ndarray) -> np.ndarray:
    """
    Scientific fusion overlay:
    Reference in Red channel, aligned target in Green channel.
    Physical terrain overlap appears vibrant yellow/white where accurately aligned.
    """
    rh, rw = ref_gray.shape[:2]
    overlay = np.zeros((rh, rw, 3), dtype=np.uint8)
    overlay[:, :, 2] = ref_gray          # Reference -> Red channel
    overlay[:, :, 1] = aligned_target    # Aligned Target -> Green channel
    overlay[:, :, 0] = (0.2 * aligned_target).astype(np.uint8)  # Mild blue for tonal balance
    return overlay
