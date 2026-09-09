"""
verification.py - Confidence scoring, algorithm ranking, and selection of the best trusted registration.
"""

from typing import Optional, List, Dict, Any


def compute_confidence(inliers: int, inlier_ratio: float, rmse: Optional[float], spatial_coverage: float, verified: bool) -> str:
    """
    Compute scientific confidence based on geometric verification, inliers,
    inlier ratio, reprojection RMSE, and spatial distribution.
    """
    if not verified or rmse is None or inliers < 8:
        return "FAILED"

    # Strict scientific criteria for lunar terrain registration
    if inliers >= 20 and inlier_ratio >= 0.45 and rmse <= 4.0 and spatial_coverage >= 0.28:
        return "HIGH"

    if inliers >= 10 and inlier_ratio >= 0.20 and rmse <= 9.0 and spatial_coverage >= 0.14:
        return "MEDIUM"

    if inliers >= 8 and rmse <= 14.0:
        return "LOW"

    return "FAILED"


def score_result(res: Any) -> float:
    """
    Scoring system based on correspondence QUALITY, not raw match count.
    A result with 500 matches but terrible geometry must lose to a result
    with 80 strong geometrically consistent matches.
    """
    if res is None or getattr(res, "status", None) != "PASS" or getattr(res, "rmse", None) is None:
        return -1e9

    inliers = getattr(res, "inliers", 0)
    inlier_ratio = getattr(res, "inlier_ratio", 0.0)
    rmse = getattr(res, "rmse", 99.0)
    coverage = getattr(res, "spatial_coverage", 0.0)
    confidence = getattr(res, "confidence", "FAILED")

    if inliers < 8:
        return -1e9

    # Bonus for confidence level
    conf_bonus = {"HIGH": 50.0, "MEDIUM": 20.0, "LOW": 0.0}.get(confidence, -50.0)

    # Score calculation:
    # - Inlier ratio heavily rewarded (0 to 100)
    # - Spatial coverage heavily rewarded (0 to 100)
    # - RMSE heavily penalized (5x multiplier)
    # - Inliers count rewarded with square root (diminishing return to avoid raw count bias)
    score = (
        (inlier_ratio * 100.0) +
        (coverage * 80.0) +
        ((inliers ** 0.5) * 6.0) -
        (rmse * 7.0) +
        conf_bonus
    )
    return score


def select_best_matcher(results: List[Any]) -> Optional[Any]:
    """Select the strongest TRUSTED result among all algorithms."""
    valid = [
        r for r in results
        if r is not None and getattr(r, "status", None) == "PASS" and getattr(r, "rmse", None) is not None
    ]
    if not valid:
        return None

    best = max(valid, key=score_result)
    # Ensure best meets minimum threshold
    if best.inliers >= 8 and best.rmse is not None and best.rmse <= 15.0:
        return best
    return None
