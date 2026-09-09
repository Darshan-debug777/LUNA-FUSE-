"""
visualization.py - Professional scientific visualization dashboard for LUNA-FUSE.

Renders high-resolution, publication-grade registration reports with:
- Mission header and data provenance
- Side-by-side pre-registration sensor views
- Four-algorithm status indicators
- Comprehensive metrics comparison table
- High-resolution registered/fused terrain overlay
- Quality assessment & final physical lunar terrain verdict
"""

import os
from typing import List, Optional
import numpy as np
import cv2
from matchers import MatcherResult
from registration import blend_overlay


BG_COLOR = (24, 17, 15)         # Deep slate/space background #0F1118
PANEL_COLOR = (42, 30, 26)      # Panel surface #1A1E2A
BORDER_COLOR = (70, 52, 45)     # Subdued border #2D3446
TEXT_PRIMARY = (245, 245, 245)  # Crisp white
TEXT_MUTED = (160, 160, 160)    # Muted silver
ACCENT_CYAN = (248, 189, 56)    # ISRO / Scientific Cyan #38BDF8
ACCENT_GREEN = (129, 199, 132)  # High Confidence Green
ACCENT_AMBER = (70, 180, 245)   # Medium Warning Amber
ACCENT_RED = (90, 90, 235)      # Inconclusive/Failed Red
CANVAS_W = 1280


def put_text(img: np.ndarray, text: str, org: tuple, scale: float = 0.6,
             color: tuple = TEXT_PRIMARY, thickness: int = 1,
             font: int = cv2.FONT_HERSHEY_SIMPLEX):
    """Draw anti-aliased text with shadow for crisp legibility."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (10, 10, 10), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_panel(h: int, w: int) -> np.ndarray:
    """Create a styled panel block."""
    panel = np.full((h, w, 3), PANEL_COLOR, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), BORDER_COLOR, 1)
    return panel


def render_header(source: str, sensor_ref: str, sensor_tgt: str, w: int) -> np.ndarray:
    """Header banner with mission branding and sensor configuration."""
    h = 130
    banner = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)

    put_text(banner, "LUNA-FUSE", (40, 48), 1.15, ACCENT_CYAN, 3)
    put_text(banner, "MULTI-SENSOR LUNAR REGISTRATION SYSTEM", (40, 78), 0.62, TEXT_MUTED, 1)

    # Provenance box on right
    box_w = 460
    box_x = w - box_w - 40
    cv2.rectangle(banner, (box_x, 22), (box_x + box_w, 108), PANEL_COLOR, -1)
    cv2.rectangle(banner, (box_x, 22), (box_x + box_w, 108), BORDER_COLOR, 1)

    put_text(banner, "SOURCE:", (box_x + 16, 52), 0.52, TEXT_MUTED, 1)
    put_text(banner, source, (box_x + 95, 52), 0.55, (255, 255, 255), 2)

    put_text(banner, "INPUT SENSORS:", (box_x + 16, 86), 0.52, TEXT_MUTED, 1)
    put_text(banner, f"{sensor_ref} <-> {sensor_tgt}", (box_x + 145, 86), 0.62, ACCENT_CYAN, 2)

    # Bottom separator
    cv2.line(banner, (40, h - 1), (w - 40, h - 1), BORDER_COLOR, 1)
    return banner


def render_pre_registration_view(ref_gray: np.ndarray, tgt_gray: np.ndarray,
                                 sensor_ref: str, sensor_tgt: str, w: int) -> np.ndarray:
    """Side-by-side view of input images prior to registration."""
    target_h = 320
    half_w = (w - 100) // 2

    def prepare_img(img, label):
        h, w_in = img.shape[:2]
        scaled = cv2.resize(img, (half_w, target_h), interpolation=cv2.INTER_AREA)
        if len(scaled.shape) == 2:
            scaled = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

        # Draw frame
        cv2.rectangle(scaled, (0, 0), (half_w - 1, target_h - 1), BORDER_COLOR, 2)
        # Label badge
        cv2.rectangle(scaled, (12, 12), (half_w - 12, 44), (20, 15, 12), -1)
        cv2.rectangle(scaled, (12, 12), (half_w - 12, 44), BORDER_COLOR, 1)
        put_text(scaled, label, (22, 34), 0.58, ACCENT_CYAN, 2)
        return scaled

    view1 = prepare_img(ref_gray, f"REFERENCE: {sensor_ref}")
    view2 = prepare_img(tgt_gray, f"TARGET: {sensor_tgt}")

    block_h = target_h + 70
    container = np.full((block_h, w, 3), BG_COLOR, dtype=np.uint8)
    put_text(container, "BEFORE REGISTRATION (UNALIGNED SCIENTIFIC INPUTS)", (40, 32), 0.62, TEXT_PRIMARY, 2)

    container[45:45 + target_h, 40:40 + half_w] = view1
    container[45:45 + target_h, 60 + half_w:60 + 2 * half_w] = view2
    return container


def render_four_algorithm_dashboard(results: List[MatcherResult], w: int) -> np.ndarray:
    """Four algorithm status cards and comparative metrics table."""
    h = 240
    container = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)
    cv2.line(container, (40, 1), (w - 40, 1), BORDER_COLOR, 1)

    put_text(container, "FOUR-ALGORITHM ANALYSIS & COMPARISON", (40, 34), 0.62, TEXT_PRIMARY, 2)

    # 1. Four status indicator boxes
    card_w = (w - 80 - 30) // 4
    for idx, res in enumerate(results):
        cx = 40 + idx * (card_w + 10)
        cy = 50
        passed = (res.status == "PASS")
        card_color = (35, 45, 30) if passed else (30, 25, 35)
        border_col = (80, 160, 80) if passed else (80, 70, 90)
        cv2.rectangle(container, (cx, cy), (cx + card_w, cy + 45), card_color, -1)
        cv2.rectangle(container, (cx, cy), (cx + card_w, cy + 45), border_col, 1)

        put_text(container, res.algorithm, (cx + 14, cy + 28), 0.55, TEXT_PRIMARY, 2)
        status_symbol = "[ PASS ]" if passed else "[ FAIL ]"
        status_col = ACCENT_GREEN if passed else ACCENT_RED
        put_text(container, status_symbol, (cx + card_w - 78, cy + 28), 0.48, status_col, 2)

    # 2. Comparison Table
    table_y = 115
    table_w = w - 80
    cv2.rectangle(container, (40, table_y), (40 + table_w, table_y + 110), PANEL_COLOR, -1)
    cv2.rectangle(container, (40, table_y), (40 + table_w, table_y + 110), BORDER_COLOR, 1)

    # Table Header
    headers = [("ALGORITHM", 55), ("STATUS", 230), ("MATCHES", 380), ("INLIERS", 510),
               ("INLIER %", 640), ("RMSE", 770), ("COVERAGE", 910), ("CONFIDENCE", 1060)]
    for title, tx in headers:
        put_text(container, title, (tx, table_y + 22), 0.46, TEXT_MUTED, 1)

    cv2.line(container, (40, table_y + 30), (40 + table_w, table_y + 30), BORDER_COLOR, 1)

    # Table Rows
    for r_idx, res in enumerate(results):
        ry = table_y + 48 + r_idx * 19
        put_text(container, res.algorithm, (55, ry), 0.48, (255, 255, 255), 2 if res.status == "PASS" else 1)
        status_col = ACCENT_GREEN if res.status == "PASS" else ACCENT_RED
        put_text(container, res.status, (230, ry), 0.48, status_col, 1)
        put_text(container, str(res.correspondences), (380, ry), 0.48, TEXT_PRIMARY, 1)
        put_text(container, str(res.inliers), (510, ry), 0.48, TEXT_PRIMARY, 1)
        put_text(container, f"{res.inlier_ratio * 100:.1f}%", (640, ry), 0.48, TEXT_PRIMARY, 1)
        rmse_str = f"{res.rmse:.2f} px" if res.rmse is not None else "--"
        put_text(container, rmse_str, (770, ry), 0.48, TEXT_PRIMARY, 1)
        put_text(container, f"{res.spatial_coverage * 100:.1f}%", (910, ry), 0.48, TEXT_PRIMARY, 1)

        conf_col = ACCENT_GREEN if res.confidence == "HIGH" else ACCENT_AMBER if res.confidence == "MEDIUM" else ACCENT_RED
        put_text(container, res.confidence, (1060, ry), 0.48, conf_col, 2)

    return container


def render_fusion_and_verdict(best_res: Optional[MatcherResult], ref_gray: np.ndarray,
                              tgt_gray: np.ndarray, sensor_ref: str, sensor_tgt: str, w: int) -> np.ndarray:
    """Render large fusion overlay, quality panel, and final verdict."""
    h = 580
    container = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)
    cv2.line(container, (40, 1), (w - 40, 1), BORDER_COLOR, 1)

    left_w = 760
    right_w = w - left_w - 100

    # Section Title
    put_text(container, "BEST MATCHER REGISTRATION & FUSION OVERLAY", (40, 36), 0.62, TEXT_PRIMARY, 2)

    # 1. Overlay Canvas (Left)
    overlay_h = 490
    overlay_w = left_w
    if best_res is not None and best_res.registered_image is not None and best_res.status == "PASS":
        overlay = blend_overlay(ref_gray, best_res.registered_image)
        scaled_overlay = cv2.resize(overlay, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
    else:
        scaled_overlay = np.full((overlay_h, overlay_w, 3), (25, 20, 20), dtype=np.uint8)
        put_text(scaled_overlay, "NO TRUSTED REGISTRATION FOUND", (overlay_w // 2 - 180, overlay_h // 2), 0.65, ACCENT_RED, 2)

    cv2.rectangle(scaled_overlay, (0, 0), (overlay_w - 1, overlay_h - 1), BORDER_COLOR, 2)
    # Legend
    cv2.rectangle(scaled_overlay, (14, 14), (overlay_w - 14, 44), (20, 15, 12), -1)
    cv2.rectangle(scaled_overlay, (14, 14), (overlay_w - 14, 44), BORDER_COLOR, 1)
    put_text(scaled_overlay, f"RED: {sensor_ref}  |  GREEN: {sensor_tgt}  |  YELLOW: PHYSICAL OVERLAP", (24, 34), 0.48, (220, 220, 220), 1)

    container[55:55 + overlay_h, 40:40 + overlay_w] = scaled_overlay

    # 2. Quality & Verdict Panel (Right)
    rx = 60 + overlay_w
    ry = 55
    right_panel = draw_panel(overlay_h, right_w)

    put_text(right_panel, "BEST MATCHER", (22, 36), 0.58, ACCENT_CYAN, 2)
    best_name = best_res.algorithm.upper() if best_res else "NONE"
    put_text(right_panel, best_name, (22, 68), 0.85, (255, 255, 255), 2)

    corr_val = str(best_res.inliers) if best_res else "0"
    put_text(right_panel, f"Verified inliers: {corr_val}", (22, 98), 0.52, TEXT_MUTED, 1)

    cv2.line(right_panel, (22, 118), (right_w - 22, 118), BORDER_COLOR, 1)

    put_text(right_panel, "REGISTRATION QUALITY", (22, 148), 0.56, ACCENT_CYAN, 2)
    metrics = [
        ("Reprojection RMSE:", f"{best_res.rmse:.2f} px" if best_res and best_res.rmse else "N/A"),
        ("Inlier Ratio:", f"{best_res.inlier_ratio*100:.1f}%" if best_res else "N/A"),
        ("Spatial Coverage:", f"{best_res.spatial_coverage*100:.1f}%" if best_res else "N/A"),
        ("Confidence Rating:", best_res.confidence if best_res else "FAILED"),
    ]
    my = 182
    for label, val in metrics:
        put_text(right_panel, label, (22, my), 0.50, TEXT_MUTED, 1)
        val_col = ACCENT_GREEN if "HIGH" in val or "px" in val else TEXT_PRIMARY
        put_text(right_panel, val, (22, my + 24), 0.60, val_col, 2)
        my += 52

    cv2.line(right_panel, (22, my), (right_w - 22, my), BORDER_COLOR, 1)
    my += 28

    # Final Verdict Card
    is_verified = (best_res is not None and best_res.confidence in ("HIGH", "MEDIUM") and best_res.status == "PASS")
    verdict_bg = (30, 48, 25) if is_verified else (40, 20, 25)
    verdict_border = ACCENT_GREEN if is_verified else ACCENT_RED
    cv2.rectangle(right_panel, (18, my), (right_w - 18, overlay_h - 20), verdict_bg, -1)
    cv2.rectangle(right_panel, (18, my), (right_w - 18, overlay_h - 20), verdict_border, 2)

    put_text(right_panel, "FINAL SCIENTIFIC VERDICT", (28, my + 28), 0.52, TEXT_MUTED, 1)
    if is_verified:
        put_text(right_panel, "SAME PHYSICAL LUNAR TERRAIN", (28, my + 58), 0.58, ACCENT_GREEN, 2)
        put_text(right_panel, "REGISTRATION VERIFIED", (28, my + 82), 0.52, (255, 255, 255), 1)
    else:
        put_text(right_panel, "REGISTRATION NOT TRUSTED", (28, my + 58), 0.58, ACCENT_RED, 2)
        put_text(right_panel, "VERIFICATION INCONCLUSIVE", (28, my + 82), 0.52, (255, 255, 255), 1)

    container[55:55 + overlay_h, rx:rx + right_w] = right_panel
    return container


def build_scientific_report(
    ref_gray: np.ndarray,
    tgt_gray: np.ndarray,
    results: List[MatcherResult],
    best_result: Optional[MatcherResult],
    source: str,
    sensor_ref: str,
    sensor_tgt: str,
    out_path: str
) -> str:
    """Generate complete publication-grade scientific dashboard."""
    w = CANVAS_W
    sec1 = render_header(source, sensor_ref, sensor_tgt, w)
    sec2 = render_pre_registration_view(ref_gray, tgt_gray, sensor_ref, sensor_tgt, w)
    sec3 = render_four_algorithm_dashboard(results, w)
    sec4 = render_fusion_and_verdict(best_result, ref_gray, tgt_gray, sensor_ref, sensor_tgt, w)

    report = np.vstack([sec1, sec2, sec3, sec4])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, report)
    return out_path
