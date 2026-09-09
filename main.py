"""
main.py - LUNA-FUSE Multi-Sensor Lunar Terrain Registration Pipeline

Processes real Chandrayaan-2 ISRO/PRADAN products (OHRC, TMC-2, IIRS) and synthetic regression demo.
Executes all four matching engines:
1. SIFT
2. SuperPoint
3. LoFTR
4. LightGlue
"""

import os
import sys
import argparse
from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch

from product_loader import load_product, ImageRecord
from preprocess import preprocess_record
from matchers import run_all_four_matchers, MatcherResult, DEVICE
from verification import select_best_matcher
from visualization import build_scientific_report


def print_banner():
    print("=" * 40)
    print("             LUNA-FUSE")
    print("========================================")
    print()


def run_sensor_pair_registration(
    ref_record: ImageRecord,
    tgt_record: ImageRecord,
    out_path: str
) -> Tuple[List[MatcherResult], Optional[MatcherResult]]:
    """Execute end-to-end registration on a reference and target ImageRecord."""
    source_label = ref_record.source
    sensor_ref = ref_record.sensor
    sensor_tgt = tgt_record.sensor

    print(f"SOURCE:\n{source_label}\n")
    print(f"INPUT:\n{sensor_ref} <-> {sensor_tgt}\n")
    print(f"DEVICE:\n{DEVICE.upper()}\n")

    print("MATCHING")
    print("-" * 40)

    # Preprocessing (scientific original preserved; matching image generated)
    ref_gray_raw, ref_proc = preprocess_record(ref_record)
    tgt_gray_raw, tgt_proc = preprocess_record(tgt_record)

    def on_matcher_done(res: MatcherResult):
        status_str = "OK" if res.status == "PASS" else res.status
        time_str = f"{res.elapsed_time:.2f}s"
        print(f"{res.algorithm:<12}{status_str:<9}{time_str}")

    # Execute all FOUR matchers with live status reporting
    results = run_all_four_matchers(
        ref_proc, tgt_proc, ref_gray_raw, tgt_gray_raw, sensor_ref, sensor_tgt,
        on_matcher_complete=on_matcher_done
    )

    best = select_best_matcher(results)

    print("\nREGISTRATION")
    print("-" * 40)

    if best is not None:
        rmse_str = f"{best.rmse:.2f} px" if best.rmse is not None else "N/A"
        print(f"Best Matcher:       {best.algorithm}")
        print(f"Correspondences:    {best.correspondences}")
        print(f"Inliers:            {best.inliers}")
        print(f"Inlier Ratio:       {best.inlier_ratio * 100:.1f}%")
        print(f"RMSE:               {rmse_str}")
        print(f"Spatial Coverage:   {best.spatial_coverage * 100:.1f}%")
        print(f"Confidence:         {best.confidence}")
        print()
        print("=" * 40)
        print("FINAL VERDICT")
        print("=" * 40)
        print()
        if best.confidence in ("HIGH", "MEDIUM"):
            print("SAME PHYSICAL LUNAR TERRAIN")
            print("REGISTRATION VERIFIED")
        else:
            print("REGISTRATION NOT TRUSTED")
            print("VERIFICATION INCONCLUSIVE")
    else:
        print("Best Matcher:       NONE")
        print("Confidence:         FAILED")
        print()
        print("=" * 40)
        print("FINAL VERDICT")
        print("=" * 40)
        print()
        print("REGISTRATION NOT TRUSTED")
        print("NO TRUSTED REGISTRATION FOUND")

    # Generate scientific visual report
    build_scientific_report(
        ref_gray_raw, tgt_gray_raw, results, best,
        source_label, sensor_ref, sensor_tgt, out_path
    )

    print()
    print("Report:")
    print(out_path)
    print("=" * 40 + "\n")

    return results, best


def main():
    parser = argparse.ArgumentParser(
        description="LUNA-FUSE: Cross-sensor lunar terrain registration using SIFT, SuperPoint, LoFTR, and LightGlue."
    )
    parser.add_argument("--ohrc", nargs="+", help="One or more paths to OHRC images or PRADAN products")
    parser.add_argument("--tmc", nargs="+", help="One or more paths to TMC-2 images or PRADAN products")
    parser.add_argument("--iirs", nargs="+", help="One or more paths to IIRS images or PRADAN products")
    parser.add_argument("--source", type=str, help="Specify source type (e.g. 'isro' or path to product archive)")
    parser.add_argument("--input", nargs="+", help="General input product paths or archives")
    parser.add_argument("--demo-synthetic", action="store_true", help="Explicitly run the synthetic regression test")
    parser.add_argument("--out", type=str, default="output/luna_fuse_result.png", help="Output path for the report image")

    args = parser.parse_args()
    print_banner()

    records: List[ImageRecord] = []

    # Handle explicit synthetic regression demo
    if args.demo_synthetic:
        ohrc_demo = "data/ohrc.png"
        iirs_demo = "data/iirs.png"
        if not os.path.exists(ohrc_demo) or not os.path.exists(iirs_demo):
            import generate_data
            generate_data.main()
        records.append(load_product(ohrc_demo, sensor_hint="OHRC", is_synthetic=True))
        records.append(load_product(iirs_demo, sensor_hint="IIRS", is_synthetic=True))

    else:
        # Load user-supplied real Chandrayaan-2 products
        if args.ohrc:
            for p in args.ohrc:
                records.append(load_product(p, sensor_hint="OHRC", is_synthetic=False))
        if args.tmc:
            for p in args.tmc:
                records.append(load_product(p, sensor_hint="TMC-2", is_synthetic=False))
        if args.iirs:
            for p in args.iirs:
                records.append(load_product(p, sensor_hint="IIRS", is_synthetic=False))
        if args.input:
            for p in args.input:
                records.append(load_product(p, is_synthetic=False))

    # Guard: Require real data by default
    if not records or len(records) < 2:
        print("ERROR: Real lunar data must be supplied. At least two sensor images are required.\n")
        print("Examples with real data:")
        print("  python main.py --ohrc path/to/ohrc.tif --tmc path/to/tmc.tif")
        print("  python main.py --ohrc path/to/ohrc.tif --iirs path/to/iirs.tif")
        print("  python main.py --ohrc ohrc1.tif ohrc2.tif --tmc tmc1.tif --iirs iirs.tif")
        print("  python main.py --source isro --input path/to/pradan_product.zip path/to/tmc.tif\n")
        print("Or run the synthetic regression test with:")
        print("  python main.py --demo-synthetic\n")
        sys.exit(1)

    # Multi-image support:
    # Designate the first record (or highest resolution sensor, e.g. OHRC) as reference
    # and register all subsequent images against it
    def sensor_priority(rec: ImageRecord):
        s = rec.sensor.upper()
        if "OHRC" in s:
            return 0
        if "TMC" in s:
            return 1
        if "IIRS" in s:
            return 2
        return 3

    records.sort(key=sensor_priority)
    ref_record = records[0]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for idx, target_record in enumerate(records[1:], start=1):
        if len(records) > 2:
            base, ext = os.path.splitext(args.out)
            out_file = f"{base}_pair_{idx}_{ref_record.sensor}_{target_record.sensor}{ext}"
        else:
            out_file = args.out

        run_sensor_pair_registration(ref_record, target_record, out_file)


if __name__ == "__main__":
    main()
