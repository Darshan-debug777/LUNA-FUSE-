# LUNA-FUSE

Cross-Sensor Lunar Terrain Registration System for Chandrayaan-2 Imagery.

LUNA-FUSE discovers corresponding physical terrain features across multi-sensor lunar datasets (OHRC, TMC-2, IIRS), executes all four independent matching engines simultaneously, performs strict RANSAC geometric and spatial distribution verification, aligns the images, and generates publication-grade scientific reports.

## The Four Matching Engines

LUNA-FUSE runs all four algorithms as first-class engines behind a unified interface:
1. **SIFT**: Scale-Invariant Feature Transform with Lowe's ratio test (OpenCV).
2. **SuperPoint**: Self-supervised deep keypoint detection and 256-d descriptor extraction with Mutual Nearest Neighbor (MNN) verification.
3. **LoFTR**: Semi-dense detector-free local feature matching with Transformers (Kornia).
4. **LightGlue**: Adaptive deep graph neural network matcher with SuperPoint front-end.

## Installation

```bash
pip install -r requirements.txt
```

*(Note: LightGlue installs directly from `git+https://github.com/cvg/LightGlue.git`)*

## Running Real Chandrayaan-2 Data

The pipeline defaults to user-supplied real ISRO/PRADAN data. At least two sensor inputs are required:

### Pairwise Sensor Registration:
```bash
# OHRC and TMC-2
python main.py --ohrc path/to/ohrc.tif --tmc path/to/tmc.tif

# OHRC and IIRS
python main.py --ohrc path/to/ohrc.tif --iirs path/to/iirs.tif

# TMC-2 and IIRS
python main.py --tmc path/to/tmc.tif --iirs path/to/iirs.tif
```

### Multiple Image Ingestion:
```bash
python main.py \
    --ohrc ohrc1.tif ohrc2.tif \
    --tmc tmc1.tif \
    --iirs iirs1.tif iirs2.tif
```

### ISRO PRADAN Product Archive Ingestion:
```bash
python main.py --source isro --input path/to/ch2_product.zip path/to/reference.tif
```

Supported real formats: PNG, JPG, JPEG, TIFF, GeoTIFF, and unzipped or zipped ISRO PRADAN PDS4 product directories.

## Running Synthetic Regression Demo

For regression testing without real lunar products:
```bash
python main.py --demo-synthetic
```

## Architecture

- `main.py` — Orchestrates CLI, sensor pairing, multi-engine execution, and terminal report.
- `product_loader.py` — Ingests real Chandrayaan-2 products, parses PDS4 XML labels, and creates normalized `ImageRecord` objects.
- `preprocess.py` — Sensor-specific enhancement (percentile normalization, CLAHE, IIRS continuum extraction) and multi-scale pyramids.
- `matchers.py` — The four matching engines (SIFT, SuperPoint, LoFTR, LightGlue) with multi-scale resolution bridging.
- `registration.py` — RANSAC homography estimation, degenerate matrix rejection, sub-pixel refinement, RMSE, and spatial distribution coverage.
- `verification.py` — Confidence rating (HIGH, MEDIUM, LOW, FAILED) and correspondence-quality scoring.
- `visualization.py` — Publication-grade scientific dashboard renderer.
