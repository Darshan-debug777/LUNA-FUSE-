"""
api.py - FastAPI Web Service for LUNA-FUSE Lunar Image Registration & Fusion.

Provides production-ready REST API endpoints:
- GET  /health
- POST /api/v1/register
- GET  /api/v1/result/{job_id}

Compatible with Render Web Service deployment and local CLI/development.
"""

import os
import sys
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("luna_fuse_api")

# Application directories
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "output" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Valid lunar sensors supported by the scientific engine
VALID_SENSORS = {"OHRC", "TMC-2", "IIRS"}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".geotif", ".geotiff", ".zip", ".img"}

app = FastAPI(
    title="LUNA-FUSE API",
    description="Multi-Sensor Lunar Terrain Registration and Fusion Web Service (Chandrayaan-2 OHRC, TMC-2, IIRS)",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
allowed_origins_raw = os.environ.get("ALLOWED_ORIGINS", "*")
if allowed_origins_raw.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_sensor_name(sensor: str) -> str:
    """Normalize sensor name to standard format."""
    s = sensor.strip().upper()
    if s in ("TMC", "TMC2", "TMC-2"):
        return "TMC-2"
    if s == "OHRC":
        return "OHRC"
    if s == "IIRS":
        return "IIRS"
    return s


@app.api_route("/health", methods=["GET", "HEAD"], summary="Service Health Check")
def health():
    """
    Lightweight health check endpoint.
    Executes quickly without loading heavy neural network weights.
    """
    return {
        "status": "ok",
        "service": "luna-fuse",
        "version": "0.1.0"
    }


@app.get("/", summary="Root status and documentation link")
def root():
    return {
        "service": "LUNA-FUSE Multi-Sensor Lunar Terrain Registration",
        "version": "0.1.0",
        "health": "/health",
        "documentation": "/docs",
        "endpoints": {
            "register": "POST /api/v1/register",
            "result": "GET /api/v1/result/{job_id}"
        }
    }


@app.post("/api/v1/register", summary="Register and fuse two lunar images")
async def register_images(
    request: Request,
    source_image: UploadFile = File(..., description="Reference/source lunar image or PRADAN product (TIFF, PNG, JPG, ZIP)"),
    target_image: UploadFile = File(..., description="Target lunar image or PRADAN product to register against reference"),
    source_sensor: str = Form(..., description="Sensor for source image: OHRC, TMC-2, or IIRS"),
    target_sensor: str = Form(..., description="Sensor for target image: OHRC, TMC-2, or IIRS"),
):
    """
    Execute end-to-end multi-sensor registration using the real LUNA-FUSE engine.
    Runs all four matching engines: SIFT, SuperPoint, LoFTR, and LightGlue.
    Estimates verified RANSAC homography, computes RMSE and spatial coverage,
    and returns comprehensive metrics alongside a visual report URL.
    """
    # 1. Validate sensors
    norm_source_sensor = normalize_sensor_name(source_sensor)
    norm_target_sensor = normalize_sensor_name(target_sensor)

    if norm_source_sensor not in VALID_SENSORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid source_sensor '{source_sensor}'. Supported sensors: {', '.join(sorted(VALID_SENSORS))}."
        )
    if norm_target_sensor not in VALID_SENSORS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid target_sensor '{target_sensor}'. Supported sensors: {', '.join(sorted(VALID_SENSORS))}."
        )

    # 2. Validate file extensions
    src_ext = Path(source_image.filename or "").suffix.lower()
    tgt_ext = Path(target_image.filename or "").suffix.lower()

    if src_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format for source_image ('{source_image.filename}'). Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )
    if tgt_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format for target_image ('{target_image.filename}'). Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    # 3. Create unique job and temporary workspace
    job_id = str(uuid.uuid4())
    job_temp_dir = BASE_DIR / "output" / "temp" / job_id
    job_temp_dir.mkdir(parents=True, exist_ok=True)

    src_file_path = job_temp_dir / f"source_{Path(source_image.filename).name}"
    tgt_file_path = job_temp_dir / f"target_{Path(target_image.filename).name}"
    out_report_path = RESULTS_DIR / f"{job_id}.png"

    try:
        # Write uploaded files safely
        with open(src_file_path, "wb") as f_src:
            shutil.copyfileobj(source_image.file, f_src)
        with open(tgt_file_path, "wb") as f_tgt:
            shutil.copyfileobj(target_image.file, f_tgt)

        if src_file_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded source image '{source_image.filename}' is empty (0 bytes)."
            )
        if tgt_file_path.stat().st_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded target image '{target_image.filename}' is empty (0 bytes)."
            )

        # 4. Ingest products via real product loader
        from product_loader import load_product
        from main import run_sensor_pair_registration

        try:
            ref_record = load_product(str(src_file_path), sensor_hint=norm_source_sensor, is_synthetic=False)
            tgt_record = load_product(str(tgt_file_path), sensor_hint=norm_target_sensor, is_synthetic=False)
        except Exception as e:
            logger.warning(f"Error loading uploaded image data: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse or decode image data: {str(e)}"
            )

        # 5. Run the full 4-matcher LUNA-FUSE registration pipeline
        logger.info(f"Starting registration job {job_id}: {norm_source_sensor} <-> {norm_target_sensor}")
        results, best = run_sensor_pair_registration(ref_record, tgt_record, str(out_report_path))
        logger.info(f"Completed job {job_id}. Best matcher: {best.algorithm if best else 'None'}")

        # 6. Format individual matcher results
        formatted_results: Dict[str, Any] = {}
        for r in results:
            formatted_results[r.algorithm] = {
                "status": r.status,
                "correspondences": int(r.correspondences),
                "inliers": int(r.inliers),
                "inlier_ratio": round(float(r.inlier_ratio), 4),
                "rmse": round(float(r.rmse), 4) if r.rmse is not None else None,
                "spatial_coverage": round(float(r.spatial_coverage), 4),
                "confidence": r.confidence,
                "elapsed_time": round(float(r.elapsed_time), 2),
                "reason": r.reason
            }

        # 7. Format best registration summary
        if best is not None:
            best_algorithm = best.algorithm
            registration_summary = {
                "correspondences": int(best.correspondences),
                "inliers": int(best.inliers),
                "inlier_ratio": round(float(best.inlier_ratio), 4),
                "rmse": round(float(best.rmse), 4) if best.rmse is not None else None,
                "spatial_coverage": round(float(best.spatial_coverage), 4),
                "confidence": best.confidence
            }
        else:
            best_algorithm = "NONE"
            registration_summary = {
                "correspondences": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rmse": None,
                "spatial_coverage": 0.0,
                "confidence": "FAILED"
            }

        result_image_url = f"/api/v1/result/{job_id}"

        return {
            "job_id": job_id,
            "source_sensor": norm_source_sensor,
            "target_sensor": norm_target_sensor,
            "status": "completed",
            "best_matcher": best_algorithm,
            "results": formatted_results,
            "registration": registration_summary,
            "result_image_url": result_image_url
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected processing error during job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration processing failure: {str(e)}"
        )
    finally:
        # Cleanup uploaded temporary files
        try:
            if job_temp_dir.exists():
                shutil.rmtree(job_temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp dir {job_temp_dir}: {e}")


@app.api_route("/api/v1/result/{job_id}", methods=["GET", "HEAD"], summary="Retrieve registration report image")
def get_result_image(job_id: str):
    """
    Serve the generated scientific registration report image.
    Returns 404 if the job result image does not exist.
    """
    # Sanitize job_id to avoid directory traversal
    clean_id = Path(job_id).name
    image_path = RESULTS_DIR / f"{clean_id}.png"

    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Registration report for job '{job_id}' not found."
        )

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=f"luna_fuse_report_{clean_id}.png"
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting LUNA-FUSE server on 0.0.0.0:{port}")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
