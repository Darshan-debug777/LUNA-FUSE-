"""
product_loader.py - Real Chandrayaan-2 / ISRO data ingestion for LUNA-FUSE.

Supports:
- PNG, JPG, JPEG
- TIFF, GeoTIFF (single-band and multi-band scientific data)
- ISRO PRADAN product archives (ZIP files or unzipped folders with PDS4 XML labels)
- Metadata extraction (sensor, acquisition time, coordinates, resolution, data types)
- Non-destructive: never alters original source files.
"""

import os
import zipfile
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import cv2

try:
    import tifffile
except ImportError:
    tifffile = None


@dataclass
class ImageRecord:
    """Normalized internal representation of a lunar image."""
    sensor: str  # 'OHRC', 'TMC-2', 'IIRS', or 'UNKNOWN'
    image: np.ndarray  # Raw scientific image array (2D or 3D)
    original_path: str
    original_format: str
    acquisition_time: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    resolution: Optional[float] = None  # meters / pixel
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "REAL CHANDRAYAAN-2 DATA"  # or "SYNTHETIC DEMO"


def detect_sensor_from_text(text: str) -> str:
    """Detect sensor name from filename, path, or XML text."""
    t = text.lower()
    if "ohrc" in t or "ohr" in t or "orbiter high resolution" in t:
        return "OHRC"
    if "tmc-2" in t or "tmc2" in t or "tmc" in t or "terrain mapping camera" in t:
        return "TMC-2"
    if "iirs" in t or "iir" in t or "imaging infrared" in t or "hyperspectral" in t:
        return "IIRS"
    return "UNKNOWN"


def parse_pds4_xml(xml_content: str) -> Dict[str, Any]:
    """Extract metadata from ISRO PRADAN PDS4 XML label."""
    meta: Dict[str, Any] = {}
    try:
        # Strip XML namespaces for easier querying
        root = ET.fromstring(xml_content)
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        # Extract logical identifier & title
        lid = root.find(".//logical_identifier")
        if lid is not None and lid.text:
            meta["logical_identifier"] = lid.text.strip()
            if "sensor" not in meta:
                detected = detect_sensor_from_text(lid.text)
                if detected != "UNKNOWN":
                    meta["sensor"] = detected

        title = root.find(".//title")
        if title is not None and title.text:
            meta["title"] = title.text.strip()
            if "sensor" not in meta:
                detected = detect_sensor_from_text(title.text)
                if detected != "UNKNOWN":
                    meta["sensor"] = detected

        # Instrument / Observing system
        for comp in root.findall(".//Observing_System_Component/name"):
            if comp.text:
                meta.setdefault("observing_system", []).append(comp.text.strip())
                detected = detect_sensor_from_text(comp.text)
                if detected != "UNKNOWN":
                    meta["sensor"] = detected

        # Time
        start_time = root.find(".//start_date_time")
        if start_time is not None and start_time.text:
            meta["start_date_time"] = start_time.text.strip()
            meta["acquisition_time"] = start_time.text.strip()

        stop_time = root.find(".//stop_date_time")
        if stop_time is not None and stop_time.text:
            meta["stop_date_time"] = stop_time.text.strip()

        # Coordinates / Bounding box
        lat_elem = root.find(".//latitude") or root.find(".//center_latitude")
        if lat_elem is not None and lat_elem.text:
            try:
                meta["latitude"] = float(lat_elem.text)
            except ValueError:
                pass

        lon_elem = root.find(".//longitude") or root.find(".//center_longitude")
        if lon_elem is not None and lon_elem.text:
            try:
                meta["longitude"] = float(lon_elem.text)
            except ValueError:
                pass

        # Resolution / scale
        res_elem = root.find(".//pixel_resolution") or root.find(".//spatial_resolution")
        if res_elem is not None and res_elem.text:
            try:
                meta["resolution"] = float(res_elem.text)
            except ValueError:
                pass

        # Target
        target = root.find(".//Target_Identification/name")
        if target is not None and target.text:
            meta["target"] = target.text.strip()

        # Referenced file names
        data_files = []
        for file_elem in root.findall(".//File/file_name"):
            if file_elem.text:
                data_files.append(file_elem.text.strip())
        if data_files:
            meta["referenced_files"] = data_files

    except Exception as e:
        meta["xml_parse_error"] = str(e)

    return meta


def load_array_from_file(path: str) -> Tuple[np.ndarray, str]:
    """Load image data array from file using tifffile or cv2."""
    ext = os.path.splitext(path)[1].lower()

    if ext in [".tif", ".tiff", ".geotif", ".geotiff"]:
        if tifffile is not None:
            try:
                arr = tifffile.imread(path)
                return arr, "GeoTIFF" if "geo" in ext or arr.ndim >= 2 else "TIFF"
            except Exception:
                pass
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is not None:
            return arr, "TIFF"
        raise ValueError(f"Failed to read TIFF file: {path}")

    if ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is not None:
            fmt = "PNG" if ext == ".png" else "JPEG"
            return arr, fmt
        raise ValueError(f"Failed to read image file: {path}")

    # Fallback to OpenCV IMREAD_UNCHANGED
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is not None:
        return arr, ext.upper().lstrip(".")

    raise ValueError(f"Unsupported or unreadable file format: {path}")


def load_pradan_directory(dir_path: str, sensor_hint: Optional[str] = None) -> ImageRecord:
    """Load product from unzipped ISRO PRADAN product folder."""
    xml_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(".xml")]
    meta: Dict[str, Any] = {}
    if xml_files:
        with open(xml_files[0], "r", encoding="utf-8", errors="replace") as f:
            meta = parse_pds4_xml(f.read())

    sensor = sensor_hint or meta.get("sensor") or detect_sensor_from_text(dir_path)
    if sensor == "UNKNOWN" and xml_files:
        sensor = detect_sensor_from_text(os.path.basename(xml_files[0]))

    # Locate image file: prefer TIFF/GeoTIFF over browse PNG
    img_candidates = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".tif", ".tiff", ".geotif", ".geotiff", ".png", ".jpg", ".jpeg", ".img"]:
                img_candidates.append(os.path.join(root, f))

    if not img_candidates:
        raise FileNotFoundError(f"No image data found in product directory: {dir_path}")

    # Sort so TIFF / GeoTIFF is preferred, then PNG/JPG
    def sort_key(p):
        e = os.path.splitext(p)[1].lower()
        if "browse" in p.lower() or "thumb" in p.lower():
            return 2
        if e in [".tif", ".tiff", ".geotif", ".geotiff"]:
            return 0
        return 1

    img_candidates.sort(key=sort_key)
    chosen_path = img_candidates[0]
    arr, fmt = load_array_from_file(chosen_path)

    return ImageRecord(
        sensor=sensor,
        image=arr,
        original_path=dir_path,
        original_format=f"PRADAN_PDS4 ({fmt})",
        acquisition_time=meta.get("acquisition_time"),
        latitude=meta.get("latitude"),
        longitude=meta.get("longitude"),
        resolution=meta.get("resolution"),
        metadata=meta,
        source="REAL CHANDRAYAAN-2 DATA",
    )


def load_pradan_zip(zip_path: str, sensor_hint: Optional[str] = None) -> ImageRecord:
    """Inspect and extract data from ISRO PRADAN ZIP package safely."""
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        xml_names = [n for n in names if n.lower().endswith(".xml")]
        meta: Dict[str, Any] = {}
        if xml_names:
            xml_data = z.read(xml_names[0]).decode("utf-8", errors="replace")
            meta = parse_pds4_xml(xml_data)

        sensor = sensor_hint or meta.get("sensor") or detect_sensor_from_text(zip_path)
        if sensor == "UNKNOWN" and xml_names:
            sensor = detect_sensor_from_text(xml_names[0])

        # Find target image in zip
        def zip_sort_key(p):
            e = os.path.splitext(p)[1].lower()
            if "browse" in p.lower() or "thumb" in p.lower():
                return 2
            if e in [".tif", ".tiff", ".geotif", ".geotiff"]:
                return 0
            if e in [".png", ".jpg", ".jpeg"]:
                return 1
            return 3

        img_names = [n for n in names if os.path.splitext(n)[1].lower() in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]]
        if not img_names:
            raise FileNotFoundError(f"No valid image file found inside PRADAN ZIP: {zip_path}")

        img_names.sort(key=zip_sort_key)
        target_name = img_names[0]

        # Extract target image to temporary file
        temp_dir = tempfile.mkdtemp(prefix="luna_fuse_")
        extracted_path = z.extract(target_name, temp_dir)
        arr, fmt = load_array_from_file(extracted_path)

        return ImageRecord(
            sensor=sensor,
            image=arr,
            original_path=zip_path,
            original_format=f"PRADAN_ZIP ({fmt})",
            acquisition_time=meta.get("acquisition_time"),
            latitude=meta.get("latitude"),
            longitude=meta.get("longitude"),
            resolution=meta.get("resolution"),
            metadata=meta,
            source="REAL CHANDRAYAAN-2 DATA",
        )


def load_product(path: str, sensor_hint: Optional[str] = None, is_synthetic: bool = False) -> ImageRecord:
    """
    Main loader entry point.
    Loads real ISRO/PRADAN products (ZIP, folder, GeoTIFF, TIFF, PNG, JPG)
    or synthetic regression demo files.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Product path does not exist: {path}")

    source_label = "SYNTHETIC DEMO" if is_synthetic else "REAL CHANDRAYAAN-2 DATA"

    # 1. Directory product
    if os.path.isdir(path):
        record = load_pradan_directory(path, sensor_hint=sensor_hint)
        record.source = source_label
        return record

    # 2. ZIP archive
    if zipfile.is_zipfile(path):
        record = load_pradan_zip(path, sensor_hint=sensor_hint)
        record.source = source_label
        return record

    # 3. Direct image file (TIFF, GeoTIFF, PNG, JPEG)
    arr, fmt = load_array_from_file(path)
    sensor = sensor_hint or detect_sensor_from_text(path)
    if sensor == "UNKNOWN":
        sensor = "OHRC"  # default fallback if sensor unspecified

    meta: Dict[str, Any] = {
        "dimensions": arr.shape,
        "dtype": str(arr.dtype),
    }

    # Check for accompanying XML label with same basename
    base_no_ext = os.path.splitext(path)[0]
    sidecar_xml = base_no_ext + ".xml"
    if os.path.exists(sidecar_xml):
        with open(sidecar_xml, "r", encoding="utf-8", errors="replace") as f:
            meta.update(parse_pds4_xml(f.read()))
            if sensor_hint is None and "sensor" in meta:
                sensor = meta["sensor"]

    return ImageRecord(
        sensor=sensor,
        image=arr,
        original_path=path,
        original_format=fmt,
        acquisition_time=meta.get("acquisition_time"),
        latitude=meta.get("latitude"),
        longitude=meta.get("longitude"),
        resolution=meta.get("resolution"),
        metadata=meta,
        source=source_label,
    )
