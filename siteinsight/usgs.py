from __future__ import annotations

from dataclasses import dataclass
import requests

from .utils import SiteInsightError

@dataclass(frozen=True)
class USGSRequest:
    bbox: str  # "lon_min,lat_min,lon_max,lat_max"
    size: int  # pixels (square)

USGS_3DEP_EXPORT = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"

# Official distribution endpoints for vendor JS (stable package CDN; no scraping arbitrary HTML)
THREE_MIN_JS_URL = "https://unpkg.com/three@0.160.0/build/three.min.js"
ORBIT_CONTROLS_URL = "https://unpkg.com/three@0.160.0/examples/js/controls/OrbitControls.js"

def fetch_usgs_dem_geotiff(req: USGSRequest, timeout_s: int = 60) -> bytes:
    params = {
        "f": "image",
        "bbox": req.bbox,
        "bboxSR": 4326,
        "imageSR": 4326,
        "size": f"{req.size},{req.size}",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation",
    }
    r = requests.get(USGS_3DEP_EXPORT, params=params, timeout=timeout_s)
    if r.status_code != 200:
        raise SiteInsightError(f"USGS DEM request failed (HTTP {r.status_code}). Try a smaller USA bbox.")
    ct = (r.headers.get("Content-Type", "") or "").lower()
    if "tiff" not in ct and "geotiff" not in ct and "application/octet-stream" not in ct:
        snippet = (r.text or "")[:200].replace("\n", " ")
        raise SiteInsightError(f"USGS returned non-TIFF response. Try a USA bbox. Details: {snippet}")
    if not r.content or len(r.content) < 1024:
        raise SiteInsightError("USGS returned an empty DEM response.")
    return r.content

def download_vendor_js(three_path, orbit_path, timeout_s: int = 60) -> None:
    def _dl(url: str, out_path: str):
        rr = requests.get(url, timeout=timeout_s)
        if rr.status_code != 200 or not rr.content:
            raise SiteInsightError(f"Failed to download vendor asset from {url} (HTTP {rr.status_code}).")
        with open(out_path, "wb") as f:
            f.write(rr.content)

    import os
    three_path = str(three_path)
    orbit_path = str(orbit_path)
    if not os.path.exists(three_path):
        _dl(THREE_MIN_JS_URL, three_path)
    if not os.path.exists(orbit_path):
        _dl(ORBIT_CONTROLS_URL, orbit_path)
