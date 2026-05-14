from __future__ import annotations

import zipfile
from pathlib import Path


def export_run_zip(run_dir: Path) -> Path:
    zip_path = run_dir.parent / f"{run_dir.name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in run_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=f"{run_dir.name}/{path.relative_to(run_dir)}")

    return zip_path