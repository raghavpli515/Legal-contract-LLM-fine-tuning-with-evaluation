"""Download CUAD v1 (data.zip -> CUADv1.json) into data/raw and verify its sha256.

    python -m legal_ft.data.download
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import requests

from legal_ft.config import REPO_ROOT, load_config

CUAD_JSON = "CUADv1.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
    tmp.replace(dest)


def fetch_cuad(force: bool = False) -> Path:
    """Return the path to CUADv1.json, downloading and extracting it if needed."""
    src = load_config("data")["source"]
    raw_dir = REPO_ROOT / src["raw_dir"]
    zip_path = raw_dir / "data.zip"
    json_path = raw_dir / CUAD_JSON

    if force or not zip_path.exists():
        print(f"Downloading {src['url']}")
        download(src["url"], zip_path)

    digest = sha256_of(zip_path)
    if src.get("sha256") and digest != src["sha256"]:
        raise ValueError(
            f"sha256 mismatch for {zip_path}: got {digest}, expected {src['sha256']}. "
            "Delete the file and re-download, or check the release changed."
        )
    if not src.get("sha256"):
        print(f"sha256 {digest} (pin this in configs/data.yaml)")

    if force or not json_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            member = next(n for n in zf.namelist() if n.endswith(CUAD_JSON))
            json_path.write_bytes(zf.read(member))
    return json_path


if __name__ == "__main__":
    print(fetch_cuad())
