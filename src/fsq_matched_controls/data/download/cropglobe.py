"""CropGlobe download: the Sentinel-2 median composites the letter uses.

Source: Hugging Face dataset x-ytong/CropGlobe (Tong and Wang, 2025), folder ``Median/``: one file per country split,
``median_S2_{CC}_month5-11_day5.npz`` (May to November, a five-day cadence). The files are written unchanged to
``<data root>/cropglobe/raw/Median/``, where ``fsq_matched_controls.data.cropglobe`` reads them; no conversion step.
Each file's SHA-256 is checked against the one the Hugging Face API reports.
"""
from __future__ import annotations

import logging

import requests

from fsq_matched_controls import paths
from fsq_matched_controls.data.download._http import download_file
from fsq_matched_controls.data.download.inventory import DatasetInventory, scan_dataset_dir

log = logging.getLogger(__name__)

HF_REPO = "x-ytong/CropGlobe"
FOLDER = "Median"
TREE_API = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main/{FOLDER}"
RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{FOLDER}/{{name}}"
# the country splits of the letter: the eight-split pool and the two evaluation regions (France; the Australian split, Queensland)
COUNTRIES = ["ARG", "AUS", "BEL", "CHN", "FRA", "GBR", "NLD", "USA", "USA_17", "USA_20"]


def file_name(cc: str) -> str:
    return f"median_S2_{cc}_month5-11_day5.npz"


def remote_files() -> dict[str, dict]:
    """name -> {size, sha256} of the files in the repository's Median folder."""
    r = requests.get(TREE_API, timeout=30)
    r.raise_for_status()
    return {e["path"].split("/")[-1]: {"size": e.get("size", 0), "sha256": (e.get("lfs") or {}).get("oid")}
            for e in r.json() if e.get("type") == "file"}


def download(countries: list[str] | None = None, *, dry_run: bool = False) -> DatasetInventory | list[dict]:
    """Download the median composites of ``countries`` (default: all ten splits of the letter).

    With ``dry_run`` nothing is written: the files, sizes and destinations are returned and logged.
    """
    countries = [c.upper() for c in countries] if countries else COUNTRIES
    unknown = set(countries) - set(COUNTRIES)
    if unknown:
        raise ValueError(f"unknown country split(s) {sorted(unknown)}; valid: {COUNTRIES}")
    remote = remote_files()
    dest_dir = paths.DATA_ROOT / "cropglobe" / "raw" / FOLDER
    plan = []
    for cc in countries:
        name = file_name(cc)
        if name not in remote:
            raise RuntimeError(f"{name} not found in {HF_REPO}/{FOLDER}")
        plan.append(dict(name=name, url=RESOLVE.format(name=name), dest=dest_dir / name, **remote[name]))
    total = sum(p["size"] for p in plan)
    log.info("CropGlobe: %d file(s), %.0f MB -> %s", len(plan), total / 1e6, dest_dir)
    if dry_run:
        for p in plan:
            log.info("  %s  %.1f MB  sha256 %s", p["name"], p["size"] / 1e6, p["sha256"])
        return plan
    for p in plan:
        download_file(p["url"], p["dest"], expected=("sha256", p["sha256"]) if p["sha256"] else None)
    return scan_dataset_dir("cropglobe", paths.DATA_ROOT / "cropglobe",
                            meta={"source": f"https://huggingface.co/datasets/{HF_REPO}", "folder": FOLDER, "countries": countries})
