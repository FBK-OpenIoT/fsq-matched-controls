"""EuroCropsML download: the preprocessed parcel files and the benchmark's split files.

Source: Zenodo record 13789558 (Reuss et al., 2025). Two of its archives are needed:
- ``preprocess.zip`` (1.45 GB): ``preprocess/<NUTS3>_<parcel id>_<HCAT code>.npz``, one file per parcel (706,683);
- ``split.zip`` (10 MB): ``split/latvia_portugal_vs_estonia/{pretrain,finetune,meta,counts}/...``, the Latvia + Portugal
  pretraining split, Estonia held out, and the official few-shot splits (``finetune/region_split_{k}.json``).
``raw_data.zip`` (3.26 GB) is not used. The archives go to ``<data root>/eurocropml/raw/`` and are extracted into
``<data root>/eurocropml/``, which gives exactly the folders ``fsq_matched_controls.data.eurocropml`` reads; no conversion
step. On first use, the data module consolidates the parcel files it needs into ``eurocropml/cache/`` (a one-time read).
Each archive's MD5 is checked against the one Zenodo reports.
"""
from __future__ import annotations

import logging

from fsq_matched_controls import paths
from fsq_matched_controls.data.download._http import download_file, extract_zip, zenodo_files, _headers
from fsq_matched_controls.data.download.inventory import DatasetInventory, scan_dataset_dir

log = logging.getLogger(__name__)

ZENODO_RECORD_ID = "13789558"
ARCHIVES = ["preprocess.zip", "split.zip"]


def download(archives: list[str] | None = None, *, dry_run: bool = False, keep_archives: bool = True) -> DatasetInventory | list[dict]:
    """Download and extract ``archives`` (default: preprocess.zip and split.zip).

    With ``dry_run`` nothing is written: the archives, sizes, checksums and destinations are returned and logged.
    ``keep_archives=False`` deletes each archive after it has been extracted.
    """
    archives = archives or ARCHIVES
    files = {f["key"]: f for f in zenodo_files(ZENODO_RECORD_ID)}
    missing = [a for a in archives if a not in files]
    if missing:
        raise RuntimeError(f"{missing} not in Zenodo record {ZENODO_RECORD_ID} (has {sorted(files)})")
    root = paths.DATA_ROOT / "eurocropml"
    plan = [dict(name=a, url=files[a]["links"]["self"], size=files[a]["size"], md5=files[a]["checksum"].removeprefix("md5:"),
                 dest=root / "raw" / a) for a in archives]
    log.info("EuroCropsML: %d archive(s), %.2f GB -> %s (extracted into %s)", len(plan), sum(p["size"] for p in plan) / 1e9,
             root / "raw", root)
    if dry_run:
        for p in plan:
            log.info("  %s  %.2f GB  md5 %s", p["name"], p["size"] / 1e9, p["md5"])
        return plan
    for p in plan:
        download_file(p["url"], p["dest"], expected=("md5", p["md5"]), headers=_headers())
        extract_zip(p["dest"], root)
        if not keep_archives:
            p["dest"].unlink()
    return scan_dataset_dir("eurocropml", root, meta={"source": f"https://zenodo.org/records/{ZENODO_RECORD_ID}", "archives": archives})
