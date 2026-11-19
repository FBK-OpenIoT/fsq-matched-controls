"""Dataset inventory: after a download, a JSON record of what is on disk (file count, size, source) in <data root>/_inventory/."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fsq_matched_controls import paths

log = logging.getLogger(__name__)


@dataclass
class DatasetInventory:
    name: str
    root: str
    downloaded_at: str
    total_files: int = 0
    total_size_bytes: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / 1024 ** 3


def scan_dataset_dir(name: str, path: Path, meta: dict | None = None) -> DatasetInventory:
    """Walk ``path`` and count its files and bytes."""
    n = size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                size += (Path(dirpath) / f).stat().st_size
                n += 1
            except OSError:
                pass
    return DatasetInventory(name=name, root=str(path), downloaded_at=datetime.now(timezone.utc).isoformat(),
                            total_files=n, total_size_bytes=size, meta=meta or {})


def write_inventory(inv: DatasetInventory) -> Path:
    d = paths.DATA_ROOT / "_inventory"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{inv.name}.json"
    dest.write_text(json.dumps(asdict(inv), indent=2), encoding="utf-8")
    log.info("Inventory written: %s (%d files, %.2f GB)", dest, inv.total_files, inv.total_size_gb)
    return dest
