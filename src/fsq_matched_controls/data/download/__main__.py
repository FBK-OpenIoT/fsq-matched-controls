"""Dataset downloads without Prefect.

    python -m fsq_matched_controls.data.download [cropglobe|eurocropsml|all] [--countries ARG BEL ...] [--archives split.zip ...]
                                                 [--dry-run] [--delete-archives]

The datasets go to FSQ_DATA_ROOT (default data/): CropGlobe to cropglobe/raw/Median/, EuroCropsML to eurocropml/{preprocess,split}/.
--dry-run lists the files, sizes, checksums and destinations without writing anything.
"""
from __future__ import annotations

import argparse
import logging

from fsq_matched_controls import paths
from fsq_matched_controls.data.download import cropglobe, eurocropsml
from fsq_matched_controls.data.download.inventory import write_inventory


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fsq_matched_controls.data.download", description=__doc__.split("\n")[0])
    ap.add_argument("dataset", nargs="?", default="all", choices=["cropglobe", "eurocropsml", "all"])
    ap.add_argument("--countries", nargs="+", help=f"CropGlobe splits (default: all of {cropglobe.COUNTRIES})")
    ap.add_argument("--archives", nargs="+", help=f"EuroCropsML archives (default: {eurocropsml.ARCHIVES})")
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded; write nothing")
    ap.add_argument("--delete-archives", action="store_true", help="EuroCropsML: delete each archive after extraction")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("data root: %s", paths.DATA_ROOT)
    if a.dataset in ("cropglobe", "all"):
        inv = cropglobe.download(a.countries, dry_run=a.dry_run)
        if not a.dry_run:
            write_inventory(inv)
    if a.dataset in ("eurocropsml", "all"):
        inv = eurocropsml.download(a.archives, dry_run=a.dry_run, keep_archives=not a.delete_archives)
        if not a.dry_run:
            write_inventory(inv)


if __name__ == "__main__":
    main()
