"""Prefect flows for the dataset downloads (retries, logging and a run record in Prefect).

    python -m fsq_matched_controls.data.download.flows [cropglobe|eurocropsml|all]

Needs the optional dependency: pip install -e ".[download]". The same downloads run without Prefect with
python -m fsq_matched_controls.data.download (see there).
"""
from __future__ import annotations

import sys

from prefect import flow, task
from prefect.logging import get_run_logger

from fsq_matched_controls.data.download import cropglobe, eurocropsml
from fsq_matched_controls.data.download.inventory import DatasetInventory, write_inventory


@task(name="download-cropglobe", retries=3, retry_delay_seconds=60)
def task_cropglobe(countries: list[str] | None) -> DatasetInventory:
    get_run_logger().info("CropGlobe download (splits: %s)", countries or "all ten")
    return cropglobe.download(countries)


@task(name="download-eurocropsml", retries=3, retry_delay_seconds=60)
def task_eurocropsml(archives: list[str] | None) -> DatasetInventory:
    get_run_logger().info("EuroCropsML download (archives: %s)", archives or eurocropsml.ARCHIVES)
    return eurocropsml.download(archives)


@task(name="save-inventory")
def task_inventory(inv: DatasetInventory) -> str:
    path = write_inventory(inv)
    get_run_logger().info("Inventory saved: %s (%d files, %.2f GB)", path, inv.total_files, inv.total_size_gb)
    return str(path)


@flow(name="cropglobe-download", log_prints=True)
def cropglobe_flow(countries: list[str] | None = None) -> DatasetInventory:
    """CropGlobe median composites (Hugging Face x-ytong/CropGlobe, Median/) -> <data root>/cropglobe/raw/Median/."""
    inv = task_cropglobe(countries)
    task_inventory(inv)
    return inv


@flow(name="eurocropsml-download", log_prints=True)
def eurocropsml_flow(archives: list[str] | None = None) -> DatasetInventory:
    """EuroCropsML preprocess.zip and split.zip (Zenodo 13789558) -> <data root>/eurocropml/{preprocess,split}/."""
    inv = task_eurocropsml(archives)
    task_inventory(inv)
    return inv


@flow(name="datasets-download", log_prints=True)
def all_datasets_flow() -> None:
    """Both datasets."""
    cropglobe_flow()
    eurocropsml_flow()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"cropglobe": cropglobe_flow, "eurocropsml": eurocropsml_flow, "all": all_datasets_flow}[which]()
