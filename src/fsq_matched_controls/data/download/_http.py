"""HTTP download helpers: streamed download with resume, retries and checksum check; Zenodo file listing; zip extraction."""
from __future__ import annotations

import hashlib
import logging
import os
import time
import zipfile
from pathlib import Path

import requests

log = logging.getLogger(__name__)

ZENODO_API = "https://zenodo.org/api/records/{record_id}"
CHUNK = 8 * 1024 * 1024
MAX_RETRIES = 5
BACKOFF = 2.0


def _headers() -> dict[str, str]:
    token = os.environ.get("ZENODO_TOKEN")          # optional; public records need none
    return {"Authorization": f"Bearer {token}"} if token else {}


def file_hash(path: Path, algorithm: str = "md5") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, *, expected: tuple[str, str] | None = None, resume: bool = True,
                  headers: dict[str, str] | None = None) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial file (HTTP Range) and retrying with exponential back-off.

    ``expected`` = (algorithm, hex digest), e.g. ("md5", "...") or ("sha256", "..."), is checked after the download.
    A file that is already complete and matches ``expected`` is not downloaded again.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected and dest.exists() and file_hash(dest, expected[0]) == expected[1].lower():
        log.info("Already present and verified: %s", dest)
        return dest
    attempt = 0
    while True:
        attempt += 1
        have = dest.stat().st_size if (resume and dest.exists()) else 0
        h = dict(headers or {})
        if have:
            h["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=h, stream=True, timeout=60) as r:
                if r.status_code == 416:            # range not satisfiable: already complete
                    break
                r.raise_for_status()
                mode = "ab" if have and r.status_code == 206 else "wb"
                total = int(r.headers.get("Content-Length", 0)) + (have if mode == "ab" else 0)
                done = have if mode == "ab" else 0
                next_report = done + 100 * 1024 * 1024
                with open(dest, mode) as fh:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        fh.write(chunk)
                        done += len(chunk)
                        if done >= next_report:
                            log.info("  %s: %.0f / %.0f MB", dest.name, done / 1e6, total / 1e6)
                            next_report += 100 * 1024 * 1024
            break
        except (requests.RequestException, OSError) as exc:
            if attempt >= MAX_RETRIES:
                raise RuntimeError(f"Failed to download {url} after {attempt} attempts") from exc
            wait = BACKOFF ** attempt
            log.warning("Download error (%s). Retry %d/%d in %.0f s", exc, attempt, MAX_RETRIES, wait)
            time.sleep(wait)
    if expected:
        got = file_hash(dest, expected[0])
        if got != expected[1].lower():
            raise ValueError(f"{expected[0]} mismatch for {dest.name}: expected {expected[1]}, got {got}")
        log.info("  %s verified (%s)", dest.name, expected[0])
    return dest


def zenodo_files(record_id: str | int) -> list[dict]:
    """File entries of a Zenodo record: ``key`` (file name), ``size`` (bytes), ``checksum`` ("md5:..."), ``links.self`` (URL)."""
    r = requests.get(ZENODO_API.format(record_id=record_id), headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json().get("files", [])


def extract_zip(archive: Path, dest_dir: Path) -> Path:
    """Extract ``archive`` into ``dest_dir``; refuses members that would land outside it."""
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting %s -> %s", Path(archive).name, dest_dir)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            target = (dest_dir / name).resolve()
            if dest_dir != target and dest_dir not in target.parents:
                raise ValueError(f"unsafe path in {archive}: {name}")
        zf.extractall(dest_dir)
    return dest_dir
