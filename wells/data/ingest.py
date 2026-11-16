"""Acquisition and structural checks for the two published CSV partitions."""

import csv
import hashlib
import io
import re
import shutil
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import pandas as pd

SOURCE_URL = "https://huggingface.co/datasets/Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2"
REVISION = "4004bb78ca8811a87c2892a89e457c59d22ace2b"
CSV_COLUMNS = ("well_id", "well_type", "timestamp", "parameter", "value", "unit", "n_obs", "flag")
RAW_COLUMNS = ("source_file", "source_row", *CSV_COLUMNS, "split")
MANIFEST = {
    "train": (
        "8bf92fd97ac9b0d7ec2e7aa61be90f335c7f3b6ab2a84634fb8758b057efe26a",
        206_706, "2026-01-01 00:00:00", "2026-08-08 22:00:00",
    ),
    "test": (
        "8029113613ea2965cb94fc7bea50eee6f4c327b673ecc856d88d25dfb652c7d6",
        51_681, "2026-08-08 23:00:00", "2026-09-30 23:00:00",
    ),
}
HOUR_FORMAT = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:00:00")


def parse_hour(label: str) -> datetime | None:
    if isinstance(label, str) and HOUR_FORMAT.fullmatch(label):
        try:
            return datetime.fromisoformat(label)
        except ValueError:
            return None
    return None


def _check_hash(actual: str, expected: str, filename: str) -> None:
    if actual != expected:
        raise ValueError(f"{filename}: SHA-256 mismatch, expected {expected}, got {actual}")


def download_inputs(raw_dir: Path) -> None:
    """Verify cached files and publish downloads only after checking their hashes."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, (expected, *_) in MANIFEST.items():
        path = raw_dir / f"{split}.csv"
        if path.exists():
            with path.open("rb") as cached:
                _check_hash(hashlib.file_digest(cached, "sha256").hexdigest(), expected, path.name)
            continue

        url = f"{SOURCE_URL}/resolve/{REVISION}/{path.name}"
        output = NamedTemporaryFile(dir=raw_dir, prefix=f".{split}-", delete=False)
        temporary = Path(output.name)
        try:
            with output, urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, output)
            with temporary.open("rb") as downloaded:
                _check_hash(hashlib.file_digest(downloaded, "sha256").hexdigest(), expected, path.name)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def _read_csv(data: bytes, filename: str, split: str) -> pd.DataFrame:
    with io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            if tuple(next(reader, ())) != CSV_COLUMNS:
                raise ValueError(f"{filename}: unexpected CSV header")
            records = []
            for source_row, fields in enumerate(reader, start=1):
                if len(fields) != len(CSV_COLUMNS):
                    raise ValueError(f"{filename}: record {source_row} has {len(fields)} fields")
                records.append((filename, source_row, *fields, split))
        except (csv.Error, UnicodeError) as error:
            raise ValueError(f"{filename}: malformed CSV near physical line {reader.line_num}: {error}") from error
    return pd.DataFrame.from_records(records, columns=RAW_COLUMNS)


def read_inputs(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read verified bytes and preserve original text and one-based record IDs."""
    partitions = []
    meta = {"revision": REVISION, "source_url": SOURCE_URL}
    for split, (expected, count, start, end) in MANIFEST.items():
        path = raw_dir / f"{split}.csv"
        data = path.read_bytes()
        _check_hash(hashlib.sha256(data).hexdigest(), expected, path.name)
        frame = _read_csv(data, path.name, split)
        if len(frame) != count:
            raise ValueError(f"{path.name}: expected {count} records, found {len(frame)}")
        hours = [label for label in frame["timestamp"].unique() if parse_hour(label) is not None]
        if not hours or (min(hours), max(hours)) != (start, end):
            raise ValueError(f"{path.name}: expected inclusive hour bounds {start} to {end}")
        meta.update({f"{split}_sha256": expected, f"{split}_start": start, f"{split}_end": end})
        partitions.append(frame)
    return pd.concat(partitions, ignore_index=True), meta
