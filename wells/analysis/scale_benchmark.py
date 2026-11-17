"""Repeat the NK loading benchmark on a deterministic synthetic 1M-row copy."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from importlib.metadata import version
from itertools import cycle, islice
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from wells.analysis.benchmark import CHUNK_ROWS, DEFAULT_RAW, benchmark_loading
from wells.data.ingest import CSV_COLUMNS, MANIFEST, REVISION, SOURCE_URL, download_inputs, read_inputs

DEFAULT_ROWS = 1_000_000
DEFAULT_OUT = Path("reports/scale_benchmark.json")


def write_synthetic_inputs(source: pd.DataFrame, directory: Path, rows: int) -> dict[str, int]:
    """Repeat rows in their original order within each of the two partitions."""
    if rows < 2:
        raise ValueError("the synthetic input requires at least two rows")
    partitions = {split: source.loc[source["split"].eq(split), list(CSV_COLUMNS)] for split in MANIFEST}
    train = len(partitions["train"])
    test = len(partitions["test"])
    if min(train, test) == 0:
        raise ValueError("both source partitions must contain rows")

    scaled_train = max(1, rows * train // (train + test))
    counts = {"train": scaled_train, "test": rows - scaled_train}
    directory.mkdir(parents=True, exist_ok=True)
    for split, frame in partitions.items():
        records = frame.itertuples(index=False, name=None)
        with (directory / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(CSV_COLUMNS)
            writer.writerows(islice(cycle(records), counts[split]))
    return counts


def _source_metadata() -> dict:
    return {
        "url": SOURCE_URL,
        "revision": REVISION,
        "partitions": {split: {"sha256": sha256, "rows": rows}
                       for split, (sha256, rows, *_bounds) in MANIFEST.items()},
    }


def run_scale_test(raw_dir: Path, rows: int) -> dict:
    """Measure the published files and the enlarged copy using the same loader."""
    if rows < 2:
        raise ValueError("the synthetic input requires at least two rows")
    download_inputs(raw_dir)
    with TemporaryDirectory(prefix="well-scale-") as temporary:
        work_dir = Path(temporary)
        synthetic_dir = work_dir / "synthetic"
        source, _ = read_inputs(raw_dir)
        counts = write_synthetic_inputs(source, synthetic_dir, rows)
        del source
        original = benchmark_loading(raw_dir, work_dir)
        scaled = benchmark_loading(synthetic_dir, work_dir)
    if any(result["rows"] != rows for result in scaled):
        raise RuntimeError("the loader did not process the requested synthetic row count")
    return {
        "source": _source_metadata(),
        "synthetic_input": {
            "rows": rows,
            "partition_rows": counts,
            "method": "Deterministically cycle rows in their original order within each partition. "
                      "Split counts approximate the published train/test ratio. "
                      "Duplicate measurements are intentional. This input exercises loading only.",
        },
        "measurement": {
            "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": version("pyarrow")},
            "chunk_rows": CHUNK_ROWS,
            "time": "Elapsed seconds for one run. Input generation and checksum verification are outside timing.",
            "memory": "Memory and file sizes use MiB. Peak allocations use tracemalloc. "
                      "DataFrame size uses deep memory accounting.",
            "scope": "Chunked aggregation reads well_id and parameter and counts rows per pair. "
                     "The other approaches load every column. Parquet write time is reported separately.",
        },
        "published_loading": original,
        "synthetic_loading": scaled,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        report = run_scale_test(args.raw_dir, args.rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"scale benchmark failed: {error}", file=sys.stderr)
        return 1
    for result in report["synthetic_loading"]:
        print(f"{result['approach']}: {result['rows']:,} rows in {result['seconds']:.3f} s, "
              f"{result['peak_mb']:.2f} MiB peak Python allocations")
    print(f"Report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
