import csv

import pandas as pd

from wells.analysis.scale_benchmark import write_synthetic_inputs
from wells.data.ingest import CSV_COLUMNS


def test_synthetic_inputs_preserve_values_and_order_within_each_split(tmp_path):
    records = [
        ("NK-1", "train", "1.00", ""),
        ("NK-1", "train", "", "weight_overflow"),
        ("NK-2", "test", "3.25", ""),
        ("NK-1", "train", "2.00", ""),
        ("NK-2", "test", "0", ""),
    ]
    source = pd.DataFrame(
        [(well, "fountain", "2026-01-01 00:00:00", "tubing_pressure", value, "atm", "1", flag, split)
         for well, split, value, flag in records],
        columns=[*CSV_COLUMNS, "split"],
    )

    assert write_synthetic_inputs(source, tmp_path, 12) == {"train": 7, "test": 5}

    first_bytes = {}
    for split, expected in {
        "train": ["1.00", "", "2.00", "1.00", "", "2.00", "1.00"],
        "test": ["3.25", "0", "3.25", "0", "3.25"],
    }.items():
        path = tmp_path / f"{split}.csv"
        first_bytes[path] = path.read_bytes()
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert tuple(reader.fieldnames) == CSV_COLUMNS
            rows = list(reader)
        assert [row["value"] for row in rows] == expected
        assert all(row["flag"] == "weight_overflow" for row in rows if row["value"] == "")

    write_synthetic_inputs(source, tmp_path, 12)
    assert all(path.read_bytes() == data for path, data in first_bytes.items())
