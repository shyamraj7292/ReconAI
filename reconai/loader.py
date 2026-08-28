"""CSV loading with typed amounts. Shared by pipeline, eval and tests."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(path: Path, amount_key="amount"):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row[amount_key] = float(row[amount_key])
            rows.append(row)
    return rows


def load_all(data_dir: Path = DATA_DIR):
    return (
        _load(data_dir / "ledger.csv"),
        _load(data_dir / "bank.csv"),
        _load(data_dir / "gateway.csv"),
    )
