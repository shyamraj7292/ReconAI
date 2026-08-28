"""
Generates three synthetic finance-ops sources (bank statement, internal ledger,
gateway settlement report) with known ground truth, so the matcher and the LLM
classifier can be scored against real numbers instead of eyeballed.

Run: python data/generate_synthetic_data.py
Outputs (gitignored, regenerate anytime): bank.csv, ledger.csv, gateway.csv,
ground_truth.json — all in this directory.
"""

import csv
import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_LEDGER_RECORDS = 200
OUT_DIR = Path(__file__).parent

random.seed(SEED)

MERCHANTS = [
    "Kavya Textiles", "BluePeak Electronics", "Namma Grocers", "Orbit Fitness",
    "Trailhead Books", "Cinnamon Cafe", "Vertex Studio", "Foothill Traders",
    "Sundar Motors", "Aster Wellness",
]

# scenario -> relative weight (must sum to something reasonable; normalized below)
SCENARIOS = {
    "exact": 55,          # clean match, all three sources agree
    "timing_lag": 12,     # settlement lands 1-4 days after ledger date
    "fee_deduction": 12,  # gateway amount = ledger amount - platform fee
    "duplicate": 6,       # bank shows the debit twice
    "partial_refund": 6,  # a linked refund record for part of the amount
    "rounding": 5,        # sub-rupee rounding drift
    "missing": 4,         # genuinely absent from one side - true exception
}


def weighted_scenario():
    names, weights = zip(*SCENARIOS.items())
    return random.choices(names, weights=weights, k=1)[0]


def random_date(start: date, end: date) -> date:
    span = (end - start).days
    return start + timedelta(days=random.randint(0, span))


def fmt(d: date) -> str:
    return d.isoformat()


def main():
    start = date(2026, 7, 1)
    end = date(2026, 8, 20)

    ledger_rows = []
    bank_rows = []
    gateway_rows = []
    ground_truth = {}

    bank_id_counter = 1
    gateway_id_counter = 1

    for i in range(1, N_LEDGER_RECORDS + 1):
        ledger_id = f"LDG{i:05d}"
        ref = f"PAY{100000 + i}"
        merchant = random.choice(MERCHANTS)
        ledger_date = random_date(start, end)
        amount = round(random.uniform(150, 45000), 2)
        scenario = weighted_scenario()

        ledger_rows.append({
            "ledger_id": ledger_id,
            "ref": ref,
            "date": fmt(ledger_date),
            "amount": amount,
            "merchant": merchant,
        })

        gt_entry = {
            "scenario": scenario,
            "bank_ids": [],
            "gateway_ids": [],
            "ledger_amount": amount,
        }

        if scenario == "exact":
            bank_id = f"BNK{bank_id_counter:05d}"; bank_id_counter += 1
            gw_id = f"GTW{gateway_id_counter:05d}"; gateway_id_counter += 1
            bank_rows.append({"bank_id": bank_id, "ref": ref, "date": fmt(ledger_date), "amount": amount})
            gateway_rows.append({"gateway_id": gw_id, "ref": ref, "date": fmt(ledger_date), "amount": amount})
            gt_entry["bank_ids"] = [bank_id]
            gt_entry["gateway_ids"] = [gw_id]

        elif scenario == "timing_lag":
            lag = random.randint(1, 4)
            settle_date = ledger_date + timedelta(days=lag)
            bank_id = f"BNK{bank_id_counter:05d}"; bank_id_counter += 1
            gw_id = f"GTW{gateway_id_counter:05d}"; gateway_id_counter += 1
            bank_rows.append({"bank_id": bank_id, "ref": ref, "date": fmt(settle_date), "amount": amount})
            gateway_rows.append({"gateway_id": gw_id, "ref": ref, "date": fmt(settle_date), "amount": amount})
            gt_entry["bank_ids"] = [bank_id]
            gt_entry["gateway_ids"] = [gw_id]

        elif scenario == "fee_deduction":
            fee_pct = random.uniform(0.015, 0.025)
            fee = round(amount * fee_pct, 2)
            net = round(amount - fee, 2)
            bank_id = f"BNK{bank_id_counter:05d}"; bank_id_counter += 1
            gw_id = f"GTW{gateway_id_counter:05d}"; gateway_id_counter += 1
            bank_rows.append({"bank_id": bank_id, "ref": ref, "date": fmt(ledger_date), "amount": net})
            gateway_rows.append({"gateway_id": gw_id, "ref": ref, "date": fmt(ledger_date), "amount": net})
            gt_entry["bank_ids"] = [bank_id]
            gt_entry["gateway_ids"] = [gw_id]
            gt_entry["fee"] = fee

        elif scenario == "duplicate":
            bank_id_1 = f"BNK{bank_id_counter:05d}"; bank_id_counter += 1
            bank_id_2 = f"BNK{bank_id_counter:05d}"; bank_id_counter += 1
            gw_id = f"GTW{gateway_id_counter:05d}"; gateway_id_counter += 1
            bank_rows.append({"bank_id": bank_id_1, "ref": ref, "date": fmt(ledger_date), "amount": amount})
            bank_rows.append({"bank_id": bank_id_2, "ref": ref, "date": fmt(ledger_date), "amount": amount})
            gateway_rows.append({"gateway_id": gw_id, "ref": ref, "date": fmt(ledger_date), "amount": amount})
            gt_entry["bank_ids"] = [bank_id_1, bank_id_2]
            gt_entry["gateway_ids"] = [gw_id]

        elif scenario == "partial_refund":
            refund_amount = round(amount * random.uniform(0.2, 0.5), 2)
            net = round(amount - refund_amount, 2)
            bank_id = f"BNK{bank_id_counter:05d}"; bank_id_counter += 1
            gw_id = f"GTW{gateway_id_counter:05d}"; gateway_id_counter += 1
            bank_rows.append({"bank_id": bank_id, "ref": ref, "date": fmt(ledger_date), "amount": net})
            gateway_rows.append({"gateway_id": gw_id, "ref": ref, "date": fmt(ledger_date), "amount": amount})
            gt_entry["bank_ids"] = [bank_id]
            gt_entry["gateway_ids"] = [gw_id]
            gt_entry["refund_amount"] = refund_amount

        elif scenario == "rounding":
            drift = round(random.uniform(-0.9, 0.9), 2)
            bank_amount = round(amount + drift, 2)
            bank_id = f"BNK{bank_id_counter:05d}"; bank_id_counter += 1
            gw_id = f"GTW{gateway_id_counter:05d}"; gateway_id_counter += 1
            bank_rows.append({"bank_id": bank_id, "ref": ref, "date": fmt(ledger_date), "amount": bank_amount})
            gateway_rows.append({"gateway_id": gw_id, "ref": ref, "date": fmt(ledger_date), "amount": amount})
            gt_entry["bank_ids"] = [bank_id]
            gt_entry["gateway_ids"] = [gw_id]

        elif scenario == "missing":
            # present in gateway (money was collected) but never showed up in the bank feed
            gw_id = f"GTW{gateway_id_counter:05d}"; gateway_id_counter += 1
            gateway_rows.append({"gateway_id": gw_id, "ref": ref, "date": fmt(ledger_date), "amount": amount})
            gt_entry["bank_ids"] = []
            gt_entry["gateway_ids"] = [gw_id]

        ground_truth[ledger_id] = gt_entry

    # shuffle so file order gives no matching hints
    random.shuffle(bank_rows)
    random.shuffle(gateway_rows)

    _write_csv(OUT_DIR / "ledger.csv", ledger_rows, ["ledger_id", "ref", "date", "amount", "merchant"])
    _write_csv(OUT_DIR / "bank.csv", bank_rows, ["bank_id", "ref", "date", "amount"])
    _write_csv(OUT_DIR / "gateway.csv", gateway_rows, ["gateway_id", "ref", "date", "amount"])

    with open(OUT_DIR / "ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"ledger: {len(ledger_rows)} rows")
    print(f"bank:   {len(bank_rows)} rows")
    print(f"gateway:{len(gateway_rows)} rows")
    scenario_counts = {}
    for v in ground_truth.values():
        scenario_counts[v["scenario"]] = scenario_counts.get(v["scenario"], 0) + 1
    print("scenario distribution:", scenario_counts)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
