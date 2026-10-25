"""Audit the frozen feature builder's whole-file price-history gate.

Author: Gilmour Jia

Reads local input caches and writes aggregate evidence plus hashes, never raw
price or return observations. It does not import or modify the feature builder.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER = Path("research/ML金融学习计划/D_A股数据/build_features.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_short_file(path: Path, minimum_rows: int) -> dict | None:
    """Count all CSV records; retain dates only for potentially short files."""
    row_count = 0
    short_dates = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "date" not in (reader.fieldnames or []):
            raise ValueError(f"Missing date column: {path.name}")
        for row in reader:
            row_count += 1
            if row_count < minimum_rows:
                short_dates.append(row["date"])
    if row_count >= minimum_rows:
        return None
    dates = [date.fromisoformat(value[:10]) for value in short_dates]
    monthly_counts = Counter(value.strftime("%Y-%m") for value in dates)
    months = sorted(monthly_counts)
    span = 0
    if months:
        first = min(dates).replace(day=1)
        last = max(dates).replace(day=1)
        span = (last.year - first.year) * 12 + last.month - first.month + 1
    return {
        "code": path.stem,
        "source_file": f"prices/{path.name}",
        "sha256": sha256(path),
        "daily_records": row_count,
        "observed_months": len(months),
        "calendar_span_months": span,
        "first_month": months[0] if months else None,
        "last_month": months[-1] if months else None,
        "monthly_counts": monthly_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Local directory containing prices/ and returns_monthly.csv")
    parser.add_argument("--output", type=Path,
                        default=REPO / "results/repaired/entry_gate_audit.json")
    parser.add_argument("--min-months", type=int, default=24)
    parser.add_argument("--min-days", type=int, default=10)
    parser.add_argument("--minimum-file-rows", type=int, default=60)
    args = parser.parse_args()
    if min(args.min_months, args.min_days, args.minimum_file_rows) < 1:
        parser.error("All thresholds must be positive")

    returns_file = args.raw_dir / "returns_monthly.csv"
    with returns_file.open(encoding="utf-8-sig", newline="") as stream:
        have_returns = {row["code"] for row in csv.DictReader(stream)}
    files = sorted((args.raw_dir / "prices").glob("*.csv"))
    if not files:
        raise ValueError("No price CSV files found")
    unmatched = []
    excluded = []
    for path in files:
        if path.stem not in have_returns:
            unmatched.append(path.stem)
            continue
        item = inspect_short_file(path, args.minimum_file_rows)
        if item is None:
            continue
        counts = item.pop("monthly_counts")
        eligible = [month for history_months, month in enumerate(sorted(counts), 1)
                    if history_months >= args.min_months and counts[month] >= args.min_days]
        item["eligible_stock_months"] = len(eligible)
        item["eligible_months"] = eligible
        excluded.append(item)

    def extrema(key: str) -> list | None:
        values = [item[key] for item in excluded if item[key] is not None]
        return [min(values), max(values)] if values else None

    evidence = {
        "schema": "gkx-entry-gate-audit/1",
        "author": "Gilmour Jia",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen price19 build entry gate; read-only input inspection",
        "parameters": {
            "whole_file_minimum_daily_records": args.minimum_file_rows,
            "minimum_observed_history_months": args.min_months,
            "minimum_days_in_feature_month": args.min_days,
            "file_limit": 0,
        },
        "method": [
            "Enumerate every prices/*.csv, matching the builder's unlimited sorted file list.",
            "First require the filename stem to occur in raw returns_monthly.csv code values.",
            "Count CSV data records, matching the builder's whole-file len(px) gate.",
            "For files below the gate, group valid daily dates by calendar month; count observed history months cumulatively.",
            "Count each stock-month with cumulative observed history >= min-months and current-month daily records >= min-days.",
            "Publish only aggregate counts, short-file coverage metadata, and SHA-256 hashes; do not redistribute raw observations or local absolute paths.",
        ],
        "summary": {
            "price_files_total": len(files),
            "raw_monthly_return_codes": len(have_returns),
            "price_files_matching_raw_returns": len(files) - len(unmatched),
            "price_files_without_raw_returns": len(unmatched),
            "files_excluded_by_daily_record_gate": len(excluded),
            "daily_records_range": extrema("daily_records"),
            "observed_months_range": extrema("observed_months"),
            "calendar_span_months_range": extrema("calendar_span_months"),
            "first_month_range": extrema("first_month"),
            "last_month_range": extrema("last_month"),
            "eligible_stocks_excluded": sum(item["eligible_stock_months"] > 0 for item in excluded),
            "eligible_stock_months_excluded": sum(item["eligible_stock_months"] for item in excluded),
        },
        "unmatched_price_codes": unmatched,
        "excluded_files": excluded,
        "source_binding": {
            "builder_file": BUILDER.as_posix(),
            "builder_sha256": sha256(REPO / BUILDER),
            "audit_script_file": "scripts/audit_entry_gate.py",
            "audit_script_sha256": sha256(Path(__file__)),
            "raw_returns_file": "returns_monthly.csv",
            "raw_returns_sha256": sha256(returns_file),
        },
        "interpretation": {
            "direct_eligibility_effect": "The excluded files do not directly remove any eligible stock-month when eligible_stock_months_excluded is zero.",
            "numerical_invariance_not_established": "Feature winsorization precedes eligibility filtering. Removing the gate may change the cross-sectional winsorization bounds for months covered by excluded files, even when none is eligible.",
            "release_action": "Preserve the frozen builder and its recorded results; remove the residual whole-file gate in a future rebuild and evaluate the resulting panel and predictions.",
            "boundary": "This audit concerns this entry gate only; it does not establish point-in-time correctness or complete universe coverage.",
        },
        "reproduction_command": "python scripts/audit_entry_gate.py --raw-dir <LOCAL_RAW_DATA_DIR> --min-months 24 --min-days 10 --minimum-file-rows 60",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
