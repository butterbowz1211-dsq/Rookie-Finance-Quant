"""Synthetic smoke test; produces no A-share empirical claim. Author: Gilmour Jia."""
from __future__ import annotations
import argparse
import os
import pathlib
import sys
import tempfile
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--nn", action="store_true", help="Also exercise a small NN ensemble")
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    output = args.output or pathlib.Path(tempfile.mkdtemp(prefix="gkx-smoke-results-"))
    with tempfile.TemporaryDirectory(prefix="gkx-synthetic-input-") as tmp:
        os.environ["GKX_DERIVED_DIR"] = tmp
        os.environ["GKX_OUT_DIR"] = str(output)
        sys.path.insert(0, str(root / "research" / "ML金融学习计划" / "D_A股数据"))
        import run_ashare
        rng = np.random.default_rng(20261025)
        months = pd.period_range("2009-12", "2025-11", freq="M")
        rows = []
        for month in months:
            for stock in range(24):
                x = rng.normal(size=len(run_ashare.FEATS))
                rows.append({"code": f"SYN{stock:03d}", "month": str(month),
                             "label_month": str(month+1), "eligible_asof": True,
                             "market_cap": float(np.exp(20+x[0])),
                             "ret_fwd": float(0.015*x[0] + 0.01*x[1]*x[2] + rng.normal(scale=.05)),
                             **dict(zip(run_ashare.FEATS, x))})
        pd.DataFrame(rows).to_csv(pathlib.Path(tmp) / "features_monthly.csv", index=False)
        models = "ridge,rf,gbrt" + (",nn" if args.nn else "")
        run_ashare.main(["--models", models, "--rf-trees", "8", "--rf-leaf", "10",
                        "--gb-trees", "12", "--gb-leaf", "10", "--nn-depth", "1" if args.nn else "0",
                        "--nn-ensemble", "2", "--nn-epochs", "5", "--tag", "synthetic"])
    print(f"SYNTHETIC ONLY: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
