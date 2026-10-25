"""Data contracts for Gilmour Jia's A-share research pipeline.

Input caches are never modified. GKX_DERIVED_DIR selects a separate destination.
The contracts fix implementation errors; they cannot certify vendor vintages.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GKX_DATA_DIR", HERE / "data"))
DERIVED = Path(os.environ.get("GKX_DERIVED_DIR", HERE / "data" / "repaired"))
CLEAN = Path(os.environ.get("GKX_CLEAN_DIR", HERE.parents[1] / "A股行业时点与ST数据清洗" / "01_data"))


def ensure_derived_directory() -> None:
    if DATA.resolve() == DERIVED.resolve():
        raise ValueError("GKX_DERIVED_DIR must differ from the original GKX_DATA_DIR")
    DERIVED.mkdir(parents=True, exist_ok=True)


def require_unique(frame: pd.DataFrame, keys: list[str]) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate keys in {keys}; resolve provenance before merging")


def attach_forward_returns(features: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Match t to exactly t+1, never to the next available observation."""
    require_unique(features, ["code", "month"])
    require_unique(returns, ["code", "month"])
    out = features.drop(columns=["ret_fwd", "label_month"], errors="ignore").copy()
    out["label_month"] = (pd.PeriodIndex(out["month"], freq="M") + 1).astype(str)
    target = returns[["code", "month", "ret"]].rename(
        columns={"month": "label_month", "ret": "ret_fwd"})
    return out.merge(target, on=["code", "label_month"], how="left", validate="one_to_one")


def calendar_report_lag(frame: pd.DataFrame, values: pd.Series, quarters: int = 4,
                        code_col: str = "SECURITY_CODE") -> pd.Series:
    """Lag an exact fiscal quarter, preserving a gap as missing."""
    periods = pd.PeriodIndex(frame["REPORT_DATE"], freq="Q")
    idx = pd.MultiIndex.from_arrays([frame[code_col], periods])
    if idx.has_duplicates:
        raise ValueError("Multiple statement versions require explicit vintage selection")
    source = pd.Series(np.asarray(values), index=idx)
    wanted = pd.MultiIndex.from_arrays([frame[code_col], periods - quarters])
    return pd.Series(source.reindex(wanted).to_numpy(), index=frame.index)


def ttm_values(frame: pd.DataFrame, col: str, code_col: str = "SECURITY_CODE"):
    """Return TTM and conservative availability date for its three inputs."""
    periods = pd.PeriodIndex(frame["REPORT_DATE"], freq="Q")
    idx = pd.MultiIndex.from_arrays([frame[code_col], periods])
    if idx.has_duplicates:
        raise ValueError("Duplicate report periods cannot be used for TTM")
    vals = pd.Series(pd.to_numeric(frame[col], errors="coerce").to_numpy(), index=idx)
    dates = pd.Series(pd.to_datetime(frame["NOTICE_DATE"]).to_numpy(), index=idx)
    previous_year_end = pd.PeriodIndex([f"{p.year - 1}Q4" for p in periods], freq="Q")
    key_fy = pd.MultiIndex.from_arrays([frame[code_col], previous_year_end])
    key_prev = pd.MultiIndex.from_arrays([frame[code_col], periods - 4])
    fy, prior = vals.reindex(key_fy).to_numpy(), vals.reindex(key_prev).to_numpy()
    cur = pd.to_numeric(frame[col], errors="coerce").to_numpy()
    annual = periods.quarter == 4
    result = np.where(annual, cur, cur + fy - prior)
    dependencies = pd.DataFrame({
        "cur": pd.to_datetime(frame["NOTICE_DATE"]).to_numpy(),
        "fy": dates.reindex(key_fy).to_numpy(),
        "prior": dates.reindex(key_prev).to_numpy(),
    })
    available = dependencies.max(axis=1)
    available = available.mask(~annual & dependencies.isna().any(axis=1))
    available = available.where(~annual, pd.Series(pd.to_datetime(frame["NOTICE_DATE"]).to_numpy()))
    return pd.Series(result, index=frame.index), pd.Series(available.to_numpy(), index=frame.index)


def require_financial_vintage_opt_in(allow: bool) -> None:
    if not allow:
        raise SystemExit(
            "Financial cache vintages are unverified. Historical revisions cannot be recovered "
            "from NOTICE_DATE alone. Supply --allow-unverified-vintages only for an explicitly "
            "exploratory run; do not claim point-in-time validation.")


def raw_market_cap(frame: pd.DataFrame) -> pd.Series:
    """Never reconstruct capitalisation from a ranked/winsorised size factor."""
    if "market_cap" not in frame:
        raise ValueError("Rebuild features: raw market_cap is required (mvel1 is transformed)")
    cap = pd.to_numeric(frame["market_cap"], errors="coerce")
    return cap.where(np.isfinite(cap) & (cap > 0))
