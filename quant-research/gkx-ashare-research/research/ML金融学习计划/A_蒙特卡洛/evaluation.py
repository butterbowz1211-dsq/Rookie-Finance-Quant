"""Shared, explicitly named evaluation rules. Author: Gilmour Jia."""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT = {"train": ("2010-01", "2018-12"),
         "valid": ("2019-01", "2021-12"),
         "test": ("2022-01", "2025-12")}


def _r2(y, prediction, benchmark):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y):
        raise ValueError("y and predictions must be nonempty matching vectors")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Non-finite values cannot be silently removed when scoring")
    denominator = np.sum((y - benchmark) ** 2, dtype=np.float64)
    return float(1 - np.sum((y - prediction) ** 2, dtype=np.float64) / denominator) if denominator > 0 else float("nan")


def r2_zero(y, prediction):
    """Pooled panel R² versus a zero-return prediction; fraction, not percent."""
    return _r2(y, prediction, 0.0)


def r2_sample_mean(y, prediction):
    """Diagnostic using the evaluation sample's mean: not a feasible forecast."""
    return _r2(y, prediction, np.asarray(y, dtype=float).mean())


def r2_training_mean(y, prediction, training_mean):
    """R² versus a mean estimated exclusively from the training sample."""
    return _r2(y, prediction, float(training_mean))


def r2_monthly_average(y, prediction, months):
    """Equal-month average of month-specific R²; distinct from pooled GKX R²."""
    values = pd.DataFrame({"y": y, "p": prediction, "month": months})
    scores = [r2_zero(g.y, g.p) for _, g in values.groupby("month")]
    return float(np.nanmean(scores))


def rank_characteristics(df, columns):
    """Rank available characteristics before filtering on future labels."""
    result = df.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        grouped = values.groupby(result["month"], sort=False)
        rank, count = grouped.rank(method="average"), grouped.transform("count")
        result[column] = (2 * rank / (count + 1) - 1).fillna(0).astype(np.float32)
    return result


def label_months(df):
    expected = (pd.PeriodIndex(df["month"], freq="M") + 1).astype(str)
    if "label_month" not in df:
        raise ValueError("Rebuild features: label_month is required; legacy shift(-1) labels are not accepted")
    if not np.array_equal(df["label_month"].astype(str).to_numpy(), expected.to_numpy()):
        raise ValueError("label_month must be exactly the next calendar month")
    return df["label_month"].astype(str)


def split_masks(df, split=None):
    months = label_months(df)
    split = SPLIT if split is None else split
    masks = {name: months.between(lo, hi).to_numpy() for name, (lo, hi) in split.items()}
    if np.any(np.sum(list(masks.values()), axis=0) > 1):
        raise ValueError("Target-month windows overlap")
    return masks


def encode_industry(labels, train_mask, min_count=100):
    """Learn vocabulary/frequency threshold on training rows only; drop reference."""
    labels = pd.Series(labels).fillna("UNKNOWN").astype(str).reset_index(drop=True)
    frequencies = labels.loc[np.asarray(train_mask, dtype=bool)].value_counts()
    retained = sorted(frequencies[frequencies >= min_count].index)
    # OTHER is always defined so unseen test industries do not create new columns.
    vocabulary = sorted(set(retained) | {"OTHER"})
    mapped = labels.where(labels.isin(retained), "OTHER")
    reference = vocabulary[0]
    columns = {f"ind_{name}": (mapped == name).to_numpy(dtype=np.float32)
               for name in vocabulary if name != reference}
    return pd.DataFrame(columns, index=labels.index), {"retained": retained, "reference": reference,
                                                       "min_training_count": min_count}


def portfolio_statistics(values):
    """Monthly total-return diagnostics; long-only ratio assumes rf=0 explicitly."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or (x <= -1).any():
        raise ValueError("At least two finite monthly returns > -1 required")
    wealth = np.concatenate(([1.0], np.cumprod(1 + x)))
    sd = x.std(ddof=1)
    return {"months": len(x), "monthly_mean": float(x.mean()),
            "monthly_sd": float(sd), "annualized_mean_over_sd_rf0": float(x.mean() / sd * np.sqrt(12)) if sd else None,
            "cagr": float(wealth[-1] ** (12 / len(x)) - 1),
            "cumulative_return": float(wealth[-1] - 1),
            "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1))}
