"""Scoring. The unit is a (farm, model, horizon, method) cell; nothing is pooled across them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conformal import ALPHA, default_methods, run_online_conformal
from .data import PROCESSED, SETTLEMENT_PERIOD, SETTLEMENT_STEP, Alignment, nwp_forecast
from .forecasting import HORIZONS, ISSUE_EVERY, common_issue_times

CELL_KEYS = ["farm", "model", "horizon", "method"]
STREAM_KEYS = ["farm", "model", "horizon"]

FARM_NAMES = ("kelmarsh", "penmanshiel")

SUBJECTS = {
    "persistence": "Persistence",
    "arima": "ARIMA",
    "lgbm_qr": "LightGBM-QR",
    "bootstrap": "Bootstrap ensemble",
    "tabpfn": "TabPFN",
    "gefs_power_curve": "GEFS power curve",
}

METHODS = {
    "native": "Native",
    "osscp": "OSSCP",
    "nexcp": "NexCP",
    "aci_0.01": "ACI (gamma=0.01)",
    "aci_0.05": "ACI (gamma=0.05)",
    "agaci": "AgACI",
    "dtaci": "DtACI",
    "mondrian": "Mondrian-OSSCP",
}

#: The rules OSSCP is compared against; Mondrian partitions rather than adapts.
ADAPTIVE_METHODS = ("nexcp", "aci_0.01", "aci_0.05", "agaci", "dtaci")

HORIZON_LABELS = {1: "30 min", 6: "3 h", 48: "24 h"}
LEDGER_HORIZON = 48

WIND_BANDS = (0.0, 5.0, 11.0, np.inf)
WIND_LABELS = ("Low (0-5 m/s)", "Average (5-11 m/s)", "High (11+ m/s)")


def winkler_score(y, lower, upper, alpha: float = ALPHA) -> np.ndarray:
    """Width plus 2/alpha times the distance outside. Lower is better."""
    y, lower, upper = np.asarray(y, float), np.asarray(lower, float), np.asarray(upper, float)
    return (upper - lower) + (2 / alpha) * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))


def feedback_delay(horizon: int, issue_every: int = ISSUE_EVERY) -> int:
    """Stream steps before a horizon's error is known; a left-labelled period closes one period late."""
    return max(1, -(-(horizon + 1) // issue_every))


def ratio_interval(numerator, denominator, blocks, draws: int = 2000, level: float = 0.95, seed: int = 0):
    """sum(numerator) / sum(denominator) with a block bootstrap interval: (estimate, low, high)."""
    by = pd.DataFrame({"num": numerator, "den": denominator, "block": blocks}).groupby("block")
    num, den = by["num"].sum().to_numpy(float), by["den"].sum().to_numpy(float)
    picks = np.random.default_rng(seed).integers(0, len(num), (draws, len(num)))
    resampled = num[picks].sum(axis=1) / den[picks].sum(axis=1)
    tail = (1 - level) / 2 * 100
    return float(num.sum() / den.sum()), *np.percentile(resampled, [tail, 100 - tail])


def week_of(issue_time: pd.Series) -> pd.Series:
    """The calendar week a row was issued in, the bootstrap block."""
    return issue_time.dt.tz_convert(None).dt.to_period("W").astype(str)


def coverage_interval(frame: pd.DataFrame, **kwargs):
    """Share of rows inside their band, with a weekly block-bootstrap interval. (estimate, low, high)."""
    covered = (frame["y_true"] >= frame["y_lower"]) & (frame["y_true"] <= frame["y_upper"])
    return ratio_interval(covered, np.ones(len(frame)), week_of(frame["issue_time"]), **kwargs)


def wind_band(wind_speed: pd.Series) -> pd.Series:
    """Bands at cut-in, ramp and near-rated: Mondrian's partition and the ledger's x-axis."""
    return pd.cut(wind_speed, WIND_BANDS, labels=list(WIND_LABELS), include_lowest=True)


def attach_conditions(evaluated: pd.DataFrame, nwp_grid: pd.DataFrame) -> pd.DataFrame:
    """Add the forecast wind speed each row was issued under, read at the lead the model saw."""
    step = pd.Timedelta(SETTLEMENT_PERIOD)
    parts = []
    for horizon, group in evaluated.groupby("horizon", sort=False):
        issue_times = pd.DatetimeIndex(group["issue_time"].unique()).sort_values()
        nwp = nwp_forecast(nwp_grid, issue_times, Alignment(lead=int(horizon) * step))
        parts.append(group.assign(wind_speed=group["issue_time"].map(nwp["wind_speed_80m"])))
    return pd.concat(parts, ignore_index=True)


def attach_realised_band(evaluated: pd.DataFrame, realised_wind: pd.Series) -> pd.DataFrame:
    """The wind band ERA5 says the target period fell in; for auditing only, never calibration."""
    target_time = evaluated["issue_time"] + evaluated["horizon"] * SETTLEMENT_STEP
    return evaluated.assign(
        wind_speed_realised=target_time.map(realised_wind),
        band_realised=lambda d: wind_band(d["wind_speed_realised"]),
    )


def _base_interval(group: pd.DataFrame):
    """A stream's native band, or None where the subject publishes a point forecast only."""
    if not {"y_lower", "y_upper"} <= set(group.columns):
        return None
    lower, upper = group["y_lower"].to_numpy(float), group["y_upper"].to_numpy(float)
    if np.isnan(lower).all() or np.isnan(upper).all():
        return None
    return lower, upper


def conformalize(
    stream: pd.DataFrame,
    methods: dict,
    stratum: str | None = None,
    **loop_kwargs,
) -> pd.DataFrame:
    """Wrap every rule around every stream, with the native band on the same scored rows.

    Returns one row per (farm, model, horizon, method, issue_time).
    """
    parts = []
    for _, group in stream.groupby(STREAM_KEYS, sort=False):
        group = group.sort_values("issue_time")
        horizon = int(group["horizon"].iloc[0])

        y = group["y_true"].to_numpy(float)
        base = _base_interval(group)
        lower_base, upper_base = base if base is not None else (None, None)

        strata = None
        if stratum is not None:
            labels = group[stratum]
            if labels.isna().any():
                raise ValueError(f"{int(labels.isna().sum())} rows carry no {stratum!r}")
            strata = labels.astype(str).to_numpy()

        arms = run_online_conformal(
            y,
            group["y_pred"].to_numpy(float),
            methods,
            lower_base=lower_base,
            upper_base=upper_base,
            strata=strata,
            feedback_delay=feedback_delay(horizon),
            issue_steps=(group["issue_time"] - group["issue_time"].iloc[0]).to_numpy()
            / (ISSUE_EVERY * SETTLEMENT_STEP).to_timedelta64(),
            **loop_kwargs,
        )

        scored = len(next(iter(arms.values()))[0])
        tail = group.iloc[-scored:]

        intervals = {} if base is None else {"native": (lower_base[-scored:], upper_base[-scored:])}
        intervals |= arms

        for method, (lower, upper) in intervals.items():
            parts.append(
                tail.assign(method=method, y_true=y[-scored:], y_lower=lower, y_upper=upper)
            )

    return pd.concat(parts, ignore_index=True)


def _interval_stats(group: pd.DataFrame, alpha: float) -> pd.Series:
    y, lower, upper = group["y_true"], group["y_lower"], group["y_upper"]
    return pd.Series(
        {
            "n": len(group),
            "coverage": float(((y >= lower) & (y <= upper)).mean()),
            "width": float((upper - lower).mean()),
            "winkler": float(winkler_score(y, lower, upper, alpha).mean()),
        }
    )


def scorecard(evaluated: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Coverage, mean width and mean Winkler score per cell."""
    return (
        evaluated.groupby(CELL_KEYS, sort=False)
        .apply(_interval_stats, alpha, include_groups=False)
        .reset_index()
    )


def wind_ledger(evaluated: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """The same scores split by forecast wind band. Bands are fixed, so counts are unequal."""
    framed = evaluated.assign(bucket=wind_band(evaluated["wind_speed"]))
    return (
        framed.groupby(CELL_KEYS + ["bucket"], observed=True, sort=False)
        .apply(_interval_stats, alpha, include_groups=False)
        .reset_index()
    )


def load_stream(farm: str) -> pd.DataFrame:
    """Every subject's forecasts, cut to the issue times all of them produced."""
    parts = [pd.read_parquet(PROCESSED / f"{farm}_forecasts.parquet")]
    tabpfn = PROCESSED / f"{farm}_tabpfn_forecasts.parquet"
    if tabpfn.exists():
        parts.append(pd.read_parquet(tabpfn))
    stream = pd.concat(parts, ignore_index=True)
    return common_issue_times(stream[stream["horizon"].isin(HORIZONS)])


def evaluate(farm: str, alpha: float = ALPHA) -> pd.DataFrame:
    """Conformalise every subject at one farm, and label each row by realised wind band."""
    stream = load_stream(farm)
    stream = attach_conditions(stream, pd.read_parquet(PROCESSED / f"{farm}_gefs_grid.parquet"))
    stream["band"] = wind_band(stream["wind_speed"])

    wrapped = conformalize(stream, default_methods(alpha), stratum="band")
    realised = pd.read_parquet(PROCESSED / f"{farm}_era5_wind_30min.parquet")
    return attach_realised_band(wrapped, realised["wind_speed_realised"])


def labelled(frame: pd.DataFrame, methods=tuple(METHODS)) -> pd.DataFrame:
    """Paper names and paper order, for a frame with model/method/horizon columns."""
    out = frame[frame["model"].isin(SUBJECTS) & frame["method"].isin(methods)].copy()
    out["model"] = pd.Categorical(out["model"], list(SUBJECTS), ordered=True)
    out["method"] = pd.Categorical(out["method"], list(methods), ordered=True)
    out = out.sort_values(["model", "method"])
    out["model"] = out["model"].map(SUBJECTS)
    out["method"] = out["method"].map(METHODS)
    if "horizon" in out:
        out["horizon"] = out["horizon"].map(HORIZON_LABELS)
    return out.rename(columns={"model": "Subject", "method": "Method"})


TEX_LABELS = {
    "ACI (gamma=0.01)": r"ACI ($\gamma=0.01$)",
    "ACI (gamma=0.05)": r"ACI ($\gamma=0.05$)",
    "Low (0-5 m/s)": "Low (0--5 m/s)",
    "Average (5-11 m/s)": "Average (5--11 m/s)",
    "High (11+ m/s)": "High (11+ m/s)",
    "kelmarsh": "Kelmarsh",
    "penmanshiel": "Penmanshiel",
    "coverage": "Coverage",
    "width": "Width",
    "winkler": "Winkler Score",
}


def write_latex(table: pd.DataFrame, path, float_format: str = "%.2f") -> None:
    """Write a paper table as a LaTeX tabular, index left, numbers centred."""
    table = table.rename(index=TEX_LABELS, columns=TEX_LABELS)
    table.columns.names = [None] * table.columns.nlevels
    latex = table.to_latex(
        column_format="@{}" + "l" * table.index.nlevels + "c" * table.shape[1] + "@{}",
        multirow=True,
        multicolumn_format="c",
        escape=False,
        float_format=float_format,
    )
    path.write_text(latex.replace(r"\cline", r"\cmidrule"), encoding="utf-8")
