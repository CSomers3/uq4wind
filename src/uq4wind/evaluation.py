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

#: The rules OSSCP is compared against. Native is the subject, Mondrian partitions rather than adapts.
ADAPTIVE_METHODS = ("nexcp", "aci_0.01", "aci_0.05", "agaci", "dtaci")

HORIZON_LABELS = {1: "30 min", 6: "3 h", 48: "24 h"}
LEDGER_HORIZON = 48

#: Fixed cut points either side of partial load, so "high wind" names one regime at both farms.
WIND_BANDS = (0.0, 5.0, 11.0, np.inf)
WIND_LABELS = ("Low (0-5 m/s)", "Average (5-11 m/s)", "High (11+ m/s)")


def winkler_score(y, lower, upper, alpha: float = ALPHA) -> np.ndarray:
    """Width plus 2/alpha times the distance outside. Lower is better."""
    y, lower, upper = np.asarray(y, float), np.asarray(lower, float), np.asarray(upper, float)
    return (upper - lower) + (2 / alpha) * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))


def feedback_delay(horizon: int, issue_every: int = ISSUE_EVERY) -> int:
    """Stream steps that must pass before a horizon's error is known.

    A settlement period is left-labelled, so the mean stamped t + horizon is not complete
    until one period after that. The extra period is what removes the look-ahead.
    """
    return max(1, -(-(horizon + 1) // issue_every))


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
    """The band the outturn fell in, against the forecast band Mondrian calibrated within.

    Calibration can only use what the issue time knew, so a Mondrian arm is audited on a
    partition it was never allowed to see. The gap between the two is the taxonomy's error.
    """
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
    alpha: float = ALPHA,
    stratum: str | None = None,
    **loop_kwargs,
) -> pd.DataFrame:
    """Wrap every rule around every stream, returning wrapped intervals alongside the native one.

    Each stream is conformalised on its own with the feedback delay set from its horizon.
    The native interval is emitted over exactly the rows the wrapped ones cover, so the
    comparison is on one sample.

    Neither the target nor the intervals are clipped to [0, 1]. Clipping puts an atom at a
    conformity score of exactly zero for a subject whose band is already censored there,
    which straddles the level the quantile is read at and makes the wrapper the identity.

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
            feedback_delay=feedback_delay(horizon),
            strata=strata,
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
    """Conformalise every subject at one farm, and label each row by realised wind band.

    The forecast wind band is attached before wrapping, not after: Mondrian calibrates
    within it, so it has to be on the stream the loop walks.
    """
    stream = load_stream(farm)
    stream = attach_conditions(stream, pd.read_parquet(PROCESSED / f"{farm}_gefs_grid.parquet"))
    stream["band"] = wind_band(stream["wind_speed"])

    wrapped = conformalize(stream, default_methods(alpha), alpha=alpha, stratum="band")
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
