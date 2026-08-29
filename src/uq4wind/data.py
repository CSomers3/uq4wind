"""SCADA loading and cleaning, GEFS and ERA5 retrieval, and the alignment between them."""

from __future__ import annotations

import glob
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

#: The repo root, two levels above this file. Data lives beside the source, not inside it.
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


class Farm(NamedTuple):
    name: str
    latitude: float
    longitude: float
    n_turbines: int
    turbine_rated_kw: float

    @property
    def capacity_mw(self) -> float:
        return self.n_turbines * self.turbine_rated_kw / 1000.0


#: Centroids only pick a GEFS grid cell, which is 0.25 degrees across.
FARMS = {
    "kelmarsh": Farm("Kelmarsh", 52.400604, -0.947133, 6, 2050.0),
    "penmanshiel": Farm("Penmanshiel", 55.904000, -2.305000, 14, 2050.0),
}

SETTLEMENT_PERIOD = "30min"
SETTLEMENT_STEP = pd.Timedelta(SETTLEMENT_PERIOD)

#: Longer gaps stay missing: those periods were removed as stopped, curtailed or de-rated.
MAX_INTERPOLATED_PERIODS = 1

#: Demanding full uptime would bias towards the months that had it.
MIN_REPORTING_FRACTION = 0.5

ERA5_ENDPOINT = "https://archive-api.open-meteo.com/v1/era5"
ERA5_PUBLISHED_HEIGHT = 100.0
ERA5_TARGET_HEIGHT = 80.0
ERA5_SHEAR_EXPONENT = 1 / 7
ERA5_TIMEOUT_SECONDS = 120

NWP_DATASET_ID = "noaa-gefs-forecast-35-day"
NWP_RUN_INTERVAL = pd.Timedelta("24h")

#: A cycle lands hours after its nominal init; assuming it available at init_time grants look-ahead.
NWP_PUBLICATION_LAG = pd.Timedelta("6h")

NWP_STEP = pd.Timedelta("3h")
NWP_MAX_LEAD_HOURS = 96

NWP_VARIABLES = [
    "wind_u_80m",
    "wind_v_80m",
    "temperature_80m",
    "pressure_80m",
    "precipitation_surface",
]
NWP_FEATURES = NWP_VARIABLES[:2] + ["wind_speed_80m"] + NWP_VARIABLES[2:]

#: A batch carries fixed graph-construction overhead, so small batches are worse.
NWP_FETCH_BATCH_SIZE = 100

#: Raw SCADA runs to 300+ columns; restricting parsing to these dominates load time.
SCADA_COLUMNS = {
    "# Date and time",
    "Data Availability",
    "Wind speed (m/s)",
    "Wind direction (°)",
    "Power (kW)",
    "Turbine Power setpoint (kW)",
    "Potential power default PC (kW)",
}

SCADA_READ_CHUNKSIZE = 250_000
SCADA_READ_WORKERS = 2


def load_scada(base_path: str | Path, farm: Farm):
    """Load raw 10-minute SCADA and status logs per turbine, searching base_path recursively.

    Returns (scada, logs), each keyed by turbine id ("T01", ...).
    """
    pattern = re.compile(f"(Turbine_Data|Status)_{farm.name}_(\\d+)_")
    jobs = []
    for path in sorted(glob.glob(f"{base_path}/**/*.csv", recursive=True)):
        match = pattern.search(Path(path).name)
        if match:
            kind, number = match.groups()
            jobs.append((f"T{int(number):02d}", kind == "Status", path))
    if not jobs:
        raise FileNotFoundError(f"no {farm.name} SCADA or status files found under {base_path}")

    def read(path: str, is_log: bool) -> pd.DataFrame:
        index_col = "Timestamp start" if is_log else "# Date and time"
        usecols = (
            None
            if is_log
            else (lambda c: c in SCADA_COLUMNS or ("Curtailment" in c and "kWh" in c))
        )
        chunks = pd.read_csv(
            path, skiprows=9, usecols=usecols, low_memory=False, chunksize=SCADA_READ_CHUNKSIZE
        )
        return pd.concat(chunks).set_index(index_col)

    with ThreadPoolExecutor(max_workers=SCADA_READ_WORKERS) as executor:
        frames = list(executor.map(lambda job: read(job[2], job[1]), jobs))

    scada_parts, log_parts = {}, {}
    for (turbine, is_log, _), frame in zip(jobs, frames):
        (log_parts if is_log else scada_parts).setdefault(turbine, []).append(frame)

    def joined(parts: list[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(parts)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    scada = {}
    for turbine, parts in sorted(scada_parts.items()):
        df = joined(parts)
        df.index.name = "timestamp"
        # After the availability filter, not before: the order changes which columns survive.
        df = df[df["Data Availability"] == 1].dropna(axis=1, how="all")
        scada[turbine] = df[~df.index.duplicated(keep="first")].sort_index()

    logs = {t: joined(parts).dropna(axis=1, how="all") for t, parts in sorted(log_parts.items())}
    return scada, logs


def clean_scada(df_scada: pd.DataFrame, df_log: pd.DataFrame) -> pd.DataFrame:
    """Drop periods logged as stopped, warning or curtailed; zero wind; and de-rated operation."""
    df = df_scada.copy()
    log = df_log.reset_index()

    excluded = pd.Series(False, index=df.index)
    for status in ("Stop", "Warning", "Curtailment"):
        spans = log.loc[log["Status"] == status, ["Timestamp start", "Timestamp end"]]
        spans = spans[spans["Timestamp end"] != "-"]
        for start, end in spans.itertuples(index=False):
            start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
            if pd.notna(start) and pd.notna(end):
                excluded.loc[start:end] = True

    curtailment_cols = [c for c in df.columns if "Curtailment" in c and "kWh" in c]
    no_curtailment = (df[curtailment_cols] == 0).all(axis=1) if curtailment_cols else True

    keep = (
        ~excluded
        & (df["Wind speed (m/s)"] > 0.1)
        & no_curtailment
        & (df["Turbine Power setpoint (kW)"] >= 0.99 * df["Potential power default PC (kW)"])
    )
    return df[keep]


def resample_scada(
    df: pd.DataFrame,
    freq: str = SETTLEMENT_PERIOD,
    max_gap_periods: int = MAX_INTERPOLATED_PERIODS,
) -> pd.DataFrame:
    """Resample cleaned SCADA to freq, bridging only gaps no longer than max_gap_periods."""
    resampled = df.resample(freq).mean(numeric_only=True)

    # interpolate(limit=...) fills the first `limit` values of any gap, so length is measured here.
    empty = resampled.isna().all(axis=1)
    gap_length = empty.groupby((~empty).cumsum()).transform("sum")
    bridgeable = empty & (gap_length <= max_gap_periods)

    resampled.loc[bridgeable] = resampled.interpolate(
        method="linear", limit_area="inside"
    ).loc[bridgeable]
    return resampled


def aggregate_farm_capacity_factor(
    scada_by_turbine: dict[str, pd.DataFrame],
    turbine_rated_kw: float,
    power_col: str = "Power (kW)",
    min_reporting_fraction: float = MIN_REPORTING_FRACTION,
) -> pd.Series:
    """Mean of each reporting turbine's own capacity factor, where enough turbines report.

    Averaging rather than summing kW keeps the result independent of how many turbines
    reported. The result is left as metered: parasitic draw reads marginally below zero and
    brief over-rating marginally above one, so no row sits exactly on a band edge.
    """
    power = pd.concat({tid: df[power_col] for tid, df in scada_by_turbine.items()}, axis=1)
    min_reporting = int(np.ceil(min_reporting_fraction * power.shape[1]))

    capacity_factor = (power / turbine_rated_kw).mean(axis=1, skipna=True)
    capacity_factor[power.notna().sum(axis=1) < min_reporting] = np.nan
    capacity_factor.name = "capacity_factor"
    return capacity_factor.asfreq(SETTLEMENT_PERIOD)


def fetch_nwp_ensemble_grid(
    start: str,
    end: str,
    farm: Farm,
    max_lead_hours: int = NWP_MAX_LEAD_HOURS,
    run_interval: pd.Timedelta = NWP_RUN_INTERVAL,
) -> pd.DataFrame:
    """Per-run, per-member GEFS forecasts at the farm centroid.

    Indexed by (init_time, valid_time, ensemble_member).
    """
    import dynamical_catalog
    import xarray as xr

    dataset = dynamical_catalog.open(NWP_DATASET_ID, chunks="auto")
    point = dataset[NWP_VARIABLES].sel(
        latitude=farm.latitude, longitude=farm.longitude, method="nearest"
    )

    # One run before `start`, so early issue times still have a published run to draw on.
    init_times = pd.date_range(
        pd.Timestamp(start).floor(run_interval) - run_interval, end, freq=run_interval, tz="UTC"
    ).tz_localize(None)
    # Lead 0 is excluded: precipitation accumulates since the previous step.
    lead_times = pd.to_timedelta(
        np.arange(NWP_STEP / pd.Timedelta("1h"), max_lead_hours + 1, NWP_STEP / pd.Timedelta("1h")),
        unit="h",
    )

    fetched = [
        point.sel(
            init_time=init_times[i : i + NWP_FETCH_BATCH_SIZE],
            lead_time=lead_times,
            method="nearest",
        ).compute()
        for i in range(0, len(init_times), NWP_FETCH_BATCH_SIZE)
    ]
    grid = xr.concat(fetched, dim="init_time").to_dataframe()[NWP_VARIABLES].reset_index()

    u, v = grid["wind_u_80m"], grid["wind_v_80m"]
    grid["wind_speed_80m"] = np.sqrt(u**2 + v**2)
    grid["nwp_lead_hours"] = grid["lead_time"] / pd.Timedelta("1h")
    grid["init_time"] = grid["init_time"].dt.tz_localize("UTC")
    grid["valid_time"] = grid["init_time"] + grid["lead_time"]

    grid = grid.set_index(["init_time", "valid_time", "ensemble_member"])
    grid = grid[NWP_FEATURES + ["nwp_lead_hours"]].sort_index()

    # An absent run snaps to its neighbour and appears twice; one copy leaves it missing.
    grid = grid[~grid.index.duplicated(keep="first")]
    missing = len(init_times) - grid.index.get_level_values("init_time").nunique()
    if missing:
        print(f"{missing} of {len(init_times)} GEFS runs absent from the archive")
    return grid


def fetch_era5_wind(start: str, end: str, farm: Farm) -> pd.Series:
    """Realised hub-height wind speed, hourly ERA5 resampled onto settlement periods.

    Open-Meteo publishes 100m; the bands are cut at 80m, so it is brought down by the power
    law at the open-terrain exponent. Both sides of the taxonomy then mean the same thing.
    """
    query = urllib.parse.urlencode(
        {
            "latitude": farm.latitude,
            "longitude": farm.longitude,
            "start_date": start,
            "end_date": end,
            "hourly": "wind_speed_100m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        }
    )
    with urllib.request.urlopen(
        f"{ERA5_ENDPOINT}?{query}", timeout=ERA5_TIMEOUT_SECONDS
    ) as response:
        hourly = json.load(response)["hourly"]

    speed = pd.Series(
        hourly["wind_speed_100m"],
        index=pd.DatetimeIndex(hourly["time"], tz="UTC"),
        dtype=float,
        name="wind_speed_realised",
    )
    speed *= (ERA5_TARGET_HEIGHT / ERA5_PUBLISHED_HEIGHT) ** ERA5_SHEAR_EXPONENT
    return speed.resample(SETTLEMENT_PERIOD).asfreq().interpolate("time")


class Alignment(NamedTuple):
    """How an issue time reaches a forecast: the lead wanted, against the release cadence.

    The three travel together because overriding one alone would quietly grant look-ahead.
    """

    lead: pd.Timedelta = pd.Timedelta(0)
    publication_lag: pd.Timedelta = NWP_PUBLICATION_LAG
    run_interval: pd.Timedelta = NWP_RUN_INTERVAL

    def resolve(self, issue_times: pd.DatetimeIndex) -> pd.MultiIndex:
        """The run each issue time may draw on, paired with the step serving its target."""
        runs = (issue_times - self.publication_lag).floor(self.run_interval)
        valid_times = (issue_times + self.lead).floor(NWP_STEP)
        return pd.MultiIndex.from_arrays([runs, valid_times])


def nwp_ensemble_speeds(
    grid: pd.DataFrame,
    issue_times: pd.DatetimeIndex,
    alignment: Alignment = Alignment(),
) -> pd.DataFrame:
    """Every member's 80m speed for the aligned target, one column per member."""
    speeds = grid["wind_speed_80m"].unstack("ensemble_member")
    speeds = speeds.reindex(alignment.resolve(issue_times))
    speeds.index = issue_times
    return speeds


def nwp_forecast(
    grid: pd.DataFrame,
    issue_times: pd.DatetimeIndex,
    alignment: Alignment = Alignment(),
) -> pd.DataFrame:
    """The same forecast summarised over members: a central value, a spread, and its staleness.

    Speed is averaged directly but direction comes from the mean vector, since the circular
    mean of 350 and 10 degrees is 0, not 180. Rows with no published run are NaN.
    """
    target = alignment.resolve(issue_times)
    valid_times = target.get_level_values(1)

    by_run = grid.groupby(level=["init_time", "valid_time"])
    mean = by_run.mean().reindex(target)
    speed = grid["wind_speed_80m"].groupby(level=["init_time", "valid_time"])

    forecast = pd.DataFrame(index=issue_times)
    for column in ("wind_speed_80m", "temperature_80m", "pressure_80m", "precipitation_surface"):
        forecast[column] = mean[column].to_numpy()

    u, v = mean["wind_u_80m"].to_numpy(), mean["wind_v_80m"].to_numpy()
    forecast["wind_direction_80m"] = np.degrees(np.arctan2(-u, -v)) % 360

    forecast["nwp_spread_80m"] = speed.std().reindex(target).to_numpy()
    forecast["nwp_spread_p10_80m"] = speed.quantile(0.10).reindex(target).to_numpy()
    forecast["nwp_spread_p90_80m"] = speed.quantile(0.90).reindex(target).to_numpy()

    forecast["nwp_lead_hours"] = mean["nwp_lead_hours"].to_numpy()
    forecast["nwp_age_minutes"] = (issue_times + alignment.lead - valid_times) / pd.Timedelta("1min")
    return forecast
