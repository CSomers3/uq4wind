"""Run the expanding-window backtest and write the forecast stream.

Five subjects run by default. TabPFN is the sixth and needs `--tabpfn`, because it is a
hosted service wanting TABPFN_API_KEY. It is checkpointed per horizon, so hitting the daily
cap costs one horizon rather than the run, and the checkpoint name carries a digest of the
inputs behind it so a changed input is recomputed rather than silently reused.

Penmanshiel's record starts later than Kelmarsh's, so at the default 730D no retrain window
ever opens and the run fails rather than writing an empty file.

Usage:
    python scripts/run_backtest.py kelmarsh
    python scripts/run_backtest.py penmanshiel --min-train-window 365D
    TABPFN_API_KEY=... python scripts/run_backtest.py kelmarsh --tabpfn
"""

from __future__ import annotations

import argparse
import hashlib

import pandas as pd

from .. import data as D
from .. import forecasting as F
from ..data import PROCESSED

VINTAGE_CHARS = 10


def vintage(features, targets, schedule, forecaster) -> str:
    """Digest of everything the predictions depend on beyond the farm and horizon."""
    identity = (
        tuple(features.columns),
        pd.util.hash_pandas_object(features, index=True).sum(),
        pd.util.hash_pandas_object(targets, index=True).sum(),
        tuple(str(field) for field in schedule),
        forecaster.model_path,
        forecaster.n_estimators,
        forecaster.max_train_rows,
        forecaster.alpha,
    )
    return hashlib.sha256(repr(identity).encode()).hexdigest()[:VINTAGE_CHARS]


def run_standard(farm, capacity_factor, nwp_grid, features, targets, schedule) -> pd.DataFrame:
    """The five subjects needing no credentials."""
    streams = []
    for name, factory in (
        ("persistence", lambda: F.PersistenceForecaster(F.ALPHA)),
        ("lgbm_qr", lambda: F.QuantileIntervalForecaster(F.ALPHA)),
        ("bootstrap", lambda: F.BootstrapEnsembleForecaster(F.ALPHA)),
    ):
        print(f"{name}...", flush=True)
        streams.append(
            F.run_tabular_backtest(
                features, targets, factory, capacity_factor.index, schedule
            ).assign(model=name)
        )

    # This subject reads a different set of members per horizon, so it runs one at a time.
    print("gefs_power_curve...", flush=True)
    for horizon in F.HORIZONS:
        member_speeds = D.nwp_ensemble_speeds(
            nwp_grid, capacity_factor.index, D.Alignment(lead=horizon * D.SETTLEMENT_STEP)
        )
        streams.append(
            F.run_tabular_backtest(
                {horizon: features[horizon]},
                targets,
                lambda speeds=member_speeds: F.EnsemblePowerCurveForecaster(F.ALPHA, speeds),
                capacity_factor.index,
                schedule,
            ).assign(model="gefs_power_curve")
        )

    print("arima...", flush=True)
    streams.append(F.run_arima_backtest(capacity_factor, schedule=schedule).assign(model="arima"))
    return pd.concat(streams, ignore_index=True).assign(farm=farm)


def run_tabpfn(farm, capacity_factor, nwp_grid, targets, schedule) -> pd.DataFrame:
    """The hosted subject, checkpointed per horizon."""
    checkpoints = []
    for horizon in F.HORIZONS:
        features = F.build_features(capacity_factor, nwp_grid, horizon)
        digest = vintage(features, targets[horizon], schedule, F.TabPFNForecaster(F.ALPHA))

        checkpoint = PROCESSED / f"{farm}_tabpfn_h{horizon}_{digest}.parquet"
        checkpoints.append(checkpoint)
        if checkpoint.exists():
            print(f"  h={horizon}: reusing {checkpoint.name}")
            continue

        result = F.run_tabular_backtest(
            {horizon: features},
            targets,
            lambda: F.TabPFNForecaster(F.ALPHA),
            capacity_factor.index,
            schedule,
        )
        result.to_parquet(checkpoint)
        print(f"  h={horizon}: {len(result)} predictions -> {checkpoint.name}")

    return pd.concat([pd.read_parquet(p) for p in checkpoints], ignore_index=True).assign(
        model="tabpfn", farm=farm
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("farm", choices=sorted(D.FARMS))
    parser.add_argument("--min-train-window", default=str(F.MIN_TRAIN_WINDOW))
    parser.add_argument("--tabpfn", action="store_true", help="run the hosted TabPFN subject")
    args = parser.parse_args()
    schedule = F.Schedule(min_train_window=pd.Timedelta(args.min_train_window))

    capacity_factor = pd.read_parquet(
        PROCESSED / f"{args.farm}_capacity_factor_30min.parquet"
    )["capacity_factor"]
    nwp_grid = pd.read_parquet(PROCESSED / f"{args.farm}_gefs_grid.parquet")
    targets = F.build_horizon_targets(capacity_factor, F.HORIZONS)

    if args.tabpfn:
        forecasts = run_tabpfn(args.farm, capacity_factor, nwp_grid, targets, schedule)
        out = PROCESSED / f"{args.farm}_tabpfn_forecasts.parquet"
    else:
        features = {h: F.build_features(capacity_factor, nwp_grid, h) for h in F.HORIZONS}
        forecasts = run_standard(
            args.farm, capacity_factor, nwp_grid, features, targets, schedule
        )
        out = PROCESSED / f"{args.farm}_forecasts.parquet"

    if forecasts.empty:
        raise RuntimeError(
            f"no forecasts produced: {args.min_train_window} leaves no retrain window inside "
            f"{capacity_factor.index[0]} to {capacity_factor.index[-1]}"
        )

    forecasts.to_parquet(out)
    print(f"\n{out.name}: {len(forecasts)} rows")
    print(forecasts.groupby("model").size().to_string())


if __name__ == "__main__":
    main()
