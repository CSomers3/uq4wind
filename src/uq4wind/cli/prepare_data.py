"""Turn raw SCADA and the GEFS archive into the parquet everything else reads.

Usage:
    python scripts/prepare_data.py kelmarsh --raw data/raw/kelmarsh
    python scripts/prepare_data.py kelmarsh --nwp-only
    python scripts/prepare_data.py kelmarsh --era5-only
"""

from __future__ import annotations

import argparse

from .. import data as D
from ..data import PROCESSED

#: GEFS coverage begins 2020-10-01; the SCADA releases end with 2024.
NWP_START, NWP_END = "2022-01-01", "2024-12-31"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("farm", choices=sorted(D.FARMS))
    parser.add_argument("--raw", help="directory holding the farm's extracted SCADA CSVs")
    parser.add_argument("--nwp-only", action="store_true")
    parser.add_argument("--era5-only", action="store_true")
    parser.add_argument("--scada-only", action="store_true")
    args = parser.parse_args()

    weather_only = args.nwp_only or args.era5_only
    if not weather_only and not args.raw:
        parser.error("--raw is required unless --nwp-only or --era5-only is given")

    farm = D.FARMS[args.farm]
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if not weather_only:
        scada, logs = D.load_scada(args.raw, farm)
        print(f"{farm.name}: {len(scada)} turbines loaded")

        resampled = {}
        for turbine, raw in scada.items():
            cleaned = D.clean_scada(raw, logs[turbine])
            resampled[turbine] = D.resample_scada(cleaned)
            kept = 100 * len(cleaned) / len(raw)
            print(f"  {turbine}: {len(raw):>7} rows -> {kept:.1f}% survive cleaning")

        capacity_factor = D.aggregate_farm_capacity_factor(resampled, farm.turbine_rated_kw)
        capacity_factor.to_frame().to_parquet(
            PROCESSED / f"{args.farm}_capacity_factor_30min.parquet"
        )
        print(
            f"capacity factor: {capacity_factor.notna().sum()} of {len(capacity_factor)} periods, "
            f"{capacity_factor.index[0]} to {capacity_factor.index[-1]}"
        )

    if args.scada_only:
        return

    if not args.nwp_only:
        realised = D.fetch_era5_wind(NWP_START, NWP_END, farm)
        realised.to_frame().to_parquet(PROCESSED / f"{args.farm}_era5_wind_30min.parquet")
        print(f"ERA5 wind: {realised.notna().sum()} periods at {D.ERA5_TARGET_HEIGHT:.0f}m")

    if args.era5_only:
        return

    grid = D.fetch_nwp_ensemble_grid(NWP_START, NWP_END, farm)
    grid.to_parquet(PROCESSED / f"{args.farm}_gefs_grid.parquet")
    runs = grid.index.get_level_values("init_time").nunique()
    print(f"GEFS grid: {len(grid)} rows across {runs} runs")


if __name__ == "__main__":
    main()
