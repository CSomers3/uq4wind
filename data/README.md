# Data

Generate the data locally with the commands below. Raw files, processed parquet and notebook
outputs are excluded from version control.

## Sources

- SCADA: [Kelmarsh](https://doi.org/10.5281/zenodo.16807551) and
  [Penmanshiel](https://doi.org/10.5281/zenodo.16807304), Plumley and Takeuchi (2025), CC-BY-4.0.
  Extract each archive, including turbine data and status logs, into `raw/<farm>/`.
  Keep the original CSV names and headers; the loader searches subdirectories.
- Forecast weather: [NOAA GEFS via dynamical.org](https://dynamical.org/catalog/noaa-gefs-forecast-35-day/),
  dataset `noaa-gefs-forecast-35-day`. The preparation command retrieves 31 members at the farm
  centroid, using daily 00Z runs and 3-hourly leads through 96 hours.
- Realised wind: [ERA5 via Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api),
  hourly wind at 100 m, scaled to 80 m by `(80/100)**(1/7)` and interpolated to 30 minutes.
  This is used only for retrospective regime evaluation.

## Cleaning

`uq4wind-prepare` keeps available SCADA records, removes duplicate timestamps, and excludes
periods logged as Stop, Warning or Curtailment. It also excludes wind speeds at or below
0.1 m/s, nonzero curtailment energy, and power setpoints below 99% of potential power.

Each turbine is averaged to 30-minute periods. Farm capacity factor is the mean reporting
turbine power divided by 2,050 kW, where at least half the farm reports: 3 of 6 turbines at
Kelmarsh or 7 of 14 at Penmanshiel. Excluded periods remain missing; targets are neither
interpolated nor clipped to [0, 1].

The backtest aligns GEFS to each target with a six-hour publication lag. Training targets are
purged by horizon, and calibration waits until each target settlement period has closed.

## Build

Run from the repository root after installing the dependencies in the main README:

```bash
uv run uq4wind-prepare kelmarsh --raw data/raw/kelmarsh --scada-only
uv run uq4wind-prepare penmanshiel --raw data/raw/penmanshiel --scada-only
for farm in kelmarsh penmanshiel; do
    uv run --group data uq4wind-prepare "$farm" --nwp-only
    uv run uq4wind-prepare "$farm" --era5-only
done
uv run uq4wind-backtest kelmarsh
uv run uq4wind-backtest penmanshiel --min-train-window 365D
TABPFN_API_KEY=... uv run --group tabpfn uq4wind-backtest kelmarsh --tabpfn
TABPFN_API_KEY=... uv run --group tabpfn uq4wind-backtest penmanshiel --tabpfn --min-train-window 365D
```

Weather retrieval covers 2022-2024, set in `src/uq4wind/cli/prepare_data.py`. TabPFN calls a hosted,
metered service and saves checkpoints locally; rerunning it may depend on the service's model version.

## Required format

Write these files into `data/processed/` for each `<farm>` (`kelmarsh` or `penmanshiel`).
Use UTC timestamps. The 30-minute time indexes must be sorted, unique and regular, with
missing observations represented as NaN.

| File suffix after `<farm>_` | Index | Columns |
|---|---|---|
| `capacity_factor_30min.parquet` | 30-minute DatetimeIndex | `capacity_factor` |
| `gefs_grid.parquet` | MultiIndex: `init_time`, `valid_time`, `ensemble_member` | Weather columns below |
| `era5_wind_30min.parquet` | 30-minute DatetimeIndex | `wind_speed_realised` (m/s) |
| `forecasts.parquet` | Default row index | Forecast columns below; five self-hosted models |
| `tabpfn_forecasts.parquet` | Default row index | Same forecast columns; `model = tabpfn` |

GEFS columns: `wind_u_80m`, `wind_v_80m`, `wind_speed_80m` (m/s), `temperature_80m` (C),
`pressure_80m` (Pa), `precipitation_surface`, and `nwp_lead_hours`.
`ensemble_member` runs from 0 to 30; `valid_time = init_time + lead_time`.

Forecast columns: `farm`, `model`, `issue_time`, `horizon`, `y_true`, `y_pred`, `y_lower`, `y_upper`.
Horizons are integers in settlement periods: 1, 6 and 48. Predictions, targets and endpoints are
capacity factors; endpoints must be ordered. Each `(farm, model, issue_time, horizon)` is unique,
and `y_true` must equal the target at `issue_time + horizon * 30 minutes`.
`issue_time` left-labels the latest observed SCADA period; physical issuance is at its end.

The five model names are `persistence`, `arima`, `lgbm_qr`, `bootstrap` and `gefs_power_curve`.
The notebooks require all six models and compare them on common issue-time/horizon pairs.
