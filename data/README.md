# Data

## Sources

- SCADA: [Kelmarsh](https://doi.org/10.5281/zenodo.16807551) and [Penmanshiel](https://doi.org/10.5281/zenodo.16807304), Plumley and Takeuchi (2025), CC-BY-4.0.
- Forecast weather: [NOAA GEFS](https://dynamical.org/catalog/noaa-gefs-forecast-35-day/), dataset `noaa-gefs-forecast-35-day`.
- Realised wind: [ERA5 via Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api).

## Preparation

Extract the SCADA archives, including status logs, into `data/raw/<farm>/`. Preserve the CSV names and headers. Run from the repository root:

```bash
for farm in kelmarsh penmanshiel; do
    uv run --group data uq4wind-prepare "$farm" --raw "data/raw/$farm"
done
uv run uq4wind-backtest kelmarsh
uv run uq4wind-backtest penmanshiel --min-train-window 365D
TABPFN_API_KEY=... uv run --group tabpfn uq4wind-backtest kelmarsh --tabpfn
TABPFN_API_KEY=... uv run --group tabpfn uq4wind-backtest penmanshiel --tabpfn --min-train-window 365D
```

Preparation removes duplicates, unavailable records, logged stops/warnings/curtailment, wind speeds <=0.1 m/s, nonzero curtailment energy and setpoints below 99% of potential power. SCADA is averaged to 30 minutes. Farm capacity factor averages reporting turbine power divided by 2,050 kW, requiring at least half the farm. Missing targets remain NaN; values are not clipped.

Weather retrieval covers 2022-2024. GEFS uses 31 members and daily 00Z runs; forecasting applies a six-hour publication lag. ERA5 wind is scaled from 100 m to 80 m by `(80/100)**(1/7)` and interpolated to 30 minutes for retrospective evaluation. TabPFN uses a metered API.

## File format

Files belong in `data/processed/`, prefixed with `kelmarsh_` or `penmanshiel_`. Timestamps are UTC; 30-minute indexes are sorted, unique and regular.

| Suffix | Index | Columns |
|---|---|---|
| `capacity_factor_30min.parquet` | DatetimeIndex | `capacity_factor` |
| `gefs_grid.parquet` | `init_time`, `valid_time`, `ensemble_member` | GEFS columns below |
| `era5_wind_30min.parquet` | DatetimeIndex | `wind_speed_realised` (m/s) |
| `forecasts.parquet` | Row index | Forecast columns below |
| `tabpfn_forecasts.parquet` | Row index | Same; `model = tabpfn` |

GEFS columns: `wind_u_80m`, `wind_v_80m`, `wind_speed_80m` (m/s), `temperature_80m` (C), `pressure_80m` (Pa), `precipitation_surface`, `nwp_lead_hours`. The index is a MultiIndex; members are numbered 0-30.

Forecast columns: `farm`, `model`, `issue_time`, `horizon`, `y_true`, `y_pred`, `y_lower`, `y_upper`. Horizons are 1, 6 or 48 half-hours; targets and predictions are capacity factors, with ordered endpoints. Each `(farm, model, issue_time, horizon)` is unique. `y_true` equals capacity factor at `issue_time + horizon * 30 minutes`; issuance is at the end of the period labelled by `issue_time`.

Model names: `persistence`, `arima`, `lgbm_qr`, `bootstrap`, `gefs_power_curve`, `tabpfn`. The notebooks compare all six on common issue-time/horizon pairs.
