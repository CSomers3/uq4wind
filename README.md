# uq4wind

Code for *The Cost of Coverage: Conformalising Native Forecast Uncertainty for GB Wind Power*
(AI for Power Grids workshop, NeurIPS 2026).

Six wind-power forecasters, seven conformal rules, two GB farms and three horizons.
We compare native and calibrated 90% prediction bands using coverage, width, Winkler score
and energy outside the bands.

## Run

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/CSomers3/uq4wind.git
cd uq4wind
uv sync --group notebooks
```

Follow [data/README.md](data/README.md) to source, clean and prepare the inputs. It documents
both the preparation commands and the required parquet formats. Data are generated locally.

Then run the notebooks to calculate the paper's figures and tables:

```bash
cd notebooks
uv run python 01_scorecard.py
uv run python 02_regime.py
```

Outputs are written to `results/`. To inspect a notebook interactively, run
`uv run marimo edit notebooks/01_scorecard.py` from the repository root.

## Code

| File | Purpose |
|---|---|
| `src/uq4wind/data.py` | SCADA cleaning, weather retrieval and time alignment |
| `src/uq4wind/forecasting.py` | Features, six forecasters and expanding-window backtests |
| `src/uq4wind/conformal.py` | Calibration rules and delayed-feedback loop |
| `src/uq4wind/evaluation.py` | Common sample, scoring and weekly block bootstrap |
| `notebooks/01_scorecard.py` | Native versus calibrated bands and point accuracy |
| `notebooks/02_regime.py` | Coverage and missed energy by wind regime |

Models refit every 30 days on expanding windows: two years initially at Kelmarsh and one at
Penmanshiel. Forecasts are issued hourly at 30-minute, 3-hour and 24-hour horizons. Calibration
uses a 1,000-score window and 10% warm-up, with outcomes released only after their target
period closes. Mondrian calibration uses forecast wind bands of 0-5, 5-11 and 11+ m/s.
