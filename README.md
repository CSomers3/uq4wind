# uq4wind

Code for *The Cost of Coverage: Conformalising Native Forecast Uncertainty for GB Wind Power*
(AI for Power Grids, NeurIPS 2026).

## Installation

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/CSomers3/uq4wind.git
cd uq4wind
uv sync --group notebooks
```

## Usage

Prepare the inputs following [data/README.md](data/README.md), then run:

```bash
cd notebooks
uv run python 01_scorecard.py
uv run python 02_regime.py
```

The notebooks write figures and tables to `results/`.
