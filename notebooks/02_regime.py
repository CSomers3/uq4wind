import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Conditional coverage, and what it costs

        Coverage split by wind band, and the balancing price the misses settle at.

        Produces Table 4 and Figure 2. Downloads GB imbalance prices from Elexon's public
        BMRS API on first run and caches them.
        """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent

    from uq4wind import conformal as C
    from uq4wind import data as D
    from uq4wind import evaluation as E
    from uq4wind.data import PROCESSED

    plt.style.use(["default", str(ROOT / "notebooks" / "matplotlibrc")])
    RESULTS = ROOT / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)

    PRICES = PROCESSED / "gb_imbalance_prices.parquet"
    HORIZON = E.LEDGER_HORIZON
    WIDTH = 5.5
    return (
        
        C,
        D,
        E,
        HORIZON,
        Line2D,
        MaxNLocator,
        PRICES,
        RESULTS,
        WIDTH,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Conformalise""")
    return


@app.cell
def _(E, mo, pd):
    evaluated = pd.concat([E.evaluate(farm) for farm in E.FARM_NAMES], ignore_index=True)
    mo.vstack(
        [
            mo.md("**Scored rows per farm and method**"),
            evaluated.pivot_table(
                index="farm", columns="method", values="y_true", aggfunc="size"
            ),
        ]
    )
    return (evaluated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Table 4

        Coverage by the wind band the forecast named, at 24 hours. Bands are fixed at 5 and
        11 m/s so a band means the same regime at both farms.
        """
    )
    return


@app.cell
def _(C, E, RESULTS, evaluated, mo):
    BANDED_METHODS = ("native", "osscp", "mondrian")

    _ledger = E.wind_ledger(evaluated[evaluated["horizon"] == E.LEDGER_HORIZON], C.ALPHA)
    band_table = (
        E.labelled(_ledger, BANDED_METHODS)
        .pivot_table(
            index=["Subject", "Method"],
            columns=["bucket", "farm"],
            values="coverage",
            observed=True,
            sort=False,
        )
        .reindex(columns=list(E.WIND_LABELS), level=0)
        .reindex(columns=list(E.FARM_NAMES), level=1)
    )
    band_table.round(3).to_csv(RESULTS / "table_coverage_by_band.csv")

    mo.vstack([mo.md("**Coverage by forecast wind band, 24 h**"), band_table.round(3)])
    return (band_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Prices

        Elexon's settlement system prices, one pair per half-hourly period. Since P305 took
        effect in 2015 Great Britain has cashed out every imbalance at a single price, so the
        buy and sell price are the same number. The notebook checks that rather than assuming it.
        """
    )
    return


@app.cell
def _(PRICES, evaluated, mo, pd):
    import json
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    ENDPOINT = (
        "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{}?format=json"
    )

    def fetch(day):
        for attempt in range(3):
            try:
                with urllib.request.urlopen(ENDPOINT.format(day), timeout=60) as response:
                    return json.load(response)["data"]
            except Exception:
                if attempt == 2:
                    raise
        return []

    def download(first, last):
        days = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d")
        with ThreadPoolExecutor(8) as pool:
            rows = [row for part in pool.map(fetch, days) for row in part]
        frame = pd.DataFrame(rows).assign(
            start_time=lambda d: pd.to_datetime(d["startTime"], utc=True)
        )
        frame = frame.sort_values("createdDateTime").drop_duplicates("start_time", keep="last")
        frame = frame.set_index("start_time").sort_index()[["systemSellPrice", "systemBuyPrice"]]
        return frame.astype(float).rename(
            columns={"systemSellPrice": "ssp", "systemBuyPrice": "sbp"}
        )

    target_times = evaluated["issue_time"] + evaluated["horizon"] * pd.Timedelta("30min")
    if not PRICES.exists():
        download(target_times.min().date(), target_times.max().date()).to_parquet(PRICES)

    cash_out = pd.read_parquet(PRICES)
    prices = cash_out["sbp"]

    mo.md(
        f"**{len(cash_out):,} settlement periods**, "
        f"{cash_out.index.min():%Y-%m-%d} to {cash_out.index.max():%Y-%m-%d}. "
        f"Buy price equals sell price on every period: "
        f"`{bool((cash_out['ssp'] == cash_out['sbp']).all())}`."
    )
    return prices, target_times


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Figure 2

        Coverage scored on the band the weather delivered rather than the one the forecast
        named, pooled over the six subjects. The two Mondrian lines share a rule and a buffer
        and differ only in whether the forecast named the band correctly, so the gap between
        them is the taxonomy's error rather than the calibrator's.
        """
    )
    return


@app.cell
def _(E, HORIZON, evaluated, np, prices, target_times):
    BANDS = list(E.WIND_LABELS)
    AT = np.arange(len(BANDS))

    scored = evaluated.assign(
        price=target_times.map(prices),
        covered=(evaluated["y_true"] >= evaluated["y_lower"])
        & (evaluated["y_true"] <= evaluated["y_upper"]),
        banded=np.where(evaluated["band"] == evaluated["band_realised"], "correct", "wrong"),
    )
    scored = scored[scored["horizon"] == HORIZON]

    # BMRS occasionally publishes no price for a settlement period. Those rows are dropped
    # from the priced view rather than failing the run.
    unpriced = int(scored["price"].isna().sum())
    scored = scored[scored["price"].notna()]

    def coverage_by_band(farm, method, banded=None):
        rows = scored[(scored["farm"] == farm) & (scored["method"] == method)]
        if banded is not None:
            rows = rows[rows["banded"] == banded]
        return (
            rows.groupby("band_realised", observed=True)["covered"]
            .mean()
            .reindex(BANDS)
            .to_numpy()
        )

    def price_by_band(farm):
        rows = scored[scored["farm"] == farm].drop_duplicates("issue_time")
        return (
            rows.groupby("band_realised", observed=True)["price"].mean().reindex(BANDS).to_numpy()
        )

    return AT, BANDS, coverage_by_band, price_by_band, scored, unpriced


@app.cell
def _(AT, BANDS, E, Line2D, MaxNLocator, RESULTS, WIDTH, coverage_by_band, plt, price_by_band):
    NATIVE, OSSCP, MONDRIAN = "#9ecae1", "#08519c", "#DE8F05"
    TICKS = ("Low", "Average", "High")
    RANGES = ("0-5", "5-11", "11+")

    STYLES = {
        "native": (NATIVE, "-", NATIVE),
        "osscp": (OSSCP, "-", OSSCP),
        "correct": (MONDRIAN, "-", MONDRIAN),
        "wrong": (MONDRIAN, (0, (3, 2)), "white"),
    }

    def line(panel, values, key):
        colour, style, fill = STYLES[key]
        panel.plot(
            AT, values, color=colour, ls=style, lw=1.1, marker="o",
            ms=5.0, mec="0.35", mew=0.5, mfc=fill, zorder=3,
        )

    figure, panels = plt.subplots(1, len(E.FARM_NAMES), figsize=(WIDTH, 2.5))
    for panel, farm in zip(panels, E.FARM_NAMES):
        correct = coverage_by_band(farm, "mondrian", "correct")
        wrong = coverage_by_band(farm, "mondrian", "wrong")

        panel.fill_between(
            AT, wrong, correct, where=wrong < correct, interpolate=True,
            color=MONDRIAN, alpha=0.13, lw=0, zorder=0,
        )
        line(panel, coverage_by_band(farm, "native"), "native")
        line(panel, coverage_by_band(farm, "osscp"), "osscp")
        line(panel, correct, "correct")
        line(panel, wrong, "wrong")
        panel.axhline(0.90, color="0.35", lw=0.7, ls=(0, (4, 3)), zorder=1)

        panel.set_ylim(0.45, 1.0)
        panel.yaxis.set_major_locator(MaxNLocator(4))
        panel.set_xlim(-0.35, len(BANDS) - 0.65)
        panel.set_xticks(
            AT,
            [
                f"{name} ({span})\n\N{POUND SIGN}{price:.0f}"
                for name, span, price in zip(TICKS, RANGES, price_by_band(farm))
            ],
        )
        panel.set_title(farm.capitalize(), pad=3)

    panels[0].set_ylabel("Coverage")
    panels[1].set_yticklabels([])
    panels[0].annotate(
        "90%", xy=(AT[0] - 0.28, 0.90), xytext=(0, 3),
        textcoords="offset points", ha="left", fontsize=6, color="0.35",
    )

    figure.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.22, wspace=0.06)
    figure.supxlabel(
        "Realised wind band (m/s),\n avg balancing price (\N{POUND SIGN}/MWh)", y=0.0, fontsize=8
    )
    figure.legend(
        handles=[
            Line2D([], [], color=NATIVE, marker="o", ms=5.0, mec="0.35", mew=0.5, label="Native band"),
            Line2D([], [], color=OSSCP, marker="o", ms=5.0, mec="0.35", mew=0.5, label="OSSCP band"),
            Line2D([], [], color=MONDRIAN, marker="o", ms=5.0, mec="0.35", mew=0.5,
                   label="Mondrian, band correct"),
            Line2D([], [], color=MONDRIAN, marker="o", ms=5.0, mec="0.35", mew=0.5,
                   mfc="white", ls=(0, (3, 2)), label="Mondrian, band wrong"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, alignment="left",
    )
    figure.savefig(RESULTS / "figure2_regime_audit.pdf")
    figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Summary""")
    return


@app.cell
def _(BANDS, E, RESULTS, band_table, coverage_by_band, price_by_band, unpriced):
    def summarise():
        lines = []
        for farm in E.FARM_NAMES:
            for method in ("native", "osscp", "mondrian"):
                values = coverage_by_band(farm, method)
                shown = "  ".join(f"{b.split(' ')[0]} {v:.3f}" for b, v in zip(BANDS, values))
                lines.append(f"{farm:12s} {method:10s} {shown}")
            prices = "  ".join(
                f"{b.split(' ')[0]} \N{POUND SIGN}{p:.0f}"
                for b, p in zip(BANDS, price_by_band(farm))
            )
            lines.append(f"{farm:12s} {'price':10s} {prices}")
        if unpriced:
            lines.append(f"{unpriced} rows had no published price and were dropped")
        lines.append(f"table 4: {band_table.shape[0]} rows written to {RESULTS}")
        return "\n".join(lines)

    print(summarise())
    return


if __name__ == "__main__":
    app.run()
