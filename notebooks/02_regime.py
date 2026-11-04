import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Coverage and missed energy by wind regime

    Marginal coverage can hide where a band fails. Here the day-ahead (24 h) bands are split by
    wind regime: the regime GEFS forecast at issue time, which Mondrian calibrates within, and the
    regime ERA5 says was realised, which no calibration could see.

    Writes the regime figures and tables to `results/`, and prints the numbers quoted in the paper.
    Reads only `data/processed/`; no network.
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D

    from uq4wind import data as D
    from uq4wind import evaluation as E

    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
    RESULTS = ROOT / "results"
    RESULTS.mkdir(exist_ok=True)
    plt.style.use(["default", str(ROOT / "notebooks" / "matplotlibrc")])
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical figure PDFs across runs

    WIDTH = 5.5  # NeurIPS text width, inches
    METHODS = ("native", "osscp", "mondrian")
    HIGH = E.WIND_LABELS[-1]
    return D, E, HIGH, Line2D, METHODS, RESULTS, WIDTH, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Conformalise, then keep the day-ahead rows

    Each row carries `band` (forecast regime) and `band_realised` (ERA5 regime). A miss is
    measured as the energy outside the band, $0.5\,\mathrm{h} \times$ farm capacity $\times$ the
    distance in capacity factor, and `long` is the part above the upper bound (over-production).
    About 3 minutes.
    """)
    return


@app.cell
def _(D, E, METHODS, pd):
    _evaluated = pd.concat([E.evaluate(farm) for farm in E.FARM_NAMES], ignore_index=True)
    _rows = _evaluated[(_evaluated["horizon"] == E.LEDGER_HORIZON) & _evaluated["method"].isin(METHODS)]
    _mwh = _rows["farm"].map({f: farm.capacity_mw for f, farm in D.FARMS.items()}) * 0.5
    day_ahead = _rows.assign(
        hit=(_rows["y_true"] >= _rows["y_lower"]) & (_rows["y_true"] <= _rows["y_upper"]),
        width=_rows["y_upper"] - _rows["y_lower"],
        miss=((_rows["y_lower"] - _rows["y_true"]).clip(lower=0)
              + (_rows["y_true"] - _rows["y_upper"]).clip(lower=0)) * _mwh,
        long=(_rows["y_true"] - _rows["y_upper"]).clip(lower=0) * _mwh,
        regime_right=_rows["band"].astype(str) == _rows["band_realised"].astype(str),
    )
    day_ahead.groupby(["farm", "method"]).size().unstack()
    return (day_ahead,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Coverage by forecast regime

    Pooled calibration under-covers in high wind; Mondrian, calibrating within the forecast
    regime, brings each regime back towards 0.90.
    """)
    return


@app.cell
def _(E, METHODS, RESULTS, day_ahead):
    _ledger = E.wind_ledger(day_ahead)
    table4 = (
        E.labelled(_ledger, METHODS)
        .pivot_table(index=["Subject", "Method"], columns=["bucket", "farm"], values="coverage",
                     observed=True, sort=False)
        .reindex(columns=list(E.WIND_LABELS), level=0)
        .reindex(columns=list(E.FARM_NAMES), level=1)
    )
    E.write_latex(table4, RESULTS / "table_coverage_by_band.tex")
    table4.round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Coverage by realised regime

    Both farms and the six forecasters pooled, split by whether the forecast named the regime that
    occurred. Every band uses the same periods: when the forecast was wrong, high-wind coverage
    collapses for all of them.
    """)
    return


@app.cell
def _(E, RESULTS, WIDTH, day_ahead, plt):
    _LINES = {"native": ("Native", "#9ecae1", "#5b9cc8"), "osscp": ("OSSCP", "#08519c", "#08519c"),
              "mondrian": ("Mondrian", "#d94801", "#d94801")}  # name, line colour, label colour
    _STYLE = dict(marker="o", ms=4.5, mec="0.35", mew=0.5, lw=1.2, solid_capstyle="round")
    _coverage = day_ahead.groupby(["regime_right", "band_realised", "method"], observed=True)["hit"].mean()

    figure_regime, _panels = plt.subplots(1, 2, figsize=(WIDTH, 1.6), sharey=True)
    for _panel, _right in zip(_panels, (True, False)):
        _panel.axhline(0.9, color="0.55", lw=0.6, ls=(0, (4, 3)), zorder=0)
        for _m, (_name, _c, _text) in _LINES.items():
            _y = _coverage[_right].xs(_m, level="method").reindex(E.WIND_LABELS).to_numpy()
            _panel.plot(range(3), _y, color=_c, mfc=_c, zorder=3 if _m == "mondrian" else 2, **_STYLE)
            if _right:
                _panel.annotate(_name, (2, _y[-1]), xytext=(7, 0), textcoords="offset points", va="center",
                                fontsize=7, color=_text, bbox=dict(fc="white", ec="none", pad=0.8))
        _panel.set_xticks(range(3), [_l.replace(" (", "\n").replace(")", "").replace("-", "–") for _l in E.WIND_LABELS])
        _panel.tick_params(axis="x", length=0, pad=3)
        _panel.spines["bottom"].set_color("0.6")
        _panel.set_xlim(-0.3, 2.3)
        _panel.set_title(f"Regime forecast {'right' if _right else 'wrong'}", pad=4)
    _panels[1].annotate("90%", (2.3, 0.9), xytext=(0, 2), textcoords="offset points", ha="right", va="bottom",
                        fontsize=6.5, color="0.5")
    _panels[0].set_ylim(0.5, 1.0)
    _panels[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5", "", "0.7", "", "0.9", ""])
    _panels[0].set_ylabel("Coverage at 24 h")
    figure_regime.tight_layout(w_pad=3.0)
    figure_regime.savefig(RESULTS / "figure2_realised_regime.pdf")
    figure_regime
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Coverage and width within each forecast regime

    OSSCP to Mondrian, both farms pooled: over all periods, then within each forecast regime.
    Pooled coverage sits on 0.90 and hides the regimes either side of it.
    """)
    return


@app.cell
def _(E, Line2D, RESULTS, WIDTH, day_ahead, plt):
    from matplotlib.ticker import MaxNLocator

    _MARKERS = dict(zip(E.SUBJECTS, ("o", "s", "^", "D", "v", "X")))
    _COLOURS = {"osscp": "#08519c", "mondrian": "#d94801"}
    _MARK = dict(mec="0.35", mew=0.5, ms=5.0, ls="none")
    _rows = day_ahead[day_ahead["method"].isin(_COLOURS)]
    _share = _rows.loc[_rows["method"] == "osscp", "band"].value_counts(normalize=True)
    _groups = {"All periods": _rows}
    _groups.update({f"{_b.split()[0]} wind, {_share[_b]:.0%}": _rows[_rows["band"] == _b] for _b in E.WIND_LABELS})

    figure_wind, _panels = plt.subplots(1, len(_groups), figsize=(WIDTH, 1.9), sharex=True)
    for _panel, (_title, _frame) in zip(_panels, _groups.items()):
        _stats = _frame.groupby(["model", "method"])[["hit", "width"]].mean()
        for _model in E.SUBJECTS:
            _a, _b = _stats.loc[(_model, "osscp")], _stats.loc[(_model, "mondrian")]
            _panel.plot([_a["hit"], _b["hit"]], [_a["width"], _b["width"]], color="0.5", lw=0.6, alpha=0.6, zorder=1)
            for _method, _r in (("osscp", _a), ("mondrian", _b)):
                _panel.plot(_r["hit"], _r["width"], marker=_MARKERS[_model], color=_COLOURS[_method], zorder=2, **_MARK)
        _panel.axvline(0.9, color="0.35", lw=0.7, ls=(0, (4, 3)), zorder=0)
        _panel.margins(y=0.16)
        _panel.yaxis.set_major_locator(MaxNLocator(4))
        _panel.set_title(_title, pad=3)
    _panels[0].set_xlim(0.55, 1.0)
    _panels[0].set_xticks([0.6, 0.75, 0.9])
    figure_wind.tight_layout(rect=(0.03, 0.06, 1, 0.9), w_pad=0.6)
    figure_wind.supylabel("Mean interval width", x=0.005, fontsize=8)
    figure_wind.supxlabel("Coverage at 24 h", y=0.015, fontsize=8)
    figure_wind.legend(
        handles=[Line2D([], [], color=_COLOURS["osscp"], lw=4, label="OSSCP band"),
                 Line2D([], [], color=_COLOURS["mondrian"], lw=4, label="Mondrian band")],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
    )
    figure_wind.savefig(RESULTS / "figure_width_by_wind.pdf")
    figure_wind
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Missed energy by realised regime

    *Periods*: share of forecasts. *MWh*: energy outside the band per forecast. *Energy share*:
    share of the method's missed energy at that farm. *Long share*: fraction above the upper
    bound. Compared only within a farm: the farms differ in capacity and year.
    """)
    return


@app.cell
def _(E, METHODS, RESULTS, day_ahead, pd):
    _by_regime = day_ahead.groupby(["farm", "method", "band_realised"], observed=True)
    _by_method = day_ahead.groupby(["farm", "method"])
    missed_energy = pd.DataFrame({
        "Periods": _by_regime.size() / _by_method.size(),
        "MWh": _by_regime["miss"].mean(),
        "Energy share": _by_regime["miss"].sum() / _by_method["miss"].sum(),
        "Long share": _by_regime["long"].sum() / _by_regime["miss"].sum(),
    })
    _table = missed_energy.unstack("band_realised").reindex(
        pd.MultiIndex.from_product([E.FARM_NAMES, METHODS])
    )
    _table.index = _table.index.set_names(["Farm", "Method"]).set_levels(
        [[f.capitalize() for f in _table.index.levels[0]], [E.METHODS[m] for m in _table.index.levels[1]]]
    )
    _table = _table.rename(columns=lambda b: b.split()[0], level="band_realised")
    E.write_latex(_table, RESULTS / "table_missed_energy_by_band.tex")
    _table.round(2)
    return (missed_energy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Numbers quoted in the paper
    """)
    return


@app.cell
def _(HIGH, day_ahead, missed_energy):
    def quoted_numbers():
        osscp = missed_energy.xs("osscp", level="method").xs(HIGH, level="band_realised")
        mondrian = missed_energy.xs("mondrian", level="method").xs(HIGH, level="band_realised")
        means = day_ahead.groupby(["farm", "method"])[["miss", "width"]].mean().unstack("method")
        mondrian_rows = day_ahead[day_ahead["method"] == "mondrian"]
        realised = mondrian_rows.groupby(["farm", "regime_right", "band_realised"], observed=True)["hit"].mean()
        same_rows = day_ahead[day_ahead["method"].isin(["osscp", "mondrian"])]
        same_rows = same_rows.groupby(["farm", "band_realised", "regime_right", "method"], observed=True)["hit"].mean()
        periods = mondrian_rows.groupby(["farm", "band_realised", "regime_right"], observed=True)["issue_time"].nunique()
        return "\n".join([
            "24 h coverage by realised regime, right/wrong regime forecast:",
            same_rows.unstack("method").round(2).to_string(),
            "unique settlement periods per right/wrong group:",
            periods.unstack("regime_right").to_string(),
            "24 h, realised high wind, OSSCP:",
            f"  share of periods {osscp['Periods'].round(3).to_dict()}",
            f"  share of missed energy {osscp['Energy share'].round(3).to_dict()}",
            f"  over-production share {osscp['Long share'].round(3).to_dict()}",
            f"Mondrian cuts high-wind missed energy by "
            f"{(100 * (1 - mondrian['MWh'] / osscp['MWh'])).round(1).to_dict()} %",
            f"Mondrian/OSSCP mean width over all periods "
            f"{(means['width']['mondrian'] / means['width']['osscp']).round(3).to_dict()}",
            f"lowest Mondrian coverage by realised regime: {realised.min():.2f} at {realised.idxmin()}",
        ])

    print(quoted_numbers())
    return


if __name__ == "__main__":
    app.run()
