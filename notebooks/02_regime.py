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

    Pooled over the six forecasters. Mondrian is split by whether the forecast named the regime
    that occurred: when it did not, high-wind coverage collapses.
    """)
    return


@app.cell
def _(E, Line2D, RESULTS, WIDTH, day_ahead, plt):
    _MONDRIAN = "#d94801"
    _method = day_ahead["method"]
    _lines = {
        "Native band": (_method == "native", dict(color="#9ecae1", ls="-", mfc="#9ecae1")),
        "OSSCP band": (_method == "osscp", dict(color="#08519c", ls="-", mfc="#08519c")),
        "Mondrian, regime right": ((_method == "mondrian") & day_ahead["regime_right"],
                                   dict(color=_MONDRIAN, ls="-", mfc=_MONDRIAN)),
        "Mondrian, regime wrong": ((_method == "mondrian") & ~day_ahead["regime_right"],
                                   dict(color=_MONDRIAN, ls=(0, (3, 2)), mfc="white")),
    }
    _STYLE = dict(marker="o", ms=4.5, mec="0.35", mew=0.5, lw=1.0)

    figure_regime, _panels = plt.subplots(1, len(E.FARM_NAMES), figsize=(WIDTH, 1.65), sharey=True)
    for _panel, _farm in zip(_panels, E.FARM_NAMES):
        for _mask, _style in _lines.values():
            _rows = day_ahead[_mask & (day_ahead["farm"] == _farm)]
            _coverage = _rows.groupby("band_realised", observed=True)["hit"].mean().reindex(E.WIND_LABELS)
            _panel.plot(range(3), _coverage.to_numpy(), **_STYLE, **_style)
        _panel.axhline(0.9, color="0.35", lw=0.7, ls=(0, (4, 3)), zorder=0)
        _panel.set_xticks(range(3), [_l.replace(" (", "\n(").replace("-", "–") for _l in E.WIND_LABELS])
        _panel.set_xlim(-0.3, 2.3)
        _panel.set_title(_farm.capitalize(), pad=3)
    _panels[0].set_ylabel("Coverage at 24 h")
    _panels[0].set_ylim(0.45, 1.0)
    figure_regime.tight_layout(rect=(0, 0, 1, 0.86), w_pad=1.5)
    figure_regime.legend(handles=[Line2D([], [], label=k, **_STYLE, **s) for k, (_, s) in _lines.items()],
                   loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
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
        return "\n".join([
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
