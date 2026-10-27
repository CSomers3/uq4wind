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
    # Native versus conformalised bands

    Six forecasters each publish their own 90% band. We wrap every band in seven conformal
    rules, at three horizons and two farms, and score native against wrapped on the same rows.

    Writes the coverage-width and point-accuracy figures and both full scorecards to `results/`,
    and prints the numbers quoted in the paper. Reads only `data/processed/`; no network.
    """)
    return


@app.cell
def _():
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from uq4wind import evaluation as E
    from uq4wind import forecasting as F

    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent
    RESULTS = ROOT / "results"
    RESULTS.mkdir(exist_ok=True)
    plt.style.use(["default", str(ROOT / "notebooks" / "matplotlibrc")])
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-identical figure PDFs across runs

    WIDTH = 5.5  # NeurIPS text width, inches
    MARKERS = dict(zip(E.SUBJECTS, ("o", "s", "^", "D", "v", "X")))
    return E, F, Line2D, MARKERS, MaxNLocator, RESULTS, WIDTH, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Conformalise

    `evaluate` wraps every (farm, model, horizon) stream in each rule, holding each score back
    until its settlement period has closed. The scorecard has one row per
    (farm, model, horizon, method) cell: coverage, mean width and mean Winkler score.
    About 3 minutes.
    """)
    return


@app.cell
def _(E, pd):
    evaluated = pd.concat([E.evaluate(farm) for farm in E.FARM_NAMES], ignore_index=True)
    scores = E.scorecard(evaluated)
    scores
    return evaluated, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Figure 1: the width cost of restoring coverage

    Each marker is one forecaster; the grey segment runs from its native band (light) to its
    OSSCP band (dark).
    """)
    return


@app.cell
def _(E, F, Line2D, MARKERS, MaxNLocator, RESULTS, WIDTH, plt, scores):
    _NATIVE, _OSSCP = "#9ecae1", "#08519c"
    _cells = scores.set_index(["farm", "horizon", "method", "model"])

    figure1, _axes = plt.subplots(2, len(F.HORIZONS), figsize=(WIDTH, 3.8), sharex=True)
    for _row, _farm in zip(_axes, E.FARM_NAMES):
        for _ax, _h in zip(_row, F.HORIZONS):
            for _model in E.SUBJECTS:
                _native = _cells.loc[(_farm, _h, "native", _model)]
                _osscp = _cells.loc[(_farm, _h, "osscp", _model)]
                _ax.plot([_native["coverage"], _osscp["coverage"]], [_native["width"], _osscp["width"]],
                         color="0.5", lw=0.6, alpha=0.6, zorder=1)
                for _cell, _colour in ((_native, _NATIVE), (_osscp, _OSSCP)):
                    _ax.plot(_cell["coverage"], _cell["width"], marker=MARKERS[_model], color=_colour,
                             mec="0.35", mew=0.5, ms=5.0, ls="none", zorder=2)
            _ax.axvline(0.9, color="0.35", lw=0.7, ls=(0, (4, 3)), zorder=0)
            _ax.margins(y=0.14)
            _ax.yaxis.set_major_locator(MaxNLocator(4))
        _row[0].set_ylabel(_farm.capitalize())
    for _ax, _h in zip(_axes[0], F.HORIZONS):
        _ax.set_title(E.HORIZON_LABELS[_h], pad=3)
    _axes[0, 0].set_xlim(0.78, 0.95)
    _axes[0, 0].xaxis.set_major_locator(MaxNLocator(4))
    _axes[0, 0].annotate("90%", xy=(0.9, 0.55), xycoords=("data", "axes fraction"), xytext=(3, 0),
                         textcoords="offset points", ha="left", va="center", fontsize=6, color="0.35")

    figure1.tight_layout(rect=(0.035, 0.045, 1, 0.88))
    figure1.supylabel("Mean interval width", x=0.005, fontsize=8)
    figure1.supxlabel("Coverage", y=0.015, fontsize=8)
    figure1.legend(
        handles=[Line2D([], [], color="0.35", marker=MARKERS[m], ms=5.0, ls="none", label=E.SUBJECTS[m])
                 for m in E.SUBJECTS]
        + [Line2D([], [], color=_NATIVE, lw=4, label="Native band"),
           Line2D([], [], color=_OSSCP, lw=4, label="OSSCP band")],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, alignment="left",
    )
    figure1.savefig(RESULTS / "figure1_coverage_width.pdf")
    figure1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Full scorecards
    """)
    return


@app.cell
def _(E, RESULTS, mo, scores):
    _tables = {}
    for _farm in E.FARM_NAMES:
        _table = E.labelled(scores[scores["farm"] == _farm]).pivot_table(
            index=["Subject", "Method"], columns="horizon", values=["coverage", "width", "winkler"],
            observed=True, sort=False,
        ).reindex(columns=list(E.HORIZON_LABELS.values()), level=1)
        E.write_latex(_table, RESULTS / f"table_scorecard_{_farm}.tex")
        _tables[_farm.capitalize()] = _table.round(2)
    mo.ui.tabs(_tables)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Point accuracy

    RMSE of each forecaster's point forecast (its predictive mean) at each horizon.
    """)
    return


@app.cell
def _(E, F, Line2D, MARKERS, RESULTS, WIDTH, evaluated, plt):
    _scored = evaluated[evaluated["method"] == "osscp"]
    _rmse = _scored.groupby(["farm", "model", "horizon"]).apply(
        lambda g: ((g["y_true"] - g["y_pred"]) ** 2).mean() ** 0.5, include_groups=False
    )
    _colours = dict(zip(E.SUBJECTS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    figure4, _panels = plt.subplots(1, len(E.FARM_NAMES), figsize=(WIDTH, 2.5), sharey=True)
    for _panel, _farm in zip(_panels, E.FARM_NAMES):
        for _model in E.SUBJECTS:
            _panel.plot(range(len(F.HORIZONS)), [_rmse[(_farm, _model, _h)] for _h in F.HORIZONS],
                        marker=MARKERS[_model], color=_colours[_model], mec="0.35", mew=0.5, ms=5.0, lw=1.2)
        _panel.set_xticks(range(len(F.HORIZONS)), [E.HORIZON_LABELS[_h] for _h in F.HORIZONS])
        _panel.set_xlim(-0.3, len(F.HORIZONS) - 0.7)
        _panel.set_title(_farm.capitalize(), pad=3)
    _panels[0].set_ylabel("RMSE (capacity factor)")

    figure4.tight_layout(rect=(0, 0.05, 1, 0.82), w_pad=1.5)
    figure4.supxlabel("Forecast horizon", y=0.015, fontsize=8)
    figure4.legend(
        handles=[Line2D([], [], color=_colours[m], marker=MARKERS[m], mec="0.35", mew=0.5, ms=5.0, lw=1.2,
                        label=E.SUBJECTS[m]) for m in E.SUBJECTS],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
    )
    figure4.savefig(RESULTS / "figure4_point_accuracy.pdf")
    figure4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Numbers quoted in the paper
    """)
    return


@app.cell
def _(E, evaluated, pd, scores):
    def quoted_numbers():
        lines = []
        card = scores.pivot_table(index=["farm", "model", "horizon"], columns="method",
                                  values=["coverage", "width", "winkler"])
        coverage, width, winkler = card["coverage"], card["width"], card["winkler"]

        for farm, rows in evaluated.groupby("farm"):
            n = rows.groupby(["model", "horizon", "method"]).size()
            lines.append(f"{farm}: scored {rows['issue_time'].min():%d-%m-%Y} to "
                         f"{rows['issue_time'].max():%d-%m-%Y}, N = {n.min():,}-{n.max():,} per cell")

        lines.append(f"native coverage {coverage['native'].min():.3f}-{coverage['native'].max():.3f}, "
                     f"lowest per model {coverage['native'].groupby('model').min().round(3).to_dict()}")
        lines.append(f"OSSCP coverage {coverage['osscp'].min():.3f}-{coverage['osscp'].max():.3f}")
        width_change = 100 * (width["osscp"] / width["native"] - 1)
        gain = 100 * (coverage["osscp"] - coverage["native"])
        lines.append("OSSCP over native at 24 h, % width change [coverage gain in points]\n"
                     + (width_change.round(1).astype(str) + " [" + gain.round(1).astype(str) + "]")
                     .xs(48, level="horizon").unstack("farm").to_string())
        lines.append(f"largest narrowing, any horizon: {width_change.min():.1f}%")
        worse = 100 * (winkler["osscp"] / winkler["native"] - 1)
        lines.append(f"Winkler improves in {(worse < 0).sum()} of {len(worse)} cells; the rest worsen by "
                     f"at most {worse.max():.2f}%: {worse[worse > 0].round(2).to_dict()}")

        best_gain = 100 * (1 - winkler[list(E.ADAPTIVE_METHODS)].min(axis=1) / winkler["osscp"])
        lines.append("best adaptive rule's Winkler gain over OSSCP, %: "
                     + ", ".join(f"{E.HORIZON_LABELS[h]} mean {g.mean():.1f} [max {g.max():.1f}]"
                                 for h, g in best_gain.groupby("horizon")))

        osscp = evaluated[evaluated["method"] == "osscp"].groupby(["farm", "model", "horizon"])
        ci = osscp.apply(lambda g: pd.Series(E.coverage_interval(g), index=["est", "low", "high"]),
                         include_groups=False)
        contain = ((ci["low"] <= 0.9) & (ci["high"] >= 0.9)).sum()
        lines.append(f"OSSCP coverage 95% interval contains 0.90 in {contain} of {len(ci)} cells")
        return "\n".join(lines)

    print(quoted_numbers())
    return


if __name__ == "__main__":
    app.run()
