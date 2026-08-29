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
        # Native versus conformalised bands

        Six forecasters, each with its own uncertainty band, wrapped by seven conformal rules
        at three horizons and two farms.

        Produces Figure 1, Tables 2, 3 and 5, and the wrapper-choice numbers quoted in
        Section 3. Needs no network.
        """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parent

    from uq4wind import conformal as C
    from uq4wind import evaluation as E
    from uq4wind import forecasting as F

    plt.style.use(["default", str(ROOT / "notebooks" / "matplotlibrc")])
    RESULTS = ROOT / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)

    NOMINAL = 1 - C.ALPHA
    WIDTH = 5.5  # NeurIPS text width, inches
    return (
        
        C,
        E,
        F,
        Line2D,
        MaxNLocator,
        NOMINAL,
        RESULTS,
        WIDTH,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Conformalise""")
    return


@app.cell
def _(C, E, mo, pd):
    evaluated = {farm: E.evaluate(farm) for farm in E.FARM_NAMES}
    scores = {farm: E.scorecard(evaluated[farm], C.ALPHA) for farm in E.FARM_NAMES}

    mo.vstack(
        [
            mo.md("**Scored rows per farm**"),
            pd.DataFrame(
                {
                    farm: {"rows": len(frame), "methods": frame["method"].nunique()}
                    for farm, frame in evaluated.items()
                }
            ),
        ]
    )
    return evaluated, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Figure 1

        Each marker is one forecaster. The grey segment runs from its native band to its
        OSSCP-wrapped band. Wrapping converges coverage on the nominal 0.90.
        """
    )
    return


@app.cell
def _(E, F, Line2D, MaxNLocator, NOMINAL, RESULTS, WIDTH, plt, scores):
    COVERAGE_LIM = (0.78, 0.95)
    NATIVE_COLOUR, OSSCP_COLOUR = "#9ecae1", "#08519c"
    MARKERS = dict(zip(E.SUBJECTS, ("o", "s", "^", "D", "v", "X")))

    def panel(ax, farm, horizon):
        cell = scores[farm][scores[farm]["horizon"] == horizon]
        native = cell[cell["method"] == "native"].set_index("model")
        wrapped = cell[cell["method"] == "osscp"].set_index("model")
        for model in E.SUBJECTS:
            if model not in native.index or model not in wrapped.index:
                continue
            x0, y0 = native.loc[model, "coverage"], native.loc[model, "width"]
            x1, y1 = wrapped.loc[model, "coverage"], wrapped.loc[model, "width"]
            ax.plot([x0, x1], [y0, y1], color="0.5", lw=0.6, alpha=0.6, zorder=1)
            for x, y, colour in ((x0, y0, NATIVE_COLOUR), (x1, y1, OSSCP_COLOUR)):
                ax.plot(
                    x, y, marker=MARKERS[model], color=colour,
                    mec="0.35", mew=0.5, ms=5.0, ls="none", zorder=2,
                )
        ax.axvline(NOMINAL, color="0.35", lw=0.7, ls=(0, (4, 3)), zorder=0)
        ax.margins(y=0.14)
        ax.yaxis.set_major_locator(MaxNLocator(4))

    figure, axes = plt.subplots(2, len(F.HORIZONS), figsize=(WIDTH, 3.8), sharex=True)
    for row, farm in zip(axes, E.FARM_NAMES):
        for ax, horizon in zip(row, F.HORIZONS):
            panel(ax, farm, horizon)
        row[0].set_ylabel(farm.capitalize())
    for ax, horizon in zip(axes[0], F.HORIZONS):
        ax.set_title(E.HORIZON_LABELS[horizon], pad=3)

    axes[0, 0].set_xlim(*COVERAGE_LIM)
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(4))
    axes[0, 0].annotate(
        "90%", xy=(NOMINAL, 0.55), xycoords=("data", "axes fraction"),
        xytext=(3, 0), textcoords="offset points",
        ha="left", va="center", fontsize=6, color="0.35",
    )

    figure.tight_layout(rect=(0.035, 0.045, 1, 0.88))
    figure.supylabel("Mean interval width", x=0.005, fontsize=8)
    figure.supxlabel("Coverage", y=0.015, fontsize=8)
    figure.legend(
        handles=[
            Line2D([], [], color="0.35", marker=MARKERS[m], ms=5.0, ls="none", label=E.SUBJECTS[m])
            for m in E.SUBJECTS
        ]
        + [
            Line2D([], [], color=NATIVE_COLOUR, lw=4, label="Native band"),
            Line2D([], [], color=OSSCP_COLOUR, lw=4, label="OSSCP band"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, alignment="left",
    )
    figure.savefig(RESULTS / "figure1_coverage_width.pdf")
    figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Tables 2 and 3

        Coverage, mean width and Winkler score for every (subject, rule) pair at each horizon.
        """
    )
    return


@app.cell
def _(E, RESULTS, mo, scores):
    scorecards = {}
    for _farm in E.FARM_NAMES:
        table = E.labelled(scores[_farm]).pivot_table(
            index=["Subject", "Method"],
            columns="horizon",
            values=["coverage", "width", "winkler"],
            observed=True,
            sort=False,
        )
        table = table.reindex(columns=list(E.HORIZON_LABELS.values()), level=1)
        scorecards[_farm] = table
        table.round(3).to_csv(RESULTS / f"table_scorecard_{_farm}.csv")

    mo.vstack(
        [mo.md(f"**{_f.capitalize()}**") for _f in E.FARM_NAMES[:1]]
        + [scorecards[E.FARM_NAMES[0]].round(3)]
    )
    return (scorecards,)


@app.cell
def _(E, scorecards):
    scorecards[E.FARM_NAMES[1]].round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Wrapper choice

        What fixing the rule to plain OSSCP costs against the best adaptive rule for that
        (farm, subject, horizon), in Winkler score. Mondrian is excluded: it partitions rather
        than adapts, and native is the subject rather than a rule.
        """
    )
    return


@app.cell
def _(E, RESULTS, mo, pd, scores):
    _long = pd.concat(scores.values(), ignore_index=True)
    _long = _long[_long["model"].isin(E.SUBJECTS)]

    _osscp = _long[_long["method"] == "osscp"].set_index(["farm", "model", "horizon"])["winkler"]
    _best = (
        _long[_long["method"].isin(E.ADAPTIVE_METHODS)]
        .groupby(["farm", "model", "horizon"])["winkler"]
        .min()
    )

    excess = (100 * (_osscp / _best - 1)).rename("excess_pct").reset_index()
    wrapper_cost = (
        excess.groupby("horizon")["excess_pct"]
        .agg(mean="mean", worst="max")
        .rename(index=E.HORIZON_LABELS)
        .round(1)
    )
    wrapper_cost.to_csv(RESULTS / "wrapper_choice_cost.csv")

    mo.vstack(
        [
            mo.md("**Winkler cost of fixing the rule to OSSCP, % above the best adaptive rule**"),
            wrapper_cost,
        ]
    )
    return excess, wrapper_cost


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Table 5: point accuracy""")
    return


@app.cell
def _(E, RESULTS, mo, pd):
    _streams = {farm: E.load_stream(farm) for farm in E.FARM_NAMES}
    rmse = pd.DataFrame(
        {
            farm: stream.groupby(["model", "horizon"]).apply(
                lambda g: float(((g["y_true"] - g["y_pred"]) ** 2).mean() ** 0.5),
                include_groups=False,
            )
            for farm, stream in _streams.items()
        }
    )
    rmse = rmse.reindex(
        [(m, h) for m in E.SUBJECTS for h in E.HORIZON_LABELS]
    ).dropna(how="all")
    rmse.index = pd.MultiIndex.from_tuples(
        [(E.SUBJECTS[m], E.HORIZON_LABELS[h]) for m, h in rmse.index],
        names=["Subject", "Horizon"],
    )
    rmse.columns = [c.capitalize() for c in rmse.columns]
    rmse.round(4).to_csv(RESULTS / "table_rmse.csv")

    mo.vstack([mo.md("**Point-forecast RMSE, capacity-factor units**"), rmse.round(4)])
    return (rmse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Summary""")
    return


@app.cell
def _(E, RESULTS, evaluated, scores, wrapper_cost):
    def summarise():
        lines = []
        for farm in E.FARM_NAMES:
            cell = scores[farm]
            native = cell[cell["method"] == "native"]
            osscp = cell[cell["method"] == "osscp"]
            lines.append(
                f"{farm:12s} native coverage {native['coverage'].min():.3f}-"
                f"{native['coverage'].max():.3f}, "
                f"OSSCP {osscp['coverage'].min():.3f}-{osscp['coverage'].max():.3f} "
                f"({len(evaluated[farm]):,} rows)"
            )
        for horizon, row in wrapper_cost.iterrows():
            lines.append(
                f"OSSCP vs best adaptive at {horizon:7s}: "
                f"mean {row['mean']:.1f}%, worst {row['worst']:.1f}%"
            )
        lines.append(f"written to {RESULTS}")
        return "\n".join(lines)

    print(summarise())
    return


if __name__ == "__main__":
    app.run()
