"""
evaluation_plots.py

Result figures of the evaluation of the transient removal filter (remove_noise): interactive (plotly) versions of the
figures of evaluation_denoise.ipynb (sections 5.4 and 6.4), built from the metrics saved by run_experiment
(results/experiments/experiment_{1, 2}.csv). Copied from the plotly cells of the notebook: a change in a figure of
the notebook must also be made here.

Each figure reproduces its matplotlib/seaborn version of the notebook pixel by pixel: the positions of the axes,
texts and legends are taken from the constrained layout of matplotlib applied to the same figure without data
(_<figure>_geometry, one per figure).

Experiment 1: insertion of a single artifact    -> plot_detection, plot_elimination, plot_conservation,
                                                   plot_amplitude, plot_factor_analysis
Experiment 2: insertion of several artifacts    -> plot_experiment_comparison, plot_interaction_artifacts,
              (+ idempotence)                      plot_mixture_performance, plot_conservation_multiple,
                                                   plot_idempotence

Entry point: save_figures(experiment) builds every figure of an experiment and saves it in
results/figures/experiment_{experiment}/{name}.{png, svg, html} (terminal version: denoising/save_figures.py).
"""

# ====================================================================================================================
# 1. Imports
# ====================================================================================================================

import colorsys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cbook import boxplot_stats
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, PowerNorm, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from seaborn.utils import relative_luminance

from .evaluation import FS, RESULTS_DIR, TOL, read_results
from .remove_noise import REF_INNER, REF_OUTER
from .summary_plots import (DASH_PATTERNS, FONT, MPL_COLORS, MPL_TICKFORMAT, mpl_colorscale, mpl_dashes, mpl_nticks,
                            mpl_range, mpl_text_size, rgba, rounded_rect)


# ====================================================================================================================
# 2. Constants
# ====================================================================================================================

# Paths
FIGURES_DIR = RESULTS_DIR.parent / "figures"  # denoising/results/figures (independent of the cwd)
CSV_FILES = {1: "experiment_1.csv", 2: "experiment_2.csv"}  # Results of each experiment (in RESULTS_DIR)

# Experiment parameters needed by the figures
SNR_STRONG = -20  # Experiment 2, unequal condition: SNR of the strong artifact (dB), as in run_experiment.py

# Saved images
PNG_SCALE = 5  # Scale factor of the .png images (size in px = PNG_SCALE x size of the figure)


# ====================================================================================================================
# 3. Helper functions
# ====================================================================================================================

def without_webgl(fig: go.Figure) -> go.Figure:
    """
    Copy of a plotly figure with its WebGL traces (Scattergl) as SVG traces (Scatter), for the .png images: in a
    large image (e.g. scale 5) kaleido does not draw the WebGL layer (panel A of plot_elimination and panel D of
    plot_conservation came out empty). The trace order is kept, so the traces that relied on WebGL to be drawn over
    the others (points below, medians / boxes above) keep their order.
    """
    data = fig.to_dict()
    for trace in data["data"]:
        if trace.get("type") == "scattergl":
            trace["type"] = "scatter"
    return go.Figure(data)


def save_figure(fig: go.Figure, path: Path, scale: float = PNG_SCALE) -> None:
    """
    Saves a plotly figure as <path>.png (scale x size of the figure, WebGL traces as SVG: see without_webgl),
    <path>.svg and <path>.html (plotly.js from the CDN). The folder is created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    without_webgl(fig).write_image(path.with_suffix(".png"), scale=scale)
    fig.write_image(path.with_suffix(".svg"))
    fig.write_html(path.with_suffix(".html"), include_plotlyjs="cdn")


# ====================================================================================================================
# 4. Main functions: experiment 1 (insertion of a single artifact)
# ====================================================================================================================

# --------------------------------------------------------------------------------------------------------------------
# 4.1. Detection (section 5.4.1 of evaluation_denoise.ipynb)
# --------------------------------------------------------------------------------------------------------------------

def _detection_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_detection figure as matplotlib computes it (layout="constrained"): builds the same figure
    WITHOUT data (titles, labels, limits, ticks, legends and colorbar) and returns the boxes (px from the top-left
    corner of the image cropped as bbox_inches="tight") of the axes, texts and legends.
    spec: {"A"|"B"|"C"|"D": dict(title, xlabel, ylabel, xlim, ylim, xticks, yticks, legend, loc), "cb": dict(...)}
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    axs = {"A": fig.add_subplot(gs[0, :])}
    axs["B"] = fig.add_subplot(gs[1, 0], sharey=axs["A"])
    axs["C"] = fig.add_subplot(gs[2, 0])
    axs["D"] = fig.add_subplot(gs[1:, 1])

    for k, ax in axs.items():
        s = spec[k]
        ax.set_title(s["title"], fontsize=14, fontweight="bold")
        ax.set_xlabel(s["xlabel"], fontsize=12)
        ax.set_ylabel(s["ylabel"], fontsize=12)
        if s.get("xticks"):
            ax.set_xticks(*s["xticks"])
        if s.get("yticks"):
            ax.set_yticks(*s["yticks"])
        ax.set_xlim(*s["xlim"])
        ax.set_ylim(*s["ylim"])
        ax.legend([Line2D([], [], marker="o") for _ in s["legend"]], s["legend"], loc=s["loc"], fontsize=10)
    fig.suptitle(spec["suptitle"], fontsize=16)

    c = spec["cb"]
    sm = ScalarMappable(cmap=ListedColormap(c["colors"]), norm=BoundaryNorm(np.arange(-0.5, len(c["colors"])),
                                                                           len(c["colors"])))
    cb = fig.colorbar(sm, ax=list(axs.values()), location="right", fraction=0.02, pad=0.01, aspect=60)
    cb.set_ticks(c["ticks"][0], labels=c["ticks"][1])
    cb.set_label(c["label"], fontsize=10)

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle))
    for k, ax in {**axs, "cb": cb.ax}.items():
        out[k] = dict(ax=box(ax), xlabel=box(ax.xaxis.label), ylabel=box(ax.yaxis.label), title=box(ax.title))
        if ax.get_legend():
            out[k]["legend"] = box(ax.get_legend())
            out[k]["legend_texts"] = [box(t) for t in ax.get_legend().get_texts()]
    return out


def plot_detection(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_detection (matplotlib/seaborn, section 5.4.1 of evaluation_denoise.ipynb): same
    layout (layout="constrained", figsize 12 x (9 + 0.12·n_art) in), sizes, fonts, colors and positions. The figure is
    cropped as the inline backend does (bbox_inches="tight"), so its size matches the matplotlib image.
    - A: detection rate vs SNR per artifact (+ global "All"), with its 95 % CI (bootstrap, as seaborn)
    - B: detection rate vs SNR per class of the clean signal (+ global)
    - C: false positives vs SNR (+ controls)
    - D: SNR at 50 % detection per artifact, with its 95 % CI (bootstrap over the clean signals)
    - Colorbar: color of each artifact (A, D)
    Interactivity: hover with the values (artifact, SNR, rate, CI...) in A, B, C, D and in the colorbar; click on the
    legend to hide/show each category (in B, line and band together); zoom in each panel (A and B share the y axis, as
    sharey).
    The positions of the axes and texts are taken from the constrained layout of matplotlib applied to the same figure
    without data (_detection_geometry).
    Limits: the legends are placed where loc="best" puts them with these data (A, B: upper right; C: upper left; D:
    lower right); the plotly legend symbols do not draw error bars; the plotly grid always stays below the bands; the
    bootstrap CIs use a fixed seed (seaborn does not), so they can differ slightly from the matplotlib version.

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 1 (read_results("experiment_1.csv")). It is not modified
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Data ===
    # copy (original not modified); BEFORE filtering: controls -> NaN
    results_df = results_df.assign(snr_db=pd.to_numeric(results_df["snr_db"], errors="coerce"))

    mix = results_df["art_id"] != "Control"
    mix_df = results_df[mix]

    art_order = sorted(mix_df["art_id"].unique(), key=lambda a: int(a))           # numeric order: 0, 1, 2, …, 39
    n_art = len(art_order)
    turbo = colormaps["turbo"](np.linspace(0, 1, n_art + 2)[1:-1])                 # = sns.color_palette("turbo", n)
    art_colors = {a: to_hex(c) for a, c in zip(art_order, turbo)}                  # same color in A, D and the colorbar
    snr_levels = np.sort(mix_df["snr_db"].unique())


    # --- SNR at 50 % (D) ---
    def snr_at_50(rate, levels):
        """
        SNR at which the detection rate crosses 0.5 (linear interpolation between levels). The detection decreases as
        the SNR increases: the last level with rate >= 0.5 is taken and interpolated up to the next one.
        +inf: >= 0.5 up to the last level (crossing above the range); -inf: < 0.5 in all of them (below the range).
        """
        above = rate >= 0.5
        if not above.any():
            return -np.inf
        i = np.where(above)[0].max()
        if i == len(rate) - 1:
            return np.inf
        r0, r1 = rate[i], rate[i + 1]
        return levels[i] + (0.5 - r0) * (levels[i + 1] - levels[i]) / (r1 - r0)


    # Hits and mixtures per (artifact, clean signal, SNR): arrays [art, clean, snr]
    clean_ids = sorted(mix_df["clean_id"].unique())
    g = mix_df.groupby(["art_id", "clean_id", "snr_db"])["detected"].agg(["sum", "count"])
    idx = pd.MultiIndex.from_product([art_order, clean_ids, snr_levels])
    S = g["sum"].reindex(idx, fill_value=0).to_numpy().reshape(len(art_order), len(clean_ids), len(snr_levels))
    Nn = g["count"].reindex(idx, fill_value=0).to_numpy().reshape(S.shape)

    # Bootstrap over the clean signals (independent unit): multinomial weights
    N_BOOT = 1000
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(clean_ids), np.full(len(clean_ids), 1 / len(clean_ids)), size=N_BOOT)   # (boot, clean)

    with np.errstate(invalid="ignore", divide="ignore"):
        rate_art = S.sum(1) / Nn.sum(1)                                            # (art, snr), point estimate
        rate_art_b = np.einsum("bc,acs->bas", W, S) / np.einsum("bc,acs->bas", W, Nn)   # (boot, art, snr)
        rate_all = S.sum((0, 1)) / Nn.sum((0, 1))                                  # global (as the black curve of A)
        rate_all_b = np.einsum("bc,acs->bs", W, S) / np.einsum("bc,acs->bs", W, Nn)

    snr50 = np.array([snr_at_50(r, snr_levels) for r in rate_art])
    snr50_b = np.array([[snr_at_50(r, snr_levels) for r in rb] for rb in rate_art_b])   # (boot, art)
    snr50_all = snr_at_50(rate_all, snr_levels)
    snr50_all_b = np.array([snr_at_50(r, snr_levels) for r in rate_all_b])

    # 95 % CI: the crossings out of range are clipped to the borders of the range to compute the percentiles
    lo_edge, hi_edge = snr_levels.min(), snr_levels.max()
    snr50_b_clip = np.clip(snr50_b, lo_edge, hi_edge)
    ci_lo, ci_hi = np.percentile(snr50_b_clip, [2.5, 97.5], axis=0)
    ci_all = np.percentile(np.clip(snr50_all_b, lo_edge, hi_edge), [2.5, 97.5])


    # --- Mean and 95 % CI (as seaborn) ---
    rng_ci = np.random.default_rng(1)

    def mean_ci(v):
        """Mean and bootstrap 95 % CI of the observations (errorbar=("ci", 95), n_boot=1000 of seaborn)"""
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        n = len(v)
        if np.isin(v, (0, 1)).all():                                               # binary: the resampled mean
            boots = rng_ci.binomial(n, v.mean(), N_BOOT) / n                       # follows a binomial (= bootstrap)
        else:
            boots = v[rng_ci.integers(0, n, (N_BOOT, n))].mean(1)
        return (v.mean(), *np.percentile(boots, [2.5, 97.5]))

    def ci_table(df, by, col):
        """Mean, lower CI and upper CI of col per group (index: by)"""
        gb = df.groupby(by)
        return pd.DataFrame([mean_ci(gr[col]) for _, gr in gb], index=pd.Index(gb.groups.keys()) if isinstance(by, str)
                            else pd.MultiIndex.from_tuples(gb.groups.keys()), columns=["mean", "lo", "hi"])

    # A: per artifact and global
    det_art = ci_table(mix_df, ["art_id", "snr_db"], "detected")
    det_all = ci_table(mix_df, "snr_db", "detected")

    # B: one row per clean signal and SNR (mean over artifacts and positions): the CI reflects the variation
    # between signals
    per_clean_lab = mix_df.groupby(["clean_id", "label", "snr_db"], as_index=False)["detected"].mean()
    label_order = sorted(per_clean_lab["label"].unique())
    label_colors = dict(zip(label_order, [to_hex(c) for c in colormaps["Set2"].colors]))   # = palette="Set2"
    det_lab = ci_table(per_clean_lab, ["label", "snr_db"], "detected")

    # C: one row per clean signal and SNR level (mean over artifacts and positions); controls: one per signal
    per_clean = mix_df.groupby(["clean_id", "snr_db"], as_index=False)["n_false"].mean()
    fp_mix = ci_table(per_clean, "snr_db", "n_false")
    fp_ctrl = mean_ci(results_df.loc[~mix, "n_false"])
    step = np.min(np.diff(snr_levels)) if len(snr_levels) > 1 else 5
    x_ctrl = snr_levels.max() + 1.5 * step                                         # position of the control

    # D: order of the rows
    order_c = np.argsort(-np.nan_to_num(snr50, posinf=1e9, neginf=-1e9))           # most sensitive on top


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dash = lambda lw: f"{pt(3.7*lw):.1f}px,{pt(1.6*lw):.1f}px"                    # "--" line of width lw (pt)
    snr_labels = [f"{v:g}" for v in snr_levels]

    # Limits (matplotlib autoscale: data + 5 %)
    x_range = mpl_range(snr_levels)
    y_range = mpl_range(det_art.to_numpy(), det_all.to_numpy(), det_lab.to_numpy(), [0.5])   # A and B (sharey)
    fp_vals = np.r_[fp_mix.to_numpy().ravel(), fp_ctrl]
    x_range_fp = mpl_range(np.r_[snr_levels, x_ctrl])
    y_range_fp = [0, mpl_range(fp_vals)[1]]                                        # set_ylim(bottom=0)
    x_range_d = [lo_edge - 0.5 * step, hi_edge + 0.5 * step]
    y_range_d = [n_art - 0.5, -0.5]

    legend_d = []
    if np.isfinite(snr50_all):
        legend_d += [f"All artifacts: {snr50_all:.1f} dB", f"CI 95%: ({ci_all[0]:.1f}, {ci_all[1]:.1f}) dB"]
    if np.isneginf(snr50).any():
        legend_d.append(f"< 50% in the whole range (≥ {lo_edge:g} dB)")
    if np.isposinf(snr50).any():
        legend_d.append(f"≥ 50% in the whole range (≤ {hi_edge:g} dB)")
    legend_d.append("SNR at 50% detection (CI 95%)")
    fp_name = "Mean of False Positives ± CI 95%"

    tick_idx = np.arange(0, n_art, 5)
    spec = dict(
        suptitle="Detection",
        A=dict(title="Detection rate vs SNR", xlabel="SNR (dB)", ylabel="Detection rate", xlim=x_range,
               ylim=y_range, xticks=(snr_levels, snr_labels), legend=["All"], loc="upper right"),
        B=dict(title="Detection rate vs SNR by signal class", xlabel="SNR (dB)", ylabel="Detection rate",
               xlim=x_range, ylim=y_range, xticks=(snr_levels, snr_labels), legend=label_order + ["All"],
               loc="upper right"),
        C=dict(title="Number of false positives vs SNR", xlabel="SNR (dB)", ylabel="False Positives",
               xlim=x_range_fp, ylim=y_range_fp, xticks=(list(snr_levels) + [x_ctrl], snr_labels + ["Control"]),
               legend=[r"Mean of False Positives $\pm$ CI 95%"], loc="upper left"),
        D=dict(title=r"SNR at 50% detection vs artifact", xlabel="SNR (dB)", ylabel="Artifact", xlim=x_range_d,
               ylim=y_range_d, yticks=(range(n_art), [art_order[a] for a in order_c]), legend=legend_d,
               loc="lower right"),
        cb=dict(colors=[art_colors[a] for a in art_order], ticks=(tick_idx, [art_order[i] for i in tick_idx]),
                label="Artifact"),
    )

    fig_w, fig_h = 12, 9 + 0.12 * n_art
    geo = _detection_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(3 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _detection_geometry(fig_w, fig_h, dpi, spec)
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H
    dom_x = lambda b: [X(b["x0"]), X(b["x1"])]
    dom_y = lambda b: [Y(b["bottom"]), Y(b["top"])]
    ax_len_pt = lambda b, k: (b["x1"] - b["x0"] if k == "x" else b["bottom"] - b["top"]) / PT

    def nticks(axis_range, b, k):
        """
        plotly nticks that gives the matplotlib ticks (mpl_nticks). plotly rounds range/nticks up to the NEXT
        1-2-5 step when it falls exactly on one (30/6 = 5 -> 10), so in that case one more tick is requested
        """
        n = mpl_nticks(axis_range, ax_len_pt(b, k), 3 if k == "x" else 2)
        rough = (axis_range[1] - axis_range[0]) / n
        return n + 1 if np.isclose(rough / 10**np.floor(np.log10(rough)), [1, 2, 5]).any() else n

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=False, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      tickformatstops=MPL_TICKFORMAT, exponentformat="none")      # sns.despine: no top/right axes
    x_style = dict(**axis_style, ticklabelstandoff=round(-2.2 * PT))
    gA, gB, gC, gD, gcb = (geo[k]["ax"] for k in ("A", "B", "C", "D", "cb"))
    layout = dict(
        # A
        xaxis=dict(domain=dom_x(gA), anchor="y", range=x_range, tickvals=snr_levels, ticktext=snr_labels,
                   **{**x_style, "showgrid": True}),
        yaxis=dict(domain=dom_y(gA), anchor="x", range=y_range, nticks=nticks(y_range, gA, "y"),
                   **{**axis_style, "showgrid": True}),
        # B
        xaxis2=dict(domain=dom_x(gB), anchor="y2", range=x_range, tickvals=snr_levels, ticktext=snr_labels,
                    **{**x_style, "showgrid": True}),
        yaxis2=dict(domain=dom_y(gB), anchor="x2", matches="y", range=y_range,
                    nticks=nticks(y_range, gB, "y"), **{**axis_style, "showgrid": True}),
        # C
        xaxis3=dict(domain=dom_x(gC), anchor="y3", range=x_range_fp, tickvals=list(snr_levels) + [x_ctrl],
                    ticktext=snr_labels + ["Control"], **x_style),
        yaxis3=dict(domain=dom_y(gC), anchor="x3", range=y_range_fp,
                    nticks=nticks(y_range_fp, gC, "y"), **{**axis_style, "showgrid": True}),
        # D
        xaxis4=dict(domain=dom_x(gD), anchor="y4", range=x_range_d,
                    nticks=nticks(x_range_d, gD, "x"), **{**x_style, "showgrid": True}),
        yaxis4=dict(domain=dom_y(gD), anchor="x4", range=y_range_d, tickvals=np.arange(n_art),
                    ticktext=[art_order[a] for a in order_c], **axis_style),
        # Colorbar (own axes: one discrete color per artifact)
        xaxis5=dict(domain=dom_x(gcb), anchor="y5", range=[0, 1], fixedrange=True, showline=True, mirror=True,
                    linecolor="black", linewidth=pt(0.8), showticklabels=False, ticks="", showgrid=False,
                    zeroline=False),
        yaxis5=dict(domain=dom_y(gcb), anchor="x5", side="right", range=[-0.5, n_art - 0.5], fixedrange=True,
                    tickvals=tick_idx, ticktext=[art_order[i] for i in tick_idx],
                    minor=dict(tickvals=np.arange(-0.5, n_art), ticks="outside", ticklen=pt(2), tickwidth=pt(0.6),
                               tickcolor="black", showgrid=False),
                    **{**axis_style, "mirror": True}),
    )

    # Legends: one plotly legend per item, at the position of the matplotlib text (the text of a plotly item
    # starts 40 px to the right of the x of the legend), and a common frame with rounded corners
    legends, shapes, n_leg = {}, [], 0

    def legend_items(key, names):
        nonlocal n_leg
        ids = []
        for name, t in zip(names, geo[key]["legend_texts"]):
            n_leg += 1
            lid = "legend" if n_leg == 1 else f"legend{n_leg}"
            legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)
            ids.append(lid)
        b = geo[key]["legend"]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                           line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H)))
        return dict(zip(names, ids))

    leg_A = legend_items("A", ["All"])
    leg_B = legend_items("B", label_order + ["All"])
    leg_C = legend_items("C", [fp_name])
    leg_D = legend_items("D", legend_d)

    fig = go.Figure()

    def line_trace(x, y, color, lw, mew, alpha=1.0, ms=6, mec="#000000", **kw):
        """Line with "o" markers (sns.lineplot(marker="o")); alpha on the line and the markers, as Line2D"""
        return go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=rgba(color, alpha), width=pt(lw)),
                          marker=dict(size=pt(ms), color=rgba(color, alpha),
                                      line=dict(color=rgba(mec, alpha), width=pt(mew))), **kw)

    def band_trace(x, lo, hi, color, **kw):
        """Band of the 95 % CI (fill_between, alpha 0.2 of seaborn)"""
        return go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]], mode="lines", fill="toself",
                          fillcolor=rgba(color, 0.2), line=dict(width=0), hoverinfo="skip", showlegend=False, **kw)

    hover_rate = ("%{fullData.name}<br>SNR: %{x:g} dB<br>Detection rate: %{y:.3f}<br>"
                  "CI 95%: (%{customdata[0]:.3f}, %{customdata[1]:.3f})<extra></extra>")

    # --- A: detection rate ---
    # Bands below all the lines (zorder 1 < 2 in matplotlib)
    for a in art_order:
        d = det_art.loc[a]
        fig.add_trace(band_trace(snr_levels, d["lo"].to_numpy(), d["hi"].to_numpy(), art_colors[a],
                                 xaxis="x", yaxis="y", legendgroup=f"art{a}"))
    fig.add_trace(band_trace(snr_levels, det_all["lo"].to_numpy(), det_all["hi"].to_numpy(), "#000000",
                             xaxis="x", yaxis="y", legendgroup="all_A", legend=leg_A["All"]))
    for a in art_order:
        d = det_art.loc[a]
        fig.add_trace(line_trace(snr_levels, d["mean"], art_colors[a], 1.5, 1.5, alpha=0.3, name=f"Artifact {a}",
                                 customdata=d[["lo", "hi"]].to_numpy(), hovertemplate=hover_rate, xaxis="x",
                                 yaxis="y", showlegend=False, legendgroup=f"art{a}"))
    fig.add_trace(line_trace(snr_levels, det_all["mean"], "#000000", 3, 3, name="All",
                             customdata=det_all[["lo", "hi"]].to_numpy(), hovertemplate=hover_rate, xaxis="x",
                             yaxis="y", legend=leg_A["All"], legendgroup="all_A"))

    # --- B: detection rate by class of the clean signal ---
    for lab in label_order:
        d = det_lab.loc[lab]
        fig.add_trace(band_trace(snr_levels, d["lo"].to_numpy(), d["hi"].to_numpy(), label_colors[lab],
                                 xaxis="x2", yaxis="y2", legend=leg_B[lab], legendgroup=f"lab_{lab}"))
    for lab in label_order:
        d = det_lab.loc[lab]
        fig.add_trace(line_trace(snr_levels, d["mean"], label_colors[lab], 1.5, 1.5, name=lab,
                                 customdata=d[["lo", "hi"]].to_numpy(), hovertemplate=hover_rate, xaxis="x2",
                                 yaxis="y2", legend=leg_B[lab], legendgroup=f"lab_{lab}"))
    fig.add_trace(line_trace(snr_levels, det_all["mean"], "#000000", 3, 3, name="All",
                             hovertemplate="All<br>SNR: %{x:g} dB<br>Detection rate: %{y:.3f}<extra></extra>",
                             xaxis="x2", yaxis="y2", legend=leg_B["All"], legendgroup="all_B"))

    # --- C: false positives ---
    hover_fp = ("%{customdata[2]}<br>False positives (mean): %{y:.3f}<br>"
                "CI 95%: (%{customdata[0]:.3f}, %{customdata[1]:.3f})<extra></extra>")
    error_fp = lambda m, lo, hi, color: dict(type="data", symmetric=False, array=np.asarray(hi) - m,
                                             arrayminus=m - np.asarray(lo), color=color, thickness=pt(1.5),
                                             width=pt(4))
    m = fp_mix["mean"].to_numpy()
    fig.add_trace(line_trace(snr_levels, m, "#000000", 1.5, 0.75, name=fp_name,
                             error_y=error_fp(m, fp_mix["lo"], fp_mix["hi"], "#000000"),
                             customdata=np.c_[fp_mix[["lo", "hi"]].to_numpy(), [f"SNR: {v} dB" for v in snr_labels]],
                             hovertemplate=hover_fp, xaxis="x3", yaxis="y3", legend=leg_C[fp_name],
                             legendgroup="fp"))
    fig.add_trace(line_trace([x_ctrl], [fp_ctrl[0]], "#808080", 1.5, 0.75, name="Control",
                             error_y=error_fp(np.array([fp_ctrl[0]]), [fp_ctrl[1]], [fp_ctrl[2]], "#808080"),
                             customdata=[[fp_ctrl[1], fp_ctrl[2], "Control (no artifact)"]], hovertemplate=hover_fp,
                             xaxis="x3", yaxis="y3", showlegend=False, legend=leg_C[fp_name], legendgroup="fp"))
    shapes.append(dict(type="line", xref="x3", yref="y3 domain", x0=(snr_levels.max() + x_ctrl) / 2,
                       x1=(snr_levels.max() + x_ctrl) / 2, y0=0, y1=1, layer="below",
                       line=dict(color="#d3d3d3", width=pt(1))))

    # --- D: SNR at 50 % per artifact ---
    if np.isfinite(snr50_all):
        shapes.append(dict(type="rect", xref="x4", yref="y4 domain", x0=ci_all[0], x1=ci_all[1], y0=0, y1=1,
                           layer="below", fillcolor=rgba("#808080", 0.15), line=dict(width=0),
                           name=legend_d[1], showlegend=True, legend=leg_D[legend_d[1]]))
        shapes.append(dict(type="line", xref="x4", yref="y4 domain", x0=snr50_all, x1=snr50_all, y0=0, y1=1,
                           layer="below", line=dict(color="#000000", width=pt(2), dash=dash(2)),
                           name=legend_d[0], showlegend=True, legend=leg_D[legend_d[0]]))

    name_lo, name_hi, name_50 = (f"< 50% in the whole range (≥ {lo_edge:g} dB)",
                                 f"≥ 50% in the whole range (≤ {hi_edge:g} dB)", "SNR at 50% detection (CI 95%)")
    for row, a in enumerate(order_c):
        col, est, art = art_colors[art_order[a]], snr50[a], art_order[a]
        if np.isinf(est):                                                          # crossing out of the range
            name = name_hi if est > 0 else name_lo
            fig.add_trace(go.Scatter(x=[hi_edge if est > 0 else lo_edge], y=[row], mode="markers", name=name,
                                     marker=dict(symbol="triangle-right" if est > 0 else "triangle-left", size=pt(7),
                                                 color=col, line=dict(color="black", width=pt(1))),
                                     hovertemplate=f"Artifact {art}<br>{name}<extra></extra>", xaxis="x4",
                                     yaxis="y4", showlegend=False, legend=leg_D[name], legendgroup=name))
        else:
            fig.add_trace(go.Scatter(x=[est], y=[row], mode="markers", name=name_50,
                                     marker=dict(size=pt(6), color=col, line=dict(color="black", width=pt(1))),
                                     error_x=dict(type="data", symmetric=False, array=[max(ci_hi[a] - est, 0)],
                                                  arrayminus=[max(est - ci_lo[a], 0)], color=col,
                                                  thickness=pt(1.5), width=pt(3)),
                                     hovertemplate=(f"Artifact {art}<br>SNR at 50% detection: {est:.1f} dB<br>"
                                                    f"CI 95%: ({ci_lo[a]:.1f}, {ci_hi[a]:.1f}) dB<extra></extra>"),
                                     xaxis="x4", yaxis="y4", showlegend=False, legend=leg_D[name_50],
                                     legendgroup=name_50))
    # Legend entries in black for the markers
    for name, symbol, size in [(name_lo, "triangle-left", 7), (name_hi, "triangle-right", 7), (name_50, "circle", 6)]:
        if name in leg_D:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, xaxis="x4", yaxis="y4",
                                     marker=dict(symbol=symbol, size=pt(size), color="black",
                                                 line=dict(color="black", width=pt(1))),
                                     legend=leg_D[name], legendgroup=name))

    # --- Colorbar: artifact_id ---
    colorscale = [[(k + e) / n_art, art_colors[a]] for k, a in enumerate(art_order) for e in (0, 1)]
    fig.add_trace(go.Heatmap(x=[0, 1], y=np.arange(-0.5, n_art), z=np.arange(n_art)[:, None], zmin=-0.5,
                             zmax=n_art - 0.5, colorscale=colorscale, showscale=False, xaxis="x5", yaxis="y5",
                             text=np.array(art_order)[:, None], hovertemplate="Artifact %{text}<extra></extra>"))

    # --- Lines y = 0.5 (A, B) ---
    for ax in ("", "2"):
        shapes.append(dict(type="line", xref=f"x{ax} domain", yref=f"y{ax}", x0=0, x1=1, y0=0.5, y1=0.5,
                           layer="above", line=dict(color="#808080", width=pt(2), dash=dash(2))))

    # --- Titles and labels ---
    def ann(s, b, size, rotate=False):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size)), textangle=-90 if rotate else 0)

    annotations = [ann("Detection", geo["suptitle"], 16)]
    for k in ("A", "B", "C", "D"):
        s = spec[k]
        annotations += [ann(f"<b>{s['title']}</b>", geo[k]["title"], 14), ann(s["xlabel"], geo[k]["xlabel"], 12),
                        ann(s["ylabel"], geo[k]["ylabel"], 12, rotate=True)]
    annotations.append(ann("Artifact", geo["cb"]["ylabel"], 10, rotate=True))

    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )
    return fig


# --------------------------------------------------------------------------------------------------------------------
# 4.2. Elimination (section 5.4.2)
# --------------------------------------------------------------------------------------------------------------------

def _elimination_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_elimination figure as matplotlib computes it (layout="constrained"): builds the same figure
    WITHOUT data (titles, labels, limits, ticks, legends and the right axis of C) and returns the boxes (px from the
    top-left corner of the image cropped as bbox_inches="tight") of the axes, texts and legends, and the final limits
    of each panel (A: set_aspect("equal", adjustable="datalim") changes the limits depending on the size of the axes).
    In C, the ticks of the left axis and the limits and ticks of the right one (aligned with the grid of the left one,
    as in the matplotlib version).
    spec: {"A"|"B"|"C": dict(title, xlabel, ylabel, xlim, ylim, xticks, legend, loc, legend_title, aspect),
    "C_rate_label": str, "suptitle": str}. With aspect, xlim/ylim are the limits of the DATA (without margins)
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    axs = {"A": fig.add_subplot(gs[0:2, 0]), "B": fig.add_subplot(gs[0, 1]), "C": fig.add_subplot(gs[1, 1])}

    for k, ax in axs.items():
        s = spec[k]
        ax.set_title(s["title"], fontsize=14, fontweight="bold")
        ax.set_xlabel(s["xlabel"], fontsize=12)
        ax.set_ylabel(s["ylabel"], fontsize=12)
        if s.get("xticks"):
            ax.set_xticks(*s["xticks"])
        if s.get("aspect"):                                                       # data limits + 5 % y
            ax.update_datalim(np.c_[s["xlim"], s["ylim"]])                         # aspect (adjusts the limits)
            ax.autoscale_view()
            ax.set_aspect(s["aspect"], adjustable="datalim")
        else:
            ax.set_xlim(*s["xlim"])
            ax.set_ylim(*s["ylim"])
        ax.legend([Line2D([], [], marker="o") for _ in s["legend"]], s["legend"], loc=s["loc"], fontsize=10,
                  title=s.get("legend_title"), title_fontproperties=dict(weight="bold"))

    # C: right axis (detection rate) aligned with the grid of the left one, as in plot_elimination
    ax, twin = axs["C"], axs["C"].twinx()
    y0, y1 = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if y0 <= t <= y1]
    ax.set_yticks(ticks)
    t_max = max(ticks)
    twin.set_ylim(y0 / t_max, y1 / t_max)
    twin.set_yticks([t / t_max for t in ticks if t >= 0])
    twin.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip("."))
    twin.set_ylabel(spec["C_rate_label"], fontsize=12)
    fig.suptitle(spec["suptitle"], fontsize=16)

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle))
    for k, ax in axs.items():
        leg = ax.get_legend()
        out[k] = dict(ax=box(ax), xlabel=box(ax.xaxis.label), ylabel=box(ax.yaxis.label), title=box(ax.title),
                      legend=box(leg), legend_texts=[box(t) for t in leg.get_texts()],
                      legend_title=box(leg.get_title()), xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
    out["C"]["yticks"] = ([t.get_loc() for t in ax.yaxis.get_major_ticks()],
                          [t.label1.get_text() for t in ax.yaxis.get_major_ticks()])
    out["C"]["rate"] = dict(ylabel=box(twin.yaxis.label), ylim=list(twin.get_ylim()),
                            ticks=([t.get_loc() for t in twin.yaxis.get_major_ticks()],
                                   [t.label2.get_text() for t in twin.yaxis.get_major_ticks()]))
    return out


def plot_elimination(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_elimination (matplotlib/seaborn, section 5.4.2 of evaluation_denoise.ipynb):
    same layout (layout="constrained", figsize 13 x 8 in), sizes, fonts, colors and positions. The figure is cropped as
    the inline backend does (bbox_inches="tight"), so its size matches the matplotlib image.
    - A: SNR out vs SNR in of the detected artifacts (points with jitter and median ± IQR) with a 1:1 aspect (also
      when zooming)
    - B: violins of the ΔSNR of the detected artifacts per SNR level (Scott KDE, cut=0, density_norm="width",
      quartiles), with the number of mixtures above each violin
    - C: net SNR improvement of all the mixtures (not detected -> ΔSNR = 0) per class of the clean signal and global,
      and the detection rate on the right axis (aligned with the grid of the left one)
    Interactivity: hover with the values (median and IQR or mean and 95 % CI, n, quartiles of each violin; in A, clean
    signal, artifact and ΔSNR of each mixture); click on the legend to hide/show each item (line, band and points;
    violins and their n); zoom in each panel.
    The positions of the axes and texts are taken from the constrained layout of matplotlib applied to the same figure
    without data (_elimination_geometry).
    Limits: the legends are placed where loc="best" puts them with these data (upper right); the plotly legend symbols
    are not the matplotlib ones (violins: square with edge; markers at 5-35 px instead of the 2 em segment); the plotly
    grid always stays below the points and the bands (in matplotlib, above); the jitter of A and the bootstrap CIs use
    a fixed seed (seaborn does not), so they can differ slightly from the matplotlib version; the n= texts of B are
    hidden with their violins, but the reference line of B does not extend when zooming out; the ticks of both axes of
    C are fixed (the matplotlib ones): when zooming in C both axes scale together, but the ticks are not recomputed.

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 1 (read_results("experiment_1.csv")). It is not modified
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Style constants ===
    C_DET = to_hex("tab:red")                                                   # detected in A and B
    C_RATE = to_hex("dimgray")                                                  # detection rate (right axis of C)
    C_GRAY, C_BLACK = to_hex("gray"), "#000000"
    LBL_NOCHANGE = "No change (ΔSNR = 0)"
    LBL_LINE_A = "Median ± IQR (detected)"
    LBL_RATE = "Detection rate"
    N_BOOT = 1000

    def desaturate(color, prop):
        """Color with its HLS saturation multiplied by prop (seaborn.utils.desaturate: violin saturation=0.75)"""
        h, l, s = colorsys.rgb_to_hls(*to_rgb(color))
        return to_hex(colorsys.hls_to_rgb(h, l, s*prop))

    C_VIOLIN = desaturate(C_DET, 0.75)
    lum = 0.6*colorsys.rgb_to_hls(*to_rgb(C_VIOLIN))[1]                          # linecolor="auto" of seaborn
    C_VIOLIN_LINE = to_hex((lum, lum, lum))


    # === Data ===
    # copy: results_df is not modified
    mix_df = results_df[results_df["art_id"] != "Control"].astype({"snr_db": float})

    det = mix_df[mix_df["detected"] == 1].copy()                        # detected
    lv = np.sort(det["snr_db"].unique())                                # SNR levels
    lv_labels = [f"{v:g}" for v in lv]                                  # -30, not -30.0 (minus sign, as matplotlib)

    # C: mean ΔSNR of all the mixtures (one row per clean signal and SNR); not detected -> ΔSNR = 0
    net = mix_df.assign(dsnr=mix_df["dsnr_db"].where(mix_df["detected"] == 1, 0.0))
    net_all = net.groupby(["clean_id", "snr_db"], as_index=False)["dsnr"].mean()
    net_lab = net.groupby(["clean_id", "label", "snr_db"], as_index=False)["dsnr"].mean()
    label_order = sorted(net_lab["label"].unique())
    label_colors = dict(zip(label_order, [to_hex(c) for c in colormaps["Set2"].colors]))   # = palette="Set2"
    rate = mix_df.groupby(["clean_id", "snr_db"], as_index=False)["detected"].mean()   # detection rate


    # --- Estimators and intervals (as seaborn) ---
    rng_ci = np.random.default_rng(1)

    def mean_ci(v):
        """Mean and bootstrap 95 % CI of the observations (errorbar=("ci", 95), n_boot=1000 of seaborn)"""
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        boots = v[rng_ci.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
        return (v.mean(), *np.percentile(boots, [2.5, 97.5]), len(v))

    def median_iqr(v):
        """Median and 25-75 percentile interval (estimator="median", errorbar=("pi", 50) of seaborn)"""
        v = np.asarray(v, float)
        return (np.median(v), *np.percentile(v, [25, 75]), len(v))

    def stat_table(df, by, col, fn=mean_ci):
        """
        Central estimator (mean or median, depending on fn), lower and upper limits and n of col per group (index: by)
        """
        gb = df.groupby(by)
        index = (pd.Index(list(gb.groups.keys())) if isinstance(by, str)
                 else pd.MultiIndex.from_tuples(list(gb.groups.keys())))
        return pd.DataFrame([fn(gr[col]) for _, gr in gb], index=index, columns=["est", "lo", "hi", "n"])

    # A: median ± IQR of SNR out; points with jitter (jitter=0.3 x spacing of the levels)
    io_all = stat_table(det, "snr_db", "snr_out_db", median_iqr)
    jlim = 0.3*(np.min(np.diff(lv)) if len(lv) > 1 else 1)
    det["x_jitter"] = det["snr_db"] + np.random.default_rng(0).uniform(-jlim, jlim, len(det))

    # B: violins (KDE of each level, as sns.violinplot(cut=0, density_norm="width", inner="quart", width=0.8))
    hw = 0.8 / 2                                                        # maximum half-width of each violin
    violins = []
    for i, lvl in enumerate(lv):
        obs = det.loc[det["snr_db"] == lvl, "dsnr_db"].to_numpy()
        q = np.percentile(obs, [25, 50, 75])
        if len(obs) > 1 and np.ptp(obs) > 0:
            support = np.linspace(obs.min(), obs.max(), 100)
            density = gaussian_kde(obs, bw_method="scott")(support)
            span = density / density.max() * hw
            q_span = np.interp(q, support, span)
        else:                                                           # no variance: a horizontal line
            support, span, q_span = np.full(2, obs.mean()), np.full(2, hw), np.zeros(3)
        violins.append(dict(level=lvl, pos=i, support=support, span=span, q=q, q_span=q_span, n=len(obs),
                            mean=obs.mean(), top=obs.max()))

    # C: mean and 95 % CI; detection rate (mean of the clean signals, without interval)
    net_all_t = stat_table(net_all, "snr_db", "dsnr")
    net_lab_t = stat_table(net_lab, ["label", "snr_db"], "dsnr")
    det_rate = rate.groupby("snr_db")["detected"].mean().astype(float)


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dash = lambda lw: mpl_dashes(DASH_PATTERNS["--"], lw, PT)                     # "--" line of width lw (pt)
    dot = lambda lw: mpl_dashes(DASH_PATTERNS[":"], lw, PT)                       # ":" line of width lw (pt)
    vals = lambda t: t[["est", "lo", "hi"]].to_numpy()

    # Limits (matplotlib autoscale: data + 5 %)
    x_lim_a = mpl_range(det["x_jitter"], lv, margin=0)                            # A: data limits (the
    y_lim_a = mpl_range(det["snr_out_db"], vals(io_all), lv, margin=0)            # aspect changes them)
    x_range_b = [-0.5, len(lv) - 0.5]                                              # categorical axis
    y_range_b = mpl_range(*[v["support"] for v in violins], [0])
    y_range_b[1] += 4                                                              # room for the n= texts
    x_range = mpl_range(lv)
    y_range_c = mpl_range(vals(net_lab_t), net_all_t["est"], [0])

    legend_a = [LBL_LINE_A, LBL_NOCHANGE]
    legend_b = [LBL_NOCHANGE]
    legend_c = [*label_order, "All", LBL_RATE]
    spec = dict(
        suptitle="Elimination", C_rate_label=LBL_RATE,
        A=dict(title="SNR out vs SNR in", xlabel="SNR in (dB)", ylabel="SNR out (dB)", xlim=x_lim_a,
               ylim=y_lim_a, legend=legend_a, loc="upper right", aspect="equal"),
        B=dict(title="ΔSNR vs SNR in (detected)", xlabel="SNR in (dB)", ylabel="ΔSNR (dB)", xlim=x_range_b,
               ylim=y_range_b, xticks=(range(len(lv)), lv_labels), legend=legend_b, loc="upper right"),
        C=dict(title="Net SNR improvement by signal class (all mixtures)", xlabel="SNR in (dB)",
               ylabel="Mean ΔSNR (dB)", xlim=x_range, ylim=y_range_c, xticks=(lv, lv_labels), legend=legend_c,
               loc="upper right", legend_title="Signal label"),
    )

    fig_w, fig_h = 13, 8
    geo = _elimination_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(6 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _elimination_geometry(fig_w, fig_h, dpi, spec)
    x_range_a, y_range_a = geo["A"]["xlim"], geo["A"]["ylim"]                      # after the 1:1 aspect
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H
    dom_x = lambda b: [X(b["x0"]), X(b["x1"])]
    dom_y = lambda b: [Y(b["bottom"]), Y(b["top"])]
    ax_len_pt = lambda b, k: (b["x1"] - b["x0"] if k == "x" else b["bottom"] - b["top"]) / PT

    def nticks(axis_range, b, k):
        """
        plotly nticks that gives the matplotlib ticks (mpl_nticks). plotly rounds range/nticks up to the NEXT
        1-2-5 step when it falls exactly on one (30/6 = 5 -> 10), so in that case one more tick is requested
        """
        n = mpl_nticks(axis_range, ax_len_pt(b, k), 3 if k == "x" else 2)
        rough = (axis_range[1] - axis_range[0]) / n
        return n + 1 if np.isclose(rough / 10**np.floor(np.log10(rough)), [1, 2, 5]).any() else n

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      tickformatstops=MPL_TICKFORMAT, exponentformat="none")      # sns.despine: no top/right axes
    x_style = dict(**axis_style, ticklabelstandoff=round(-2.2 * PT))
    g = {k: geo[k]["ax"] for k in "ABC"}
    rate_geo = geo["C"]["rate"]
    layout = dict(
        # A (1:1 aspect also when zooming)
        xaxis=dict(domain=dom_x(g["A"]), anchor="y", range=x_range_a, nticks=nticks(x_range_a, g["A"], "x"),
                   **x_style),
        yaxis=dict(domain=dom_y(g["A"]), anchor="x", range=y_range_a, nticks=nticks(y_range_a, g["A"], "y"),
                   scaleanchor="x", scaleratio=1, **axis_style),
        # B (categorical x axis: one SNR level per position)
        xaxis2=dict(domain=dom_x(g["B"]), anchor="y2", range=x_range_b, tickvals=np.arange(len(lv)),
                    ticktext=lv_labels, **{**x_style, "showgrid": False}),
        yaxis2=dict(domain=dom_y(g["B"]), anchor="x2", range=y_range_b, nticks=nticks(y_range_b, g["B"], "y"),
                    **axis_style),
        # C: left axis with the matplotlib ticks and right axis (detection rate) overlaid
        xaxis3=dict(domain=dom_x(g["C"]), anchor="y3", range=x_range, tickvals=lv, ticktext=lv_labels, **x_style),
        yaxis3=dict(domain=dom_y(g["C"]), anchor="x3", range=y_range_c, tickvals=geo["C"]["yticks"][0],
                    ticktext=geo["C"]["yticks"][1], **axis_style),
        yaxis4=dict(**{**axis_style, "showgrid": False, "linecolor": C_RATE, "tickcolor": C_RATE,
                       "tickfont": dict(size=em, color=C_RATE)},
                    anchor="x3", overlaying="y3", side="right", range=rate_geo["ylim"],
                    tickvals=rate_geo["ticks"][0], ticktext=rate_geo["ticks"][1]),
    )

    # Legends: one plotly legend per item, at the position of the matplotlib text (the text of a plotly item
    # starts 40 px to the right of the x of the legend), and a common frame with rounded corners. The title
    # of the legend of C is a bold annotation
    legends, frames, n_leg = {}, [], 0
    annotations = []

    def legend_items(key, names):
        nonlocal n_leg
        ids = {}
        for name, t in zip(names, geo[key]["legend_texts"]):
            n_leg += 1
            lid = "legend" if n_leg == 1 else f"legend{n_leg}"
            legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)
            ids[name] = lid
        b = geo[key]["legend"]
        frames.append(dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                           line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H)))
        return ids

    leg_A = legend_items("A", legend_a)
    leg_B = legend_items("B", legend_b)
    leg_C = legend_items("C", legend_c)
    t = geo["C"]["legend_title"]
    annotations.append(dict(text="<b>Signal label</b>", x=X((t["x0"] + t["x1"]) / 2),
                            y=Y((t["top"] + t["bottom"]) / 2 - 0.1*em), xref="paper", yref="paper", xanchor="center",
                            yanchor="middle", showarrow=False, font=dict(size=em)))

    fig = go.Figure()
    shapes = []

    def line_trace(x, y, color, lw, ms=6, mew=0.75, marker=True, pattern=None, gl=False, **kw):
        """
        Line (with "o" markers, as sns.lineplot(marker="o", markeredgecolor="black")); pattern: plotly dash;
        gl: Scattergl (plotly draws the WebGL layer over the SVG traces: needed to stay over another Scattergl)
        """
        return (go.Scattergl if gl else go.Scatter)(x=x, y=y, mode="lines+markers" if marker else "lines",
                          line=dict(color=color, width=pt(lw), dash=pattern),
                          marker=dict(size=pt(ms), color=color, line=dict(color=C_BLACK, width=pt(mew))), **kw)

    def band_trace(x, lo, hi, color, gl=False, **kw):
        """Band of the interval (seaborn fill_between: alpha 0.2, with a 1 pt edge of the same color)"""
        x, lo, hi = np.asarray(x), np.asarray(lo), np.asarray(hi)
        return (go.Scattergl if gl else go.Scatter)(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]], mode="lines", fill="toself",
                          fillcolor=rgba(color, 0.2), line=dict(color=rgba(color, 0.2), width=pt(1)),
                          hoverinfo="skip", showlegend=False, **kw)

    def stat_line(tab, name, color, lw, hover, axis, legend, group, band=True, gl=False, **kw):
        """Band + line with the estimator of tab (index: SNR); hover with the estimator, the interval and n"""
        x = tab.index.to_numpy(float)
        traces = []
        if band:
            traces.append(band_trace(x, tab["lo"], tab["hi"], color, gl=gl, xaxis=f"x{axis}", yaxis=f"y{axis}",
                                     legend=legend, legendgroup=group))
        traces.append(line_trace(x, tab["est"], color, lw, gl=gl, name=name,
                                 customdata=tab[["lo", "hi", "n"]].to_numpy(), hovertemplate=hover,
                                 xaxis=f"x{axis}", yaxis=f"y{axis}", legend=legend, legendgroup=group, **kw))
        return traces

    def hline(y, color, axis, name=None, legend=None):
        """Dashed horizontal line across the whole width of the panel (axhline, lw 2)"""
        return dict(type="line", xref=f"x{axis} domain", yref=f"y{axis}", x0=0, x1=1, y0=y, y1=y, layer="above",
                    line=dict(color=color, width=pt(2), dash=dash(2)), name=name, showlegend=legend is not None,
                    **({"legend": legend} if legend else {}))

    # --- A: SNR out vs SNR in ---
    # Points (zorder 1) below the band and the line; they are hidden with the line in the legend.
    # The plotly hover picks the point closest to the cursor (in WebGL, its center, regardless of its size): points
    # within R_FREE px (initial view) of a median are drawn the same but without hover, so that the median gets it
    R_FREE = pt(6 + 2*1.5)                                                         # diameter of the median marker
    px_x = (g["A"]["x1"] - g["A"]["x0"]) / (x_range_a[1] - x_range_a[0])           # px per dB
    px_y = (g["A"]["bottom"] - g["A"]["top"]) / (y_range_a[1] - y_range_a[0])
    med = io_all["est"].reindex(det["snr_db"]).to_numpy()
    near = np.hypot((det["x_jitter"] - det["snr_db"]).to_numpy() * px_x,
                    (det["snr_out_db"].to_numpy() - med) * px_y) < R_FREE
    for part, hover in ((det[~near], True), (det[near], False)):
        fig.add_trace(go.Scattergl(
            x=part["x_jitter"], y=part["snr_out_db"], mode="markers", name="Detected", xaxis="x", yaxis="y",
            marker=dict(size=pt(2.5), color=rgba(C_DET, 0.1), line=dict(width=0)),
            customdata=part[["snr_db", "dsnr_db", "clean_file", "art_id"]].to_numpy(float),
            hovertemplate=("Detected<br>SNR in: %{customdata[0]:g} dB<br>SNR out: %{y:.2f} dB<br>"
                           "ΔSNR: %{customdata[1]:.2f} dB<br>Clean signal: %{customdata[2]:.0f}<br>"
                           "Artifact: %{customdata[3]:.0f}<extra></extra>") if hover else None,
            hoverinfo=None if hover else "skip",
            showlegend=False, legend=leg_A[LBL_LINE_A], legendgroup="A_det"))
    fig.add_traces(stat_line(io_all, LBL_LINE_A, C_DET, 2,
                             ("Detected<br>SNR in: %{x:g} dB<br>Median SNR out: %{y:.2f} dB<br>"
                              "IQR: (%{customdata[0]:.2f}, %{customdata[1]:.2f}) dB<br>n = %{customdata[2]}"
                              "<extra></extra>"), "", leg_A[LBL_LINE_A], "A_det",
                             gl=True, mew=1.5))       # WebGL: over the points; thicker edge than in matplotlib
                                                      # (0.75 pt) to tell the medians apart over the points
    fig.add_trace(line_trace(lv, lv, C_GRAY, 2, marker=False, name=LBL_NOCHANGE, xaxis="x", yaxis="y",
                             pattern=dash(2), hoverinfo="skip", legend=leg_A[LBL_NOCHANGE]))

    # --- B: ΔSNR per level (violins) ---
    for v in violins:
        stats = (f"Detected<br>SNR in: {v['level']:g} dB<br>n = {v['n']}<br>Mean ΔSNR: {v['mean']:.2f} dB<br>"
                 f"Q1 / median / Q3: {v['q'][0]:.2f} / {v['q'][1]:.2f} / {v['q'][2]:.2f} dB")
        fig.add_trace(go.Scatter(x=np.r_[v["pos"] - v["span"], v["pos"] + v["span"][::-1]],
                                 y=np.r_[v["support"], v["support"][::-1]], mode="lines", fill="toself",
                                 fillcolor=C_VIOLIN, line=dict(color=C_VIOLIN_LINE, width=pt(1)), name="Detected",
                                 hoveron="fills", text=stats, hoverinfo="text", xaxis="x2", yaxis="y2",
                                 showlegend=False))
        for qv, qs, dq in zip(v["q"], v["q_span"], [(1.25, 0.75), (2.5, 1), (1.25, 0.75)]):   # Q1, median, Q3
            fig.add_trace(go.Scatter(x=[v["pos"] - qs, v["pos"] + qs], y=[qv, qv], mode="lines", hoverinfo="skip",
                                     line=dict(color=C_VIOLIN_LINE, width=pt(1), dash=f"{pt(dq[0]):.2f}px,{pt(dq[1]):.2f}px"),
                                     xaxis="x2", yaxis="y2", showlegend=False))
    # Reference line (after the violins, below the n= texts)
    fig.add_trace(line_trace(x_range_b, [0, 0], C_GRAY, 2, marker=False, name=LBL_NOCHANGE, xaxis="x2", yaxis="y2",
                             pattern=dash(2), hoverinfo="skip", legend=leg_B[LBL_NOCHANGE]))
    # Number of mixtures above each violin
    fig.add_trace(go.Scatter(x=[v["pos"] for v in violins], y=[v["top"] + 1 for v in violins], mode="text",
                             text=[f"n={v['n']}" for v in violins], textposition="top center",
                             textfont=dict(size=pt(8), color=C_DET), hoverinfo="skip", xaxis="x2", yaxis="y2",
                             showlegend=False))

    # --- C: net improvement per class + detection rate ---
    hover_net = ("%{fullData.name}<br>SNR in: %{x:g} dB<br>Mean ΔSNR: %{y:.2f} dB<br>"
                 "CI 95%: (%{customdata[0]:.2f}, %{customdata[1]:.2f}) dB<br>"
                 "n = %{customdata[2]} clean signals<extra></extra>")
    for lab in label_order:                                             # bands below all the lines
        tab = net_lab_t.loc[lab]
        fig.add_trace(band_trace(tab.index.to_numpy(float), tab["lo"], tab["hi"], label_colors[lab], xaxis="x3",
                                 yaxis="y3", legend=leg_C[lab], legendgroup=f"C_{lab}"))
    for lab in label_order:
        fig.add_traces(stat_line(net_lab_t.loc[lab], lab, label_colors[lab], 1.5, hover_net, "3", leg_C[lab],
                                 f"C_{lab}", band=False))
    fig.add_trace(line_trace(net_all_t.index, net_all_t["est"], C_BLACK, 2, name="All", xaxis="x3", yaxis="y3",
                             legend=leg_C["All"], legendgroup="C_all",
                             hovertemplate="All<br>SNR in: %{x:g} dB<br>Mean ΔSNR: %{y:.2f} dB<extra></extra>"))
    shapes.append(hline(0, C_GRAY, "3"))
    fig.add_trace(line_trace(det_rate.index, det_rate.to_numpy(), C_RATE, 2, marker=False, pattern=dot(2),
                             name=LBL_RATE, xaxis="x3", yaxis="y4", legend=leg_C[LBL_RATE],
                             hovertemplate="Detection rate<br>SNR in: %{x:g} dB<br>Rate: %{y:.3f}<extra></extra>"))

    # --- Titles and labels ---
    def ann(s, b, size, rotate=False, color="black"):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size), color=color), textangle=-90 if rotate else 0)

    annotations.append(ann("Elimination", geo["suptitle"], 16))
    for k in "ABC":
        s = spec[k]
        annotations += [ann(f"<b>{s['title']}</b>", geo[k]["title"], 14), ann(s["xlabel"], geo[k]["xlabel"], 12),
                        ann(s["ylabel"], geo[k]["ylabel"], 12, rotate=True)]
    annotations.append(ann(LBL_RATE, rate_geo["ylabel"], 12, rotate=True, color=C_RATE))

    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes + frames, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )
    return fig


# --------------------------------------------------------------------------------------------------------------------
# 4.3. Conservation (section 5.4.3)
# --------------------------------------------------------------------------------------------------------------------

def _conservation_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_conservation figure as matplotlib computes it (layout="constrained"): builds the same figure
    WITHOUT data (titles, labels, limits, ticks, legends) and returns the boxes (px from the top-left corner of the
    image cropped as bbox_inches="tight") of the axes, texts and legends.
    spec: {"A"|"B"|"C"|"D"|"E": dict(title, xlabel, ylabel, xlim, ylim, xticks, title_pad, legend, legend_title,
    legend_kw), "suptitle": str}
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
    axs = {"A": fig.add_subplot(gs[0, 0]), "B": fig.add_subplot(gs[0, 1]), "C": fig.add_subplot(gs[1, 0]),
           "D": fig.add_subplot(gs[1, 1]), "E": fig.add_subplot(gs[2, :])}

    for k, ax in axs.items():
        s = spec[k]
        ax.set_title(s["title"], fontsize=14, fontweight="bold", pad=s.get("title_pad"))
        ax.set_xlabel(s["xlabel"], fontsize=12)
        ax.set_ylabel(s["ylabel"], fontsize=12)
        if s.get("xticks"):
            ax.set_xticks(*s["xticks"])
        ax.set_xlim(*s["xlim"])
        ax.set_ylim(*s["ylim"])
        ax.legend([Line2D([], [], marker="o") for _ in s["legend"]], s["legend"], fontsize=10,
                  title=s.get("legend_title"), title_fontproperties=dict(weight="bold"), **s["legend_kw"])
    fig.suptitle(spec["suptitle"], fontsize=16)

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle))
    for k, ax in axs.items():
        leg = ax.get_legend()
        out[k] = dict(ax=box(ax), xlabel=box(ax.xaxis.label), ylabel=box(ax.yaxis.label), title=box(ax.title),
                      legend=box(leg), legend_texts=[box(t) for t in leg.get_texts()],
                      legend_title=box(leg.get_title()))
    return out


def plot_conservation(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_conservation (matplotlib/seaborn, section 5.4.3 of evaluation_denoise.ipynb):
    same layout (layout="constrained", figsize 14 x 11 in, suptitle), sizes, fonts, colors and positions. The figure is
    cropped as the inline backend does (bbox_inches="tight"), so its size matches the matplotlib image.
    - A: NMSE out of the zone vs SNR (mean ± 95 % CI over the clean signals) and control
    - B: boxes of the NMSE out of the zone by number of detections out of the zone (mixtures and control), with the
      number of cases above each box
    - C: NMSE out of the zone vs SNR per class of the clean signal (mean ± 95 % CI)
    - D: points (with jitter) and boxes of the NMSE out of the zone by n_false, per class (only mixtures)
    - E: cumulative distribution of the NMSE out of the zone (NMSE = -inf -> NMSE_FLOOR) with references at 1 % and
      0.01 %
    Interactivity: hover with the values (mean, 95 % CI and n; statistics of each box; in D, clean signal, artifact and
    SNR of each mixture; in E, the cumulative proportion); click on the legend to hide/show each category (lines, error
    bars, boxes and points together); zoom in each panel.
    The positions of the axes and texts are taken from the constrained layout of matplotlib applied to the same figure
    without data (_conservation_geometry).
    Limits: the legends are placed where loc="best" puts them with these data (A: lower left; B and D: upper right; E:
    upper left); the plotly legend symbols are not the matplotlib ones (markers and lines at 5-35 px instead of the
    2 em segment); the n= texts of B and D are not hidden with their boxes; the points of D (WebGL) stay over the grid;
    the width of the caps of C is computed for the initial zoom (it does not change when zooming); the jitter of D and
    the bootstrap CIs use a fixed seed (seaborn does not), so they can differ slightly from the matplotlib version.

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 1 (read_results("experiment_1.csv")). It is not modified
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Style constants ===
    CAP = 3                                                             # n_false: grouped as "≥ CAP" from here on
    NMSE_FLOOR = -100                                                   # dB: below it, "no change"
    SOURCES = ["Mixtures", "Control"]
    C_MIX, C_CTRL = "#000000", to_hex("gray")                           # curves of A and E
    PAL_BOX = {"Mixtures": to_hex("darkgray"), "Control": to_hex("whitesmoke")}   # boxes of B
    C_ORANGE, C_LIGHTGRAY = to_hex("tab:orange"), to_hex("lightgray")
    LBL_A = "Mean of NMSE ± CI 95%"
    N_BOOT = 1000


    # === Data (as the matplotlib version) ===
    # copy: results_df is not modified; controls -> NaN
    df = results_df.assign(snr_db=pd.to_numeric(results_df["snr_db"], errors="coerce"),
                           source=np.where(results_df["art_id"] == "Control", "Control", "Mixtures"))

    # A and C: cases with detections out of the zone and finite NMSE
    acted = df[(df["n_false"] > 0) & np.isfinite(df["nmse_out_zone_db"])]
    is_ctrl = acted["source"] == "Control"

    # A: one row per clean signal and SNR (mean over artifacts and positions); the control is already one per signal
    nm_mix = (acted[~is_ctrl].groupby(["clean_id", "snr_db"], as_index=False)["nmse_out_zone_db"].mean()
              .assign(is_control=False))
    nm_ctrl = acted.loc[is_ctrl, ["clean_id", "nmse_out_zone_db"]].assign(is_control=True)
    df_nm = pd.concat([nm_mix, nm_ctrl], ignore_index=True)

    lv = np.sort(nm_mix["snr_db"].unique())                             # SNR levels
    step = np.min(np.diff(lv)) if len(lv) > 1 else 5
    x_ctrl = lv.max() + 1.5 * step                                      # position of the control on the x axis
    df_nm["x"] = np.where(df_nm["is_control"], x_ctrl, df_nm["snr_db"])

    # C: mixtures, one row per clean signal, class and SNR; control, one per clean signal
    lab_mix = (acted[~is_ctrl].groupby(["clean_id", "label", "snr_db"], as_index=False)["nmse_out_zone_db"].mean()
               .assign(snr_cat=lambda d: d["snr_db"].map(lambda v: f"{v:g}")))
    lab_ctrl = acted.loc[is_ctrl, ["clean_id", "label", "nmse_out_zone_db"]].assign(snr_cat="Control")
    df_lab = pd.concat([lab_mix, lab_ctrl], ignore_index=True)
    cat_order = [f"{v:g}" for v in lv] + ["Control"]

    # B and D: all the cases with finite NMSE, grouped by number of detections out of the zone
    nm_all = df[np.isfinite(df["nmse_out_zone_db"])].astype({"n_false": int})
    max_n = nm_all["n_false"].max()
    last = max(2, min(max_n, CAP))                                      # show at least up to 2
    order_n = [str(k) for k in range(last)] + [f"≥ {CAP}" if max_n >= CAP else str(last)]
    nm_all["n_cat"] = np.where(nm_all["n_false"] >= last, order_n[-1], nm_all["n_false"].astype(str))
    nm_lab = nm_all[nm_all["source"] == "Mixtures"].copy()              # D: only mixtures

    # C and D: colors per class of the clean signal
    label_order = sorted(nm_all["label"].unique())
    lab_colors = dict(zip(label_order, [to_hex(c) for c in colormaps["Set2"].colors]))   # = sns.color_palette("Set2")


    # --- Mean and intervals (as seaborn) ---
    rng_ci = np.random.default_rng(1)

    def mean_ci(v):
        """Mean and bootstrap 95 % CI (errorbar=("ci", 95), n_boot=1000 of seaborn; no CI with a single observation)"""
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        if len(v) < 2:
            return (v.mean() if len(v) else np.nan, np.nan, np.nan, len(v))
        boots = v[rng_ci.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
        return (v.mean(), *np.percentile(boots, [2.5, 97.5]), len(v))

    def stat_table(d, by, col):
        """Mean, lower and upper limits of the 95 % CI and n of col per group (index: by)"""
        gb = d.groupby(by)
        index = (pd.Index(list(gb.groups.keys())) if isinstance(by, str)
                 else pd.MultiIndex.from_tuples(list(gb.groups.keys())))
        return pd.DataFrame([mean_ci(gr[col]) for _, gr in gb], index=index, columns=["mean", "lo", "hi", "n"])

    def box_stats(v):
        """Statistics of a box (sns.boxplot: matplotlib boxplot_stats, whis=1.5)"""
        st = boxplot_stats(np.asarray(v, float), whis=1.5)[0]
        return dict(q1=st["q1"], med=st["med"], q3=st["q3"], lo=st["whislo"], hi=st["whishi"], n=len(v))

    # A: mixtures (one point per SNR) and control
    nm_t = stat_table(df_nm, ["is_control", "x"], "nmse_out_zone_db")
    nm_mix_t, nm_ctrl_t = nm_t.loc[False], nm_t.loc[True]

    # B: boxes per n_false category and source (dodge with width=0.8 and two groups -> boxes of 0.4 at ±0.2)
    boxes_b = [dict(cat=cat, group=src, pos=i + dx, w=0.4, **box_stats(sub["nmse_out_zone_db"]))
               for i, cat in enumerate(order_n) for src, dx in zip(SOURCES, (-0.2, 0.2))
               if len(sub := nm_all[(nm_all["n_cat"] == cat) & (nm_all["source"] == src)])]

    # C: mean and CI per class and level (pointplot with dodge=0.6 -> offsets from -0.3 to 0.3)
    lab_t = stat_table(df_lab, ["label", "snr_cat"], "nmse_out_zone_db")
    dodge_c = dict(zip(label_order, np.linspace(-0.3, 0.3, len(label_order)) if len(label_order) > 1 else [0]))

    # D: boxes of 0.8/n per class; points with jitter (jitter=0.33 x width of the category / n classes)
    w_d = 0.8 / len(label_order)
    offsets_d = dict(zip(label_order, -0.4 + w_d/2 + w_d*np.arange(len(label_order))))
    boxes_d = [dict(cat=cat, group=lab, pos=i + offsets_d[lab], w=w_d, **box_stats(sub["nmse_out_zone_db"]))
               for i, cat in enumerate(order_n) for lab in label_order
               if len(sub := nm_lab[(nm_lab["n_cat"] == cat) & (nm_lab["label"] == lab)])]
    jlim = 0.33 / len(label_order)
    cat_pos = {cat: i for i, cat in enumerate(order_n)}
    nm_lab["x_jitter"] = (nm_lab["n_cat"].map(cat_pos) + nm_lab["label"].map(offsets_d)
                          + np.random.default_rng(0).uniform(-jlim, jlim, len(nm_lab)))

    # E: cumulative distribution (steps) per source; NMSE = -inf (no change) -> NMSE_FLOOR
    ecdfs = {}
    for src in SOURCES:
        v = df.loc[df["source"] == src, "nmse_out_zone_db"].replace(-np.inf, NMSE_FLOOR).dropna().to_numpy()
        xs, cnt = np.unique(v, return_counts=True)
        ecdfs[src] = (np.r_[xs[0], xs], np.r_[0, np.cumsum(cnt) / len(v)])


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dash = lambda lw: f"{pt(3.7*lw):.1f}px,{pt(1.6*lw):.1f}px"                    # "--" line of width lw (pt)
    vals = lambda t: t[["mean", "lo", "hi"]].to_numpy()

    # Limits (matplotlib autoscale: data + 5 %; categorical axes: -0.5 to n - 0.5)
    x_range_a = mpl_range(lv, [x_ctrl])
    y_range_a = mpl_range(vals(nm_t))
    x_range_n = [-0.5, len(order_n) - 0.5]
    y_range_b = [NMSE_FLOOR - 5, mpl_range([b["lo"] for b in boxes_b], [b["hi"] for b in boxes_b])[1]]
    x_range_c = [-0.5, len(cat_order) - 0.5]
    y_range_c = mpl_range(vals(lab_t))
    y_range_d = [NMSE_FLOOR - 5, nm_lab["nmse_out_zone_db"].max() + 30]
    x_range_e, y_range_e = [NMSE_FLOOR - 2, 2], [0, 1]

    xticks_a = (list(lv) + [x_ctrl], [f"{v:g}" for v in lv] + ["Control"])
    xticks_n = (list(range(len(order_n))), order_n)
    xticks_c = (list(range(len(cat_order))), cat_order)
    spec = dict(
        suptitle="Conservation",
        A=dict(title="NMSE out of zone vs SNR", xlabel="SNR (dB)", ylabel="NMSE out of zone (dB)", xlim=x_range_a,
               ylim=y_range_a, xticks=xticks_a, legend=[r"Mean of NMSE $\pm$ CI 95%"], legend_kw=dict(loc="lower left")),
        B=dict(title="NMSE out of zone vs false positives", xlabel="Detections out of zone",
               ylabel="NMSE out of zone (dB)", xlim=x_range_n, ylim=y_range_b, xticks=xticks_n, legend=SOURCES,
               legend_kw=dict(loc="upper right")),
        C=dict(title="NMSE out of zone vs SNR per class", xlabel="SNR (dB)", ylabel="NMSE out of zone (dB)",
               xlim=x_range_c, ylim=y_range_c, xticks=xticks_c, title_pad=45, legend=label_order,
               legend_title="Signal class", legend_kw=dict(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))),
        D=dict(title="NMSE out of zone vs false positives per class", xlabel="Detections out of zone",
               ylabel="NMSE out of zone (dB)", xlim=x_range_n, ylim=y_range_d, xticks=xticks_n, legend=label_order,
               legend_title="Signal class", legend_kw=dict(loc="upper right")),
        E=dict(title="Distribution of the distortion out of the artifact zone", xlabel="NMSE out of zone (dB)",
               ylabel="Cumulative proportion", xlim=x_range_e, ylim=y_range_e, legend=SOURCES,
               legend_kw=dict(loc="upper left")),
    )

    fig_w, fig_h = 14, 11
    geo = _conservation_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(6 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _conservation_geometry(fig_w, fig_h, dpi, spec)
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H
    dom_x = lambda b: [X(b["x0"]), X(b["x1"])]
    dom_y = lambda b: [Y(b["bottom"]), Y(b["top"])]
    ax_len_pt = lambda b, k: (b["x1"] - b["x0"] if k == "x" else b["bottom"] - b["top"]) / PT
    g = {k: geo[k]["ax"] for k in "ABCDE"}

    def nticks(axis_range, b, k):
        """
        plotly nticks that gives the matplotlib ticks (mpl_nticks). plotly rounds range/nticks up to the NEXT
        1-2-5 step when it falls exactly on one (30/6 = 5 -> 10), so in that case one more tick is requested
        """
        n = mpl_nticks(axis_range, ax_len_pt(b, k), 3 if k == "x" else 2)
        rough = (axis_range[1] - axis_range[0]) / n
        return n + 1 if np.isclose(rough / 10**np.floor(np.log10(rough)), [1, 2, 5]).any() else n

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      tickformatstops=MPL_TICKFORMAT, exponentformat="none")      # sns.despine: no top/right axes
    x_style = dict(**axis_style, ticklabelstandoff=round(-2.2 * PT))
    x_cat = {**x_style, "showgrid": False}                                         # categorical axes: no x grid

    def y_axis(k, i, axis_range):
        return dict(domain=dom_y(g[k]), anchor=f"x{i}", range=axis_range, nticks=nticks(axis_range, g[k], "y"),
                    **axis_style)

    layout = dict(
        # A (the control, to the right of the last SNR)
        xaxis=dict(domain=dom_x(g["A"]), anchor="y", range=x_range_a, tickvals=xticks_a[0], ticktext=xticks_a[1],
                   **x_style),
        yaxis=y_axis("A", "", y_range_a),
        # B, C and D: one n_false / SNR level per position
        xaxis2=dict(domain=dom_x(g["B"]), anchor="y2", range=x_range_n, tickvals=xticks_n[0], ticktext=xticks_n[1],
                    **x_cat),
        yaxis2=y_axis("B", 2, y_range_b),
        xaxis3=dict(domain=dom_x(g["C"]), anchor="y3", range=x_range_c, tickvals=xticks_c[0], ticktext=xticks_c[1],
                    **x_cat),
        yaxis3=y_axis("C", 3, y_range_c),
        xaxis4=dict(domain=dom_x(g["D"]), anchor="y4", range=x_range_n, tickvals=xticks_n[0], ticktext=xticks_n[1],
                    **x_cat),
        yaxis4=y_axis("D", 4, y_range_d),
        # E
        xaxis5=dict(domain=dom_x(g["E"]), anchor="y5", range=x_range_e, nticks=nticks(x_range_e, g["E"], "x"),
                    **x_style),
        yaxis5=y_axis("E", 5, y_range_e),
    )

    # Legends: one plotly legend per item, at the position of the matplotlib text (the text of a plotly item
    # starts 40 px to the right of the x of the legend), and a common frame with rounded corners. The titles
    # of C and D are bold annotations
    legends, frames, n_leg = {}, [], 0
    annotations = []

    def legend_items(key, names, title=None):
        nonlocal n_leg
        ids = {}
        for name, t in zip(names, geo[key]["legend_texts"]):
            n_leg += 1
            lid = "legend" if n_leg == 1 else f"legend{n_leg}"
            legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)
            ids[name] = lid
        if title:
            t = geo[key]["legend_title"]
            annotations.append(dict(text=f"<b>{title}</b>", x=X((t["x0"] + t["x1"]) / 2),
                                    y=Y((t["top"] + t["bottom"]) / 2 - 0.1*em), xref="paper", yref="paper",
                                    xanchor="center", yanchor="middle", showarrow=False, font=dict(size=em)))
        b = geo[key]["legend"]
        frames.append(dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                           line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H)))
        return ids

    leg_A = legend_items("A", [LBL_A])
    leg_B = legend_items("B", SOURCES)
    leg_C = legend_items("C", label_order, title="Signal class")
    leg_D = legend_items("D", label_order, title="Signal class")
    leg_E = legend_items("E", SOURCES)

    fig = go.Figure()
    shapes = []

    def none_join(segments):
        """Segments [(x, y), ...] -> x, y of a single trace with None between segments"""
        xs, ys = [], []
        for sx, sy in segments:
            xs += [*sx, None]
            ys += [*sy, None]
        return xs, ys

    def err_traces(tab, x, color, lw, cap_px, cap_lw=None, **kw):
        """
        Error bars of the CI (over the markers, as seaborn) with caps of cap_px px half-width. With cap_lw (caps of
        another width, as err_style="bars" of lineplot) two traces are drawn: caps and bars
        """
        lo, hi = (tab["mean"] - tab["lo"]).to_numpy(), (tab["hi"] - tab["mean"]).to_numpy()
        ok = ~np.isnan(lo)
        traces = []
        for w, t_lw in ([(cap_px, cap_lw), (0, lw)] if cap_lw else [(cap_px, lw)]):
            traces.append(go.Scatter(x=np.asarray(x)[ok], y=tab["mean"].to_numpy()[ok], mode="markers",
                                     marker=dict(size=0, opacity=0), hoverinfo="skip", showlegend=False,
                                     error_y=dict(type="data", symmetric=False, array=hi[ok], arrayminus=lo[ok],
                                                  color=color, thickness=pt(t_lw), width=w), **kw))
        return traces

    def vline(x, axis, color, lw, dashed=False, layer="below"):
        """Vertical line across the whole height of the panel (axvline)"""
        return dict(type="line", xref=f"x{axis}", yref=f"y{axis} domain", x0=x, x1=x, y0=0, y1=1, layer=layer,
                    line=dict(color=color, width=pt(lw), dash=dash(lw) if dashed else None))

    def box_hover(b, what):
        return (f"{b['group']}<br>{what}<br>n = {b['n']}<br>Median: {b['med']:.2f} dB<br>"
                f"Q1 / Q3: {b['q1']:.2f} / {b['q3']:.2f} dB<br>Whiskers: {b['lo']:.2f} / {b['hi']:.2f} dB")

    def box_segments(b):
        """Whiskers, caps (half the width of the box) and median of a box"""
        p, hw, cw = b["pos"], b["w"] / 2, b["w"] / 4
        return [([p, p], [b["q1"], b["lo"]]), ([p, p], [b["q3"], b["hi"]]), ([p - cw, p + cw], [b["lo"]] * 2),
                ([p - cw, p + cw], [b["hi"]] * 2), ([p - hw, p + hw], [b["med"]] * 2)]

    def box_outline(b):
        p, hw = b["pos"], b["w"] / 2
        return [p - hw, p + hw, p + hw, p - hw, p - hw], [b["q1"], b["q1"], b["q3"], b["q3"], b["q1"]]

    def legend_box(name, fill, edge, lw, legend, group, axis):
        """Empty trace whose legend symbol is a rectangle with an edge (as the one of the seaborn boxes)"""
        return go.Bar(x=[None], y=[None], name=name, marker=dict(color=fill, line=dict(color=edge, width=pt(lw))),
                      legend=legend, legendgroup=group, xaxis=f"x{axis}", yaxis=f"y{axis}")

    def count_labels(boxes, axis, y, rotate=False):
        """
        Number of cases above each box (text in axes coordinates, va="top"). The box of the plotly annotation is taller
        than the one of the matplotlib text: it is raised 0.36 em (0.27 em the rotated text)
        """
        return [dict(text=f"n={b['n']}", x=b["pos"], y=y, xref=f"x{axis}", yref=f"y{axis} domain", xanchor="center",
                     yanchor="top", yshift=(0.27 if rotate else 0.36) * pt(8), showarrow=False, font=dict(size=pt(8)),
                     textangle=-90 if rotate else 0, borderpad=0) for b in boxes]

    hover_ci = lambda first: (f"{first}<br>Mean NMSE: %{{y:.2f}} dB<br>CI 95%: (%{{customdata[0]:.2f}}, "
                              "%{customdata[1]:.2f}) dB<br>n = %{customdata[2]} clean signals<extra></extra>")

    # --- A: NMSE out of the zone vs SNR (+ control) ---
    shapes.append(vline((lv.max() + x_ctrl) / 2, "", C_LIGHTGRAY, 1))         # separates the control from the SNRs
    for tab, color, name, first in [(nm_mix_t, C_MIX, LBL_A, "Mixtures<br>SNR: %{x:g} dB"),
                                    (nm_ctrl_t, C_CTRL, "Control", "Control")]:
        x = tab.index.to_numpy(float)
        fig.add_trace(go.Scatter(x=x, y=tab["mean"], mode="lines+markers", name=name,
                                 line=dict(color=color, width=pt(1.5)),
                                 marker=dict(size=pt(6), color=color, line=dict(color="#000000", width=pt(0.75))),
                                 customdata=tab[["lo", "hi", "n"]].to_numpy(), hovertemplate=hover_ci(first),
                                 showlegend=name == LBL_A, legend=leg_A[LBL_A], legendgroup="A", xaxis="x", yaxis="y"))
        fig.add_traces(err_traces(tab, x, color, 1.5, pt(4), cap_lw=1, legend=leg_A[LBL_A], legendgroup="A",
                                  xaxis="x", yaxis="y"))

    # --- B: NMSE out of the zone by n_false ---
    for b in boxes_b:
        src = b["group"]
        bx, by = box_outline(b)
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", fill="toself", fillcolor=PAL_BOX[src],
                                 line=dict(color="#000000", width=pt(1)), name=src, hoveron="fills",
                                 text=box_hover(b, f"Detections out of zone: {b['cat']}"), hoverinfo="text",
                                 showlegend=False, legend=leg_B[src], legendgroup=f"B_{src}", xaxis="x2", yaxis="y2"))
        sx, sy = none_join(box_segments(b))
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color="#000000", width=pt(1)), hoverinfo="skip",
                                 showlegend=False, legend=leg_B[src], legendgroup=f"B_{src}", xaxis="x2", yaxis="y2"))
    for src in SOURCES:                                                 # legend items (rectangle with an edge)
        fig.add_trace(legend_box(src, PAL_BOX[src], "#000000", 1, leg_B[src], f"B_{src}", 2))
    annotations += count_labels(boxes_b, 2, 0.95)

    # --- C: NMSE out of the zone vs SNR, per class ---
    # Separation between SNR levels (dashed) and between the last SNR and the control (solid)
    for k in range(len(cat_order) - 1):
        is_ctrl_sep = k == len(lv) - 1
        shapes.append(vline(k + 0.5, 3, to_hex("gray") if is_ctrl_sep else C_LIGHTGRAY, 1.5, dashed=not is_ctrl_sep))

    px_per_x = (g["C"]["x1"] - g["C"]["x0"]) / (x_range_c[1] - x_range_c[0])      # caps of 0.08 (x units)
    for lab in label_order:
        tab = lab_t.loc[lab].reindex([c for c in cat_order if c in lab_t.loc[lab].index])
        x = np.array([cat_order.index(c) for c in tab.index]) + dodge_c[lab]
        snr = [f"{c} dB" if c != "Control" else c for c in tab.index]
        fig.add_trace(go.Scatter(x=x, y=tab["mean"], mode="markers", name=lab,
                                 marker=dict(size=pt(6), color=lab_colors[lab], line=dict(color="#000000", width=pt(1))),
                                 customdata=[[*r, s] for r, s in zip(tab[["lo", "hi", "n"]].to_numpy().tolist(), snr)],
                                 hovertemplate=hover_ci(f"{lab}<br>SNR: %{{customdata[3]}}"),
                                 legend=leg_C[lab], legendgroup=f"C_{lab}", xaxis="x3", yaxis="y3"))
        fig.add_traces(err_traces(tab, x, lab_colors[lab], 1.5, 0.04 * px_per_x, legend=leg_C[lab],
                                  legendgroup=f"C_{lab}", xaxis="x3", yaxis="y3"))

    # --- D: NMSE out of the zone by n_false, per class ---
    # Points (below) and unfilled boxes (above), all in WebGL so that the boxes stay over the points
    for lab in label_order:
        d = nm_lab[nm_lab["label"] == lab]
        fig.add_trace(go.Scattergl(
            x=d["x_jitter"], y=d["nmse_out_zone_db"], mode="markers", name=lab, xaxis="x4", yaxis="y4",
            marker=dict(size=pt(2), color=rgba(lab_colors[lab], 0.1), line=dict(width=0)),
            customdata=d[["n_false", "snr_db", "clean_file", "art_id"]].to_numpy(float),
            hovertemplate=(f"{lab}<br>Detections out of zone: %{{customdata[0]:.0f}}<br>NMSE: %{{y:.2f}} dB<br>"
                           "SNR: %{customdata[1]:g} dB<br>Clean signal: %{customdata[2]:.0f}<br>"
                           "Artifact: %{customdata[3]:.0f}<extra></extra>"),
            showlegend=False, legend=leg_D[lab], legendgroup=f"D_{lab}"))
    for b in boxes_d:
        lab = b["group"]
        sx, sy = none_join([box_outline(b), *box_segments(b)])
        fig.add_trace(go.Scattergl(x=sx, y=sy, mode="lines", line=dict(color=lab_colors[lab], width=pt(1.2)),
                                   text=box_hover(b, f"Detections out of zone: {b['cat']}"), hoverinfo="text",
                                   showlegend=False, legend=leg_D[lab], legendgroup=f"D_{lab}", xaxis="x4", yaxis="y4"))
    for lab in label_order:                                             # legend items (unfilled box)
        fig.add_trace(legend_box(lab, "rgba(0, 0, 0, 0)", lab_colors[lab], 1.2, leg_D[lab], f"D_{lab}", 4))
    annotations += count_labels(boxes_d, 4, 0.98, rotate=True)

    # --- E: cumulative distribution of the NMSE out of the zone ---
    for src, color in zip(SOURCES, (C_MIX, C_CTRL)):
        xs, ys = ecdfs[src]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=pt(2), shape="hv"), name=src,
                                 hovertemplate=(f"{src}<br>NMSE out of zone: %{{x:.2f}} dB<br>"
                                                "Cumulative proportion: %{y:.3f}<extra></extra>"),
                                 legend=leg_E[src], legendgroup=f"E_{src}", xaxis="x5", yaxis="y5"))

    # References: error of 1 % and 0.01 % of the signal power
    space = mpl_text_size(" ", 8, dpi)[0]
    for thr, txt in ((-20, "1 %"), (-40, "0.01 %")):
        shapes.append(vline(thr, 5, C_ORANGE, 2, dashed=True, layer="above"))
        annotations.append(dict(text=txt, x=thr, y=0.02, xref="x5", yref="y5", xanchor="left", yanchor="bottom",
                                xshift=space, showarrow=False, font=dict(size=pt(8), color=C_ORANGE), borderpad=0))

    # --- Titles and labels ---
    def ann(s, b, size, rotate=False):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size)), textangle=-90 if rotate else 0)

    annotations.append(ann(spec["suptitle"], geo["suptitle"], 16))
    for k in "ABCDE":
        s = spec[k]
        annotations += [ann(f"<b>{s['title']}</b>", geo[k]["title"], 14), ann(s["xlabel"], geo[k]["xlabel"], 12),
                        ann(s["ylabel"], geo[k]["ylabel"], 12, rotate=True)]

    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes + frames, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )
    return fig


# --------------------------------------------------------------------------------------------------------------------
# 4.4. Amplitude (section 5.4.4)
# --------------------------------------------------------------------------------------------------------------------

def _amplitude_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_amplitude figure as matplotlib computes it (layout="constrained" with rect=(0, 0, 1, 0.9)):
    builds the same figure WITHOUT data (titles, limits, insets, texts of the figure and legend) and returns the boxes
    (px from the top-left corner of the image cropped as bbox_inches="tight") of the axes, the insets, the texts and
    the legend.
    spec: {"panels": {key: dict(title, ylim, inset_ylim)} (in the order of axs.flat), "xlim", "inset_xlim",
    "inset_pos", "legend", "legend_title", "suptitle", "supxlabel", "supylabel"}
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    axs = dict(zip(spec["panels"], fig.subplots(2, 2, sharex=True).flat))

    insets = {}
    for k, ax in axs.items():
        s = spec["panels"][k]
        ax.set_title(s["title"], fontsize=14, fontweight="bold")
        ax.set_xlim(*spec["xlim"])
        ax.set_ylim(*s["ylim"])
        ax.tick_params(labelbottom=True)
        ax.spines[["top", "right"]].set_visible(False)                              # sns.despine
        ins = insets[k] = ax.inset_axes(spec["inset_pos"])
        ins.set_xlim(*spec["inset_xlim"])
        ins.set_ylim(*s["inset_ylim"])
        ins.tick_params(labelsize=8)

    fig.supxlabel(spec["supxlabel"], fontsize=12)
    fig.supylabel(spec["supylabel"], fontsize=12)
    fig.suptitle(spec["suptitle"], fontsize=16, y=0.995, va="top")
    leg = fig.legend([Line2D([], [], marker="o") for _ in spec["legend"]], spec["legend"], loc="upper center",
                     bbox_to_anchor=(0.5, 0.97), ncol=len(spec["legend"]), fontsize=10, title=spec["legend_title"],
                     title_fontproperties=dict(weight="bold"))
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.9))

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle),
               supxlabel=box(fig._supxlabel), supylabel=box(fig._supylabel), legend=box(leg),
               legend_texts=[box(t) for t in leg.get_texts()], legend_title=box(leg.get_title()))
    for k, ax in axs.items():
        ins = insets[k]
        out[k] = dict(ax=box(ax), title=box(ax.title), inset=box(ins))
        # y axis of the inset with the matplotlib offset notation (e.g. "1e−8" with very small values): ticks,
        # their labels and the text of the offset
        ot = ins.yaxis.get_offset_text()
        if ot.get_text():
            lo, hi = ins.get_ylim()
            ticks = [(v, t.get_text()) for v, t in zip(ins.get_yticks(), ins.get_yticklabels()) if lo <= v <= hi]
            out[k]["inset_offset"] = dict(text=ot.get_text(), box=box(ot), size=ot.get_fontsize(),
                                          tickvals=[v for v, _ in ticks], ticktext=[t for _, t in ticks])
    return out


def plot_amplitude(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_amplitude (matplotlib/seaborn, section 5.4.4 of evaluation_denoise.ipynb): same
    layout (layout="constrained", figsize 14 x 8 in, common legend on top), sizes, fonts, colors and positions. The
    figure is cropped as the inline backend does (bbox_inches="tight"), so its size matches the matplotlib image.
    - Four panels (maximum / mean RMS, local / global): RMS error before (mixture) and after (output) the algorithm vs
      SNR, and after only for the detected artifacts, median ± IQR; in the global ones, also the control (median and
      IQR band)
    - In each panel, an inset (red frame) of the three curves over the whole SNR range, with the y axis fitted to the
      medians of after and of after for the detected ones (+ 50 % margin; the IQR bands and the curve of before are
      clipped), joined with arms to the zoomed area
    Interactivity: hover with the median, the IQR and n of each point (in the panels and in the insets); click on the
    legend to hide/show each curve (line, band and insets together; the control, line and band); zoom in each panel
    (the x axis is common to the four, as sharex) and in each inset (independent).
    The positions of the axes, insets and texts are taken from the constrained layout of matplotlib applied to the same
    figure without data (_amplitude_geometry).
    Limits: the rectangle of the zoomed area and the arms follow the zoom of the panel, but the arms do not follow the
    zoom of the inset (their ends on the frame are the initial ones); plotly does not use steps of 2.5 in the ticks
    (0, 200, 400... instead of 0, 250, 500...); the IQR bands stay over the grid (in matplotlib, below); the plotly
    legend symbols are not the matplotlib ones (line at 5-35 px); if matplotlib uses offset notation on the y axis of
    an inset ("1e−8"), its ticks are fixed (they do not follow the zoom).

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 1 (read_results("experiment_1.csv")). It is not modified
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Constants (as the matplotlib version) ===
    INSET_POS = [0.42, 0.36, 0.55, 0.58]               # position of the inset in the panel (axes fraction)

    # Panel -> (column before, column after, with control?)
    PANELS = {
        "max_loc":   ("err_rms_max_loc_mix", "err_rms_max_loc_out", False),
        "max_glob":  ("err_rms_max_glob_mix", "err_rms_max_glob_out", True),
        "mean_loc":  ("err_rms_mean_loc_mix", "err_rms_mean_loc_out", False),
        "mean_glob": ("err_rms_mean_glob_mix", "err_rms_mean_glob_out", True),
    }
    TITLES = {"max_loc": "Max local RMS", "max_glob": "Max global RMS",
              "mean_loc": "Mean local RMS", "mean_glob": "Mean global RMS"}

    # Curves: (name, column, color, dashed?)
    CURVES = [("Before (mixture)", "before", to_hex("tab:orange"), True),
              ("After (output)", "after", to_hex("tab:blue"), False),
              ("After (detected)", "after_det", to_hex("tab:brown"), False)]
    C_CTRL, C_ZERO, C_ZOOM = to_hex("tab:purple"), to_hex("gray"), to_hex("red")


    # === Data ===
    # copy: results_df is not modified; numeric: the zoom is set in dB
    results_df = results_df.assign(snr_db=pd.to_numeric(results_df["snr_db"], errors="coerce"))

    mix_df = results_df[results_df["art_id"] != "Control"]
    ctrl_df = results_df[results_df["art_id"] == "Control"]

    # whole SNR range (dB): the inset only narrows the y axis
    X_SPAN = (mix_df["snr_db"].min(), mix_df["snr_db"].max())


    def iqr_limits(amp, ctrl=None, margin=0.1, cols=("before", "after"), qs=(0.25, 0.75)):
        """
        Limits of the y axis that contain the quantiles qs (by default, the IQR) of the columns cols (and the control
        band)
        """
        q = amp.groupby("snr_db")[list(cols)].quantile(list(qs))
        lo, hi = q.min().min(), q.max().max()
        if ctrl is not None:
            lo, hi = min(lo, ctrl.quantile(0.25)), max(hi, ctrl.quantile(0.75))
        pad = margin * (hi - lo if hi > lo else 1)
        return lo - pad, hi + pad


    # Median, IQR and n per SNR level of each curve (as sns.lineplot with estimator="median", errorbar=("pi", 50));
    # control: quartiles; limits of the panels (data + 5 %, including the line at 0) and of the insets
    stats, ctrl_q, ctrl_n, y_ranges, inset_ranges = {}, {}, {}, {}, {}
    for key, (col_mix, col_out, with_ctrl) in PANELS.items():
        amp = mix_df.assign(before=mix_df[col_mix], after=mix_df[col_out],
                            # after, only the detected ones (rest: NaN)
                            after_det=mix_df[col_out].where(mix_df["detected"] == 1))
        gb = amp.groupby("snr_db")
        stats[key] = {col: pd.DataFrame({"med": gb[col].median(), "q25": gb[col].quantile(0.25),
                                         "q75": gb[col].quantile(0.75), "n": gb[col].count()})
                      for _, col, _, _ in CURVES}
        ctrl = ctrl_df[col_out] if with_ctrl else None
        if with_ctrl:
            ctrl_q[key], ctrl_n[key] = ctrl.quantile([0.25, 0.5, 0.75]).to_numpy(), len(ctrl)
        y_ranges[key] = mpl_range(*[s.to_numpy()[:, :3] for s in stats[key].values()], [0], ctrl_q.get(key, [0]))
        inset_ranges[key] = list(iqr_limits(amp, cols=["after", "after_det"], qs=[0.5], margin=0.5))   # medians + 50 %
    lv = np.sort(mix_df["snr_db"].unique())                             # SNR levels
    x_range = mpl_range(lv)

    LBL_CTRL = f"Control (n = {ctrl_n['mean_glob']})"                  # matplotlib legend: the one of mean_glob
    legend_labels = [c[0] for c in CURVES] + [LBL_CTRL]


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dash = lambda lw: f"{pt(3.7*lw):.1f}px,{pt(1.6*lw):.1f}px"                    # "--" line of width lw (pt)

    spec = dict(panels={k: dict(title=TITLES[k], ylim=y_ranges[k], inset_ylim=inset_ranges[k]) for k in PANELS},
                xlim=x_range, inset_xlim=x_range, inset_pos=INSET_POS, legend=legend_labels, legend_title="Median ± IQR",
                suptitle="Amplitude", supxlabel="SNR (dB)", supylabel="RMS error (%)")

    fig_w, fig_h = 14, 8
    geo = _amplitude_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(6 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _amplitude_geometry(fig_w, fig_h, dpi, spec)
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H
    dom_x = lambda b: [X(b["x0"]), X(b["x1"])]
    dom_y = lambda b: [Y(b["bottom"]), Y(b["top"])]
    ax_len_pt = lambda b, k: (b["x1"] - b["x0"] if k == "x" else b["bottom"] - b["top"]) / PT

    def nticks(axis_range, b, k, fontsize=10):
        """
        plotly nticks that gives the matplotlib ticks (mpl_nticks). plotly rounds range/nticks up to the NEXT
        1-2-5 step when it falls exactly on one (30/6 = 5 -> 10), so in that case one more tick is requested
        """
        n = mpl_nticks(axis_range, ax_len_pt(b, k), 3 if k == "x" else 2, fontsize)
        rough = (axis_range[1] - axis_range[0]) / n
        return n + 1 if np.isclose(rough / 10**np.floor(np.log10(rough)), [1, 2, 5]).any() else n

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      tickformatstops=MPL_TICKFORMAT, exponentformat="none")      # sns.despine: no top/right axes
    inset_style = {**axis_style, "mirror": True, "linecolor": C_ZOOM, "linewidth": pt(1.2),
                   "tickfont": dict(size=pt(8))}                                   # red frame, 8 pt ticks

    # Axes: panels 1-4 (in the order of PANELS) and insets 5-8
    keys = list(PANELS)
    ax_id = {k: ("" if i == 0 else str(i + 1)) for i, k in enumerate(keys)}
    ins_id = {k: str(i + 5) for i, k in enumerate(keys)}
    layout = {}
    for k in keys:
        b, bi = geo[k]["ax"], geo[k]["inset"]
        a, i = ax_id[k], ins_id[k]
        layout[f"xaxis{a}"] = dict(domain=dom_x(b), anchor=f"y{a}", range=x_range, nticks=nticks(x_range, b, "x"),
                                   ticklabelstandoff=round(-2.2 * PT), **({"matches": "x"} if a else {}), **axis_style)
        layout[f"yaxis{a}"] = dict(domain=dom_y(b), anchor=f"x{a}", range=y_ranges[k],
                                   nticks=nticks(y_ranges[k], b, "y"), **axis_style)
        layout[f"xaxis{i}"] = dict(domain=dom_x(bi), anchor=f"y{i}", range=x_range,
                                   nticks=nticks(x_range, bi, "x", 8), ticklabelstandoff=round(-2.2 * 0.8 * PT),
                                   **inset_style)
        off = geo[k].get("inset_offset")                                           # offset notation ("1e−8"):
        ticks_i = (dict(tickvals=off["tickvals"], ticktext=off["ticktext"]) if off   # matplotlib ticks
                   else dict(nticks=nticks(inset_ranges[k], bi, "y", 8)))
        layout[f"yaxis{i}"] = dict(domain=dom_y(bi), anchor=f"x{i}", range=inset_ranges[k], **ticks_i, **inset_style)

    # Legend: one plotly legend per item, at the position of the matplotlib text (the text of a plotly item starts
    # 40 px to the right of the x of the legend), a frame with rounded corners and the title in bold
    legends, leg_id = {}, {}
    for n, (name, t) in enumerate(zip(legend_labels, geo["legend_texts"]), start=1):
        lid = leg_id[name] = "legend" if n == 1 else f"legend{n}"
        legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                            yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                            font=dict(size=em), groupclick="togglegroup", itemwidth=30)
    b = geo["legend"]
    shapes = [dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                   line=dict(color="#cccccc", width=pt(1)),
                   path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H))]

    fig = go.Figure()

    def hline(y, axis, color, lw, dashed=False, layer="below", **kw):
        """Horizontal line across the whole width of the panel (axhline)"""
        return dict(type="line", xref=f"x{axis} domain", yref=f"y{axis}", x0=0, x1=1, y0=y, y1=y, layer=layer,
                    line=dict(color=color, width=pt(lw), dash=dash(lw) if dashed else None), **kw)

    def draw(key, axis, show_legend, with_ctrl=True):
        """
        Before / after (median ± IQR) and, if any and with_ctrl, the control (median and IQR band), as the matplotlib
        draw
        """
        shapes.append(hline(0, axis, C_ZERO, 2))                                   # zorder=0: below everything
        common = lambda name, col: dict(legend=leg_id[name], legendgroup=col, xaxis=f"x{axis}", yaxis=f"y{axis}")
        for name, col, color, _ in CURVES:                                         # bands (zorder 1): below
            s = stats[key][col]                                                    # all the lines
            x = s.index.to_numpy(float)
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[s["q75"], s["q25"][::-1]], mode="lines",
                                     fill="toself", fillcolor=rgba(color, 0.2), line=dict(width=0), hoverinfo="skip",
                                     showlegend=False, **common(name, col)))
        for name, col, color, dashed in CURVES:
            s = stats[key][col]
            x = s.index.to_numpy(float)
            fig.add_trace(go.Scatter(
                x=x, y=s["med"], mode="lines+markers", name=name,
                line=dict(color=color, width=pt(2), dash=dash(2) if dashed else None),
                marker=dict(size=pt(6), color=color, line=dict(color="#000000", width=pt(0.75))),
                customdata=s[["q25", "q75", "n"]].to_numpy(),
                hovertemplate=(f"{name}<br>SNR: %{{x:g}} dB<br>Median: %{{y:.2f}} %<br>"
                               "IQR: (%{customdata[0]:.2f}, %{customdata[1]:.2f}) %<br>"
                               f"n = %{{customdata[2]:.0f}} mixtures<extra>{TITLES[key]}</extra>"),
                showlegend=show_legend, **common(name, col)))
        if with_ctrl and key in ctrl_q:
            q25, med, q75 = ctrl_q[key]
            ctrl_kw = dict(legend=leg_id[LBL_CTRL], legendgroup="control")
            shapes.append(dict(type="rect", xref=f"x{axis} domain", yref=f"y{axis}", x0=0, x1=1, y0=q25, y1=q75,
                               layer="below", fillcolor=rgba(C_CTRL, 0.2), line=dict(width=0), **ctrl_kw))
            shapes.append(hline(med, axis, C_CTRL, 2, dashed=True, layer="above", name=LBL_CTRL,
                                showlegend=show_legend, **ctrl_kw))                # over the curves, as matplotlib

    for k in keys:
        a, i = ax_id[k], ins_id[k]
        draw(k, a, show_legend=(k == "max_glob"))                                 # whole panel
        draw(k, i, show_legend=False, with_ctrl=False)                             # inset (without the control)

        # Rectangle of the zoomed area (over the curves) and arms: top corners of the rectangle ->
        # bottom corners of the inset (in data coordinates of the panel, with its initial limits)
        y0, y1 = inset_ranges[k]
        shapes.append(dict(type="rect", xref=f"x{a}", yref=f"y{a}", x0=X_SPAN[0], x1=X_SPAN[1], y0=y0, y1=y1,
                           layer="above", line=dict(color=C_ZOOM, width=pt(1.2))))
        b, bi = geo[k]["ax"], geo[k]["inset"]
        y_corner = y_ranges[k][0] + (b["bottom"] - bi["bottom"]) / (b["bottom"] - b["top"]) * np.diff(y_ranges[k])[0]
        for x_data, x_px in ((X_SPAN[0], bi["x0"]), (X_SPAN[1], bi["x1"])):
            x_corner = x_range[0] + (x_px - b["x0"]) / (b["x1"] - b["x0"]) * (x_range[1] - x_range[0])
            shapes.append(dict(type="line", xref=f"x{a}", yref=f"y{a}", x0=x_data, y0=y1, x1=x_corner, y1=y_corner,
                               layer="below", line=dict(color=C_ZOOM, width=pt(1.2))))

    # --- Titles, labels and legend ---
    def ann(s, b, size, rotate=False):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size)), textangle=-90 if rotate else 0)

    annotations = [ann(spec["suptitle"], geo["suptitle"], 16), ann(spec["supxlabel"], geo["supxlabel"], 12),
                   ann(spec["supylabel"], geo["supylabel"], 12, rotate=True),
                   ann(f"<b>{spec['legend_title']}</b>", geo["legend_title"], 10)]
    annotations += [ann(f"<b>{TITLES[k]}</b>", geo[k]["title"], 14) for k in keys]
    annotations += [ann(geo[k]["inset_offset"]["text"], geo[k]["inset_offset"]["box"], geo[k]["inset_offset"]["size"])
                    for k in keys if geo[k].get("inset_offset")]                   # offset of the y axis of the inset

    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )
    return fig


# --------------------------------------------------------------------------------------------------------------------
# 4.5. Factor analysis (section 5.4.5)
# --------------------------------------------------------------------------------------------------------------------

def _factor_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_factor_analysis figure as matplotlib computes it (layout="constrained"): builds the same figure
    WITHOUT data (empty heatmaps, titles, labels, limits, legends and colorbars) and returns the boxes (px from the
    top-left corner of the image cropped as bbox_inches="tight") of the axes, texts, legends and colorbars, the ticks of
    the colorbars and those of the log axis of C.
    spec: dict(art_rows, lv, ds_lim, art_order, lims={panel: (xlim, ylim)}, legends={panel: [labels]},
    c_markers=[(marker, color, label)], g_loc, density_ticks, gamma)
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    n_art = len(spec["art_rows"])
    gs = fig.add_gridspec(4, 2, height_ratios=[0.7 * n_art / 10 + 1, 1, 2, 1])
    fig.suptitle("Factor analysis", fontsize=16)

    ax = {"A": fig.add_subplot(gs[0, 0])}
    ax["B"] = fig.add_subplot(gs[0, 1], sharey=ax["A"])
    ax["C"] = fig.add_subplot(gs[1, 0])
    ax["D"] = fig.add_subplot(gs[1, 1])
    sub_glob = gs[2, 0].subgridspec(2, 1)
    ax["Ed"] = fig.add_subplot(sub_glob[0])
    ax["En"] = fig.add_subplot(sub_glob[1], sharex=ax["Ed"], sharey=ax["Ed"])
    ax["F"] = fig.add_subplot(gs[2, 1], sharex=ax["D"])
    ax["G"] = fig.add_subplot(gs[3, 0])
    ax["H"] = fig.add_subplot(gs[3, 1])
    cbs = {}
    leg_lines = lambda labels, **kw: [Line2D([], [], **kw) for _ in labels]
    curve_kw = dict(marker="o", markeredgecolor="black", lw=2)

    # A and B: empty heatmaps (the annotations stay inside the axes and do not change the layout)
    zeros = pd.DataFrame(0.0, index=spec["art_rows"], columns=spec["lv"])
    sns.heatmap(zeros, cmap="Purples", vmin=0, vmax=1, cbar_kws={"label": "Detection rate", "aspect": 50, "pad": 0.02},
                ax=ax["A"])
    cax_b = ax["B"].inset_axes([1.02, 0, 0.025, 1])
    sns.heatmap(zeros, cmap="RdBu", center=0, vmin=-spec["ds_lim"], vmax=spec["ds_lim"], cbar_ax=cax_b,
                cbar_kws={"label": "Mean ΔSNR (dB)"}, ax=ax["B"])
    for k, title in (("A", "Detection rate"), ("B", "Mean ΔSNR of detected artifacts")):
        cbs[k] = ax[k].collections[0].colorbar
        ax[k].set_xticklabels([f"{v:g}" for v in spec["lv"]], rotation=0)
        ax[k].tick_params(axis="y", rotation=0)
        ax[k].set_xlabel("SNR (dB)", fontsize=12)
        ax[k].set_title(title, fontsize=14, fontweight="bold")
    ax["A"].set_ylabel("Artifact", fontsize=12)
    ax["B"].set_ylabel("")

    # C: legend with the same items (markers of 60 pt²), discrete colorbar and log axis
    a = ax["C"]
    for mk, color, label in spec["c_markers"]:
        a.scatter([], [], marker=mk, s=60, color=color, edgecolor="black", label=label)
    n = len(spec["art_order"])
    cbs["C"] = fig.colorbar(ScalarMappable(cmap=ListedColormap(["k"] * n), norm=BoundaryNorm(np.arange(-0.5, n), n)),
                            ax=a, location="right", aspect=20, pad=0.02)
    tick_idx = np.arange(0, n, 5)
    cbs["C"].set_ticks(tick_idx, labels=[spec["art_order"][i] for i in tick_idx])
    cbs["C"].set_label("Artifact", fontsize=11)
    a.set_xscale("log")
    a.set_xlabel("Artifact duration (ms)", fontsize=12)
    a.set_ylabel("SNR at 50 % detection (dB)", fontsize=12)
    a.set_title("SNR at 50 % detection vs artifact duration", fontsize=14, fontweight="bold", pad=45)
    a.legend(fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    # D
    a = ax["D"]
    a.legend(leg_lines(spec["legends"]["D"], **curve_kw), spec["legends"]["D"], title="SNR", fontsize=10,
             loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, title_fontproperties=dict(weight="bold"))
    a.set_xlabel("Local relative power (dB)", fontsize=12)
    a.set_ylabel("Detection rate", fontsize=12)
    a.set_title("Detection rate vs local signal power", fontsize=14, fontweight="bold")

    # E
    norm = PowerNorm(gamma=spec["gamma"], vmin=0, vmax=1)
    cbs["E"] = fig.colorbar(ScalarMappable(cmap="Greys", norm=norm), ax=[ax["Ed"], ax["En"]], location="right",
                            aspect=40, pad=0.01)
    cbs["E"].set_label("Relative density (opacity)", fontsize=11)
    cbs["E"].set_ticks(spec["density_ticks"])
    ax["Ed"].set_title("Local power vs global SNR", fontsize=14, fontweight="bold")
    for k in ("Ed", "En"):
        ax[k].set_ylabel("Global SNR (dB)", fontsize=12)
    ax["Ed"].tick_params(labelbottom=False)
    ax["En"].set_xlabel("Local relative power (dB)", fontsize=12)

    # F
    a = ax["F"]
    a.legend(leg_lines(spec["legends"]["F"], ls="--", lw=2), spec["legends"]["F"], fontsize=10, loc="lower center",
             bbox_to_anchor=(0.5, 1.0), ncol=2)
    a.tick_params(labelbottom=True)
    a.set_xlabel("Local relative power (dB)", fontsize=12)
    a.set_ylabel("Mean ΔSNR of detected (dB)", fontsize=12)
    a.set_title("ΔSNR of detected artifacts vs local signal power", fontsize=14, fontweight="bold", pad=30)

    # G (loc="best" resolved with the data)
    a = ax["G"]
    a.legend(leg_lines(spec["legends"]["G"], **curve_kw), spec["legends"]["G"], fontsize=10, loc=spec["g_loc"],
             title="Median ± IQR", title_fontproperties=dict(weight="bold"))
    a.set_xlabel("Global SNR (dB)", fontsize=12)
    a.set_ylabel("Global RMS peak error (%)", fontsize=12)
    a.set_title("Global peak error before / after vs global SNR", fontsize=14, fontweight="bold")

    # H
    a = ax["H"]
    cbs["H"] = fig.colorbar(ScalarMappable(cmap="Blues", norm=norm), cax=a.inset_axes([1.02, 0, 0.025, 1]))
    cbs["H"].set_label("Relative density", fontsize=11)
    cbs["H"].set_ticks(spec["density_ticks"])
    a.legend([Line2D([], [], **curve_kw)] + leg_lines(spec["legends"]["H"][1:], ls="--", lw=2), spec["legends"]["H"],
             fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, borderaxespad=0.2)
    a.set_xlabel("ΔSNR in the artifact zone (dB)", fontsize=12)
    a.set_ylabel("NMSE out of zone (dB)", fontsize=12)
    a.set_title("Removal vs conservation (detected)", fontsize=14, fontweight="bold", pad=30)

    for k, (xlim, ylim) in spec["lims"].items():
        ax[k].set_xlim(*xlim)
        ax[k].set_ylim(*ylim)

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    def ticks(axis):
        """Positions and texts of the labelled ticks within the limits (major and minor)"""
        lo, hi = sorted(axis.get_view_interval())
        out = [(t.get_loc(), t.label1.get_text()) for t in axis.get_major_ticks() + axis.get_minor_ticks()]
        return [(v, s) for v, s in out if s and lo - 1e-9 <= v <= hi + 1e-9]

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle),
               c_ticks=ticks(ax["C"].xaxis), c_minor=list(ax["C"].xaxis.get_minorticklocs()))
    for k, a in ax.items():
        out[k] = dict(ax=box(a), **{name: box(t) for name, t in (("title", a.title), ("xlabel", a.xaxis.label),
                                                                  ("ylabel", a.yaxis.label)) if t.get_text()})
        if leg := a.get_legend():
            out[k].update(legend=box(leg), legend_texts=[box(t) for t in leg.get_texts()],
                          legend_title=box(leg.get_title()))
    for k, cb in cbs.items():
        out["cb" + k] = dict(ax=box(cb.ax), label=box(cb.ax.yaxis.label), ticks=ticks(cb.ax.yaxis))
    return out


def plot_factor_analysis(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_factor_analysis (matplotlib/seaborn, section 5.4.5 of
    evaluation_denoise.ipynb): same layout (layout="constrained", figsize 14 x (18 + 0.12·n_art) in), sizes, fonts,
    colors and positions. The figure is cropped as the inline backend does (bbox_inches="tight"), so its size matches
    the matplotlib image.
    - A: detection rate per artifact and SNR (rows sorted by SNR at 50 %)
    - B: mean ΔSNR of the detected artifacts per artifact and SNR (hatched: no detections)
    - C: SNR at 50 % detection vs duration of the artifact (▲ / ▼: out of the SNR range)
    - D: detection rate vs local relative power (deciles), mean ± 95 % CI per SNR
    - E: local power vs global SNR of the detected and not detected mixtures (hexagons: color = SNR, opacity = density)
    - F: ΔSNR of the detected artifacts vs local relative power, mean ± 95 % CI per SNR
    - G: error of the global peak before / after vs global SNR (median ± IQR)
    - H: ΔSNR vs NMSE out of the zone of the detected artifacts (density and median ± IQR)
    Interactivity: hover with the values of each cell, point and hexagon (in E, number of mixtures of each SNR); click
    on the legend of D to hide/show an SNR in D, E and F together, and on the other legends for each curve or
    reference; zoom in each panel (A and B share the y axis; D and F, the x axis; the two subpanels of E, both axes).
    The positions of the axes, texts, legends and colorbars are taken from the constrained layout of matplotlib applied
    to the same figure without data (_factor_geometry).
    Limits: the opacity of the hexagons of E is grouped in 32 levels (matplotlib: 256; unnoticeable difference); in H
    the grid stays below the hexagons (in matplotlib, above); the ticks of the log axis of C are the initial ones (they
    do not change when zooming) and all of them have the length of the major ones; the text boxes of E do not have
    rounded corners; the plotly legend symbols are not the matplotlib ones; the bootstrap CIs of D and F use a fixed
    seed (seaborn does not), so they can differ slightly from the matplotlib version.

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 1 (read_results("experiment_1.csv")). It is not modified
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Constants (as the matplotlib version) ===
    N_BINS = 10                                     # D and F: groups of local relative power (deciles)
    GRID = (60, 30)                                 # E: number of hexagons (x, y)
    GAMMA = 0.5                                     # E, H: density in power scale (1: linear; < 1 boosts sparse areas)
    DENSITY_TICKS = [0, 0.1, 0.25, 0.5, 0.75, 1]    # E and H: ticks of the density colorbars
    BIN_W = 5                                       # G: dB, width of the global SNR bins
    DS_BIN = 2                                      # H: dB, width of the ΔSNR bins
    MIN_N = 30                                      # G and H: minimum number of mixtures per bin
    NMSE_FLOOR = -100                               # H: dB, below = no change out of the zone
    N_BOOT = 1000                                   # D and F: resamples of the 95 % CI (as seaborn)
    E_LEVELS = 32                                   # E: opacity levels of the hexagons (traces per SNR)
    C_RED, C_ORANGE, C_BLUE, C_GRAY = (to_hex(c) for c in ("tab:red", "tab:orange", "tab:blue", "gray"))
    C_LIGHTGRAY = to_hex("lightgray")


    # === Helper functions ===
    def snr_at_50(rate, levels):
        """SNR at which the rate crosses 0.5 (linear interpolation). +inf / -inf: above / below the range"""
        above = rate >= 0.5
        if not above.any():
            return -np.inf
        i = np.where(above)[0].max()
        if i == len(rate) - 1:
            return np.inf
        r0, r1 = rate[i], rate[i + 1]
        return levels[i] + (0.5 - r0) * (levels[i + 1] - levels[i]) / (r1 - r0)

    rng_ci = np.random.default_rng(0)

    def mean_ci(v):
        """Mean and bootstrap 95 % CI (errorbar=("ci", 95), n_boot=1000 of seaborn; no CI with a single observation)"""
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        if len(v) < 2:
            return (v.mean() if len(v) else np.nan, np.nan, np.nan, len(v))
        boots = v[rng_ci.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
        return (v.mean(), *np.percentile(boots, [2.5, 97.5]), len(v))

    def median_iqr(v):
        """Median and 25th and 75th percentiles (estimator="median", errorbar=("pi", 50) of seaborn)"""
        v = np.asarray(v, float)
        return (np.nanmedian(v), *np.nanpercentile(v, [25, 75]), np.isfinite(v).sum())

    def curve_table(d, by, col, func):
        """Estimate, lower and upper limits and n of col per group (index: by, sorted)"""
        rows = {key: func(gr[col]) for key, gr in d.groupby(by)}
        return pd.DataFrame(rows.values(), index=pd.Index(list(rows)), columns=["est", "lo", "hi", "n"])

    def hexbin(x, y, gridsize, extent=None):
        """
        Non-empty hexagons of Axes.hexbin(mincnt=1): centers, number of mixtures, vertices of the hexagon (relative to
        the center) and data limits that hexbin adds to the autoscale
        """
        ax = Figure().subplots()
        h = ax.hexbin(x, y, gridsize=gridsize, extent=extent, mincnt=1)
        return h.get_offsets(), h.get_array().astype(float), h.get_paths()[0].vertices[:6], ax.dataLim

    def hex_polygons(centers, hexagon):
        """Polygons of the hexagons (x, y with None between hexagons) for a trace with fill="toself" """
        ring = np.r_[hexagon, hexagon[:1]]
        pts = centers[:, None, :] + ring[None]                                     # (hexagon, vertex, xy)
        pts = np.concatenate([pts, np.full((len(pts), 1, 2), np.nan)], axis=1).reshape(-1, 2)
        return [None if np.isnan(v) else v for v in pts[:, 0]], [None if np.isnan(v) else v for v in pts[:, 1]]


    # === Data (as the matplotlib version) ===
    results_df = results_df.assign(                                          # copy: results_df is not modified
        snr_db=pd.to_numeric(results_df["snr_db"], errors="coerce"),
        detected=(results_df["detected"].astype(str).str.strip().str.lower()
                  .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})))   # controls -> NaN

    mix_df = results_df[results_df["art_id"] != "Control"]
    det_df = mix_df[mix_df["detected"] == 1]                                 # detected mixtures (B, E, F, G, H)

    # A: detection rate per artifact and SNR; rows sorted by SNR at 50 % (most sensitive on top)
    det_rate = mix_df.pivot_table(index="art_id", columns="snr_db", values="detected", aggfunc="mean")
    lv = det_rate.columns.to_numpy()                                         # SNR levels
    snr50 = pd.Series([snr_at_50(r, lv) for r in det_rate.to_numpy()], index=det_rate.index)
    art_rows = snr50.replace({np.inf: 1e9, -np.inf: -1e9}).sort_values(ascending=False).index
    det_rate = det_rate.loc[art_rows]

    # B: mean ΔSNR of the detected ones (same order as A; no detections -> NaN)
    ds_mean = (det_df.pivot_table(index="art_id", columns="snr_db", values="dsnr_db", aggfunc="mean")
               .reindex(index=art_rows, columns=lv))
    ds_lim = np.nanmax(np.abs(ds_mean.to_numpy()))                          # symmetric scale around 0

    # C: SNR at 50 % vs duration (out of range: half a step above ▲ or below ▼)
    dur = mix_df.groupby("art_id")["art_duration_ms"].first().reindex(snr50.index)
    art_order = sorted(snr50.index, key=int)
    art_colors = {a: to_hex(c) for a, c in zip(art_order, sns.color_palette("turbo", len(art_order)))}
    step = np.min(np.diff(lv))
    y_lo, y_hi = lv.min() - 0.5 * step, lv.max() + 0.5 * step
    dur_pts = pd.DataFrame([(art, dur[art], *((y_hi, "^") if np.isposinf(s50) else (y_lo, "v") if np.isneginf(s50)
                                             else (s50, "o")), s50) for art, s50 in snr50.items()],
                           columns=["art", "x", "y", "mk", "snr50"])

    # D and F: groups of local relative power
    mix_p = mix_df.dropna(subset=["p_local_rel_db"]).copy()
    mix_p["p_bin"] = pd.qcut(mix_p["p_local_rel_db"], N_BINS)                   # deciles: same number of mixtures
    centers = mix_p.groupby("p_bin", observed=True)["p_local_rel_db"].median()
    mix_p["p_center"] = mix_p["p_bin"].map(centers).astype(float)                # each point at the median of its group
    det_p = mix_p[mix_p["detected"] == 1]

    # E: local power vs global SNR
    sc = mix_df.dropna(subset=["p_local_rel_db", "snr_global_db"])
    sub_pts = {"Detected": sc[sc["detected"] == 1], "Not detected": sc[sc["detected"] != 1]}
    extent = (sc["p_local_rel_db"].min(), sc["p_local_rel_db"].max(),
              sc["snr_global_db"].min(), sc["snr_global_db"].max())
    snr_cmap = sns.color_palette("crest_r", as_cmap=True)                        # color = SNR (as in D)
    snr_norm = Normalize(lv.min(), lv.max())

    # G: error of the global peak vs global SNR
    pk = mix_df.dropna(subset=["snr_global_db"]).assign(
        det_lbl=lambda d: np.where(d["detected"] == 1, "Detected", "Not detected"))
    edges_g = np.arange(np.floor(pk["snr_global_db"].min() / BIN_W) * BIN_W, pk["snr_global_db"].max() + BIN_W, BIN_W)
    pk["g_center"] = pd.cut(pk["snr_global_db"], edges_g).apply(lambda b: b.mid).astype(float)
    pk = pk[pk.groupby(["g_center", "det_lbl"])["snr_global_db"].transform("size") >= MIN_N]

    # H: ΔSNR vs NMSE out of the zone; median and IQR per ΔSNR bin
    tr = det_df.dropna(subset=["dsnr_db", "nmse_out_zone_db"])
    edges_h = np.arange(np.floor(tr["dsnr_db"].min() / DS_BIN) * DS_BIN, tr["dsnr_db"].max() + DS_BIN, DS_BIN)
    g = tr.groupby(pd.cut(tr["dsnr_db"], edges_h), observed=True)["nmse_out_zone_db"]
    tr_stats = g.quantile([0.25, 0.5, 0.75]).unstack()
    tr_stats = tr_stats[g.size() >= MIN_N]
    x_mid = np.array([b.mid for b in tr_stats.index], float)


    # === Statistics of the curves and hexagons ===
    snr_color = {lvl: to_hex(snr_cmap(snr_norm(lvl))) for lvl in lv}
    curves_d = curve_table(mix_p, ["snr_db", "p_center"], "detected", mean_ci)   # D: mean ± 95 % CI
    curves_f = curve_table(det_p, ["snr_db", "p_center"], "dsnr_db", mean_ci)    # F
    G_CURVES = [("Before (mixture)", pk, "err_rms_max_glob_mix", C_ORANGE, True),
                ("After (output) · detected", pk[pk["det_lbl"] == "Detected"], "err_rms_max_glob_out", C_BLUE, False),
                ("After (output) · not detected", pk[pk["det_lbl"] == "Not detected"], "err_rms_max_glob_out", C_GRAY,
                 False)]
    curves_g = {name: curve_table(d, "g_center", col, median_iqr) for name, d, col, _, _ in G_CURVES}

    # E: hexagons per subpanel and SNR (high SNRs first: the low ones stay on top where they overlap); relative
    # density = proportion of the subpanel / maximum proportion (common to both subpanels)
    hex_e, e_lim = {}, None
    for name, d in sub_pts.items():
        for lvl in sorted(d["snr_db"].unique(), reverse=True):
            dl = d[d["snr_db"] == lvl]
            off, cnt, hexagon_e, dlim = hexbin(dl["p_local_rel_db"], dl["snr_global_db"], GRID, extent)
            hex_e[name, lvl] = dict(off=off, n=cnt, prop=cnt / len(d))
            e_lim = dlim if e_lim is None else e_lim.union([e_lim, dlim])
    pmax = max(h["prop"].max() for h in hex_e.values())
    for h in hex_e.values():
        h["dens"] = h["prop"] / pmax
        idx = np.clip(np.floor(PowerNorm(GAMMA, 0, 1)(h["dens"]) * 256), 0, 255)   # index of the palette (256)
        h["level"] = (idx // (256 // E_LEVELS)).astype(int)

    # H: hexagons of the detected ones (color = Blues of the relative density, in power scale)
    off_h, cnt_h, hexagon_h, h_lim = hexbin(tr["dsnr_db"], tr["nmse_out_zone_db"], (50, 30))
    dens_h = cnt_h / cnt_h.max()
    idx_h = np.clip(np.floor(PowerNorm(GAMMA, 0, 1)(dens_h) * 256), 0, 255).astype(int)


    # === Limits (matplotlib autoscale) ===
    with_lim = lambda *a: mpl_range(*[np.asarray(v, float) for v in a])
    log_x_c = mpl_range(np.log10(dur_pts["x"].to_numpy(float)))                # log axis: margins in log
    ranges = dict(
        A=([0, len(lv)], [len(art_rows), 0]), B=([0, len(lv)], [len(art_rows), 0]),
        C=(list(10 ** np.array(log_x_c)), with_lim(dur_pts["y"])),
        D=(with_lim(centers), [-0.02, 1.02]),
        Ed=(with_lim([e_lim.x0, e_lim.x1]), with_lim([e_lim.y0, e_lim.y1], [0])),
        F=(with_lim(centers), with_lim(curves_f[["est", "lo", "hi"]], [0])),
        G=(with_lim(pk["g_center"]), with_lim(*[c[["est", "lo", "hi"]] for c in curves_g.values()])),
        H=(with_lim([h_lim.x0, h_lim.x1], x_mid, [0]),
           [NMSE_FLOOR - 5, max(0, tr["nmse_out_zone_db"].max()) + 5]),
    )
    ranges["En"] = ranges["Ed"]

    c_markers = [("o", "black", "SNR at 50% detection")]
    if np.isneginf(snr50).any():
        c_markers.append(("v", "gray", "< 50 % in the whole range"))
    if np.isposinf(snr50).any():
        c_markers.append(("^", "gray", "≥ 50 % in the whole range"))
    snr_labels = [f"{v:g} dB" for v in lv]
    legend_labels = dict(C=[m[2] for m in c_markers], D=snr_labels,
                         F=["No change (ΔSNR = 0 dB)"], G=[c[0] for c in G_CURVES],
                         H=["Median NMSE (IQR)", "No change (ΔSNR = 0 dB)"])


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dash = lambda lw: mpl_dashes(DASH_PATTERNS["--"], lw, PT)
    dot = lambda lw: mpl_dashes(DASH_PATTERNS[":"], lw, PT)

    spec = dict(art_rows=list(art_rows), lv=list(lv), ds_lim=ds_lim, art_order=art_order,
                lims={k: v for k, v in ranges.items() if k not in ("A", "B", "En")},
                legends=legend_labels, c_markers=c_markers, g_loc="upper right", density_ticks=DENSITY_TICKS,
                gamma=GAMMA)
    fig_w, fig_h = 14, 18 + 0.12 * len(art_order)
    geo = _factor_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(6 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _factor_geometry(fig_w, fig_h, dpi, spec)
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H
    dom_x = lambda b: [X(b["x0"]), X(b["x1"])]
    dom_y = lambda b: [Y(b["bottom"]), Y(b["top"])]

    def nticks(axis_range, b, k):
        """
        plotly nticks that gives the matplotlib ticks (mpl_nticks). plotly rounds range/nticks up to the NEXT
        1-2-5 step when it falls exactly on one (30/6 = 5 -> 10), so in that case one more tick is requested
        """
        length_pt = (b["x1"] - b["x0"] if k == "x" else b["bottom"] - b["top"]) / PT
        n = mpl_nticks(axis_range, length_pt, 3 if k == "x" else 2)
        rough = (axis_range[1] - axis_range[0]) / n
        return n + 1 if np.isclose(rough / 10**np.floor(np.log10(rough)), [1, 2, 5]).any() else n

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      tickformatstops=MPL_TICKFORMAT, exponentformat="none")      # sns.despine: no top/right axes
    x_shift = dict(ticklabelstandoff=round(-2.2 * PT))                            # x labels closer to the axis
    y_shift = dict(ticklabelstandoff=round(1.4 * PT))                             # and y labels farther (as matplotlib)
    heat_style = {**axis_style, "showline": False, "showgrid": False}             # seaborn heatmaps: no axes or grid

    # Axes: panels 1-9 and colorbars 10-14
    ID = dict(A="", B="2", C="3", D="4", Ed="5", En="6", F="7", G="8", H="9",
              cbA="10", cbB="11", cbC="12", cbE="13", cbH="14")
    layout = {}

    def add_axes(k, xaxis, yaxis):
        i = ID[k]
        layout[f"xaxis{i}"] = dict(domain=dom_x(geo[k]["ax"]), anchor=f"y{i}", **xaxis)
        layout[f"yaxis{i}"] = dict(domain=dom_y(geo[k]["ax"]), anchor=f"x{i}", **yaxis)

    def linear_axes(k, grid_alpha=1, x_extra=None, y_extra=None):
        style = {**axis_style, "gridcolor": rgba("#b0b0b0", grid_alpha)}
        (xr, yr), b = ranges[k], geo[k]["ax"]
        add_axes(k, dict(range=xr, nticks=nticks(xr, b, "x"), **x_shift, **style, **(x_extra or {})),
                 dict(range=yr, nticks=nticks(yr, b, "y"), **y_shift, **style, **(y_extra or {})))

    centers_x = np.arange(len(lv)) + 0.5
    centers_y = np.arange(len(art_rows)) + 0.5
    for k in "AB":
        add_axes(k, dict(range=[0, len(lv)], tickvals=centers_x, ticktext=[f"{v:g}" for v in lv], **x_shift,
                         **heat_style),
                 dict(range=[len(art_rows), 0], tickvals=centers_y, ticktext=list(art_rows), **y_shift, **heat_style,
                      **({"matches": "y"} if k == "B" else {})))

    def mathtext(s):
        """Text of a matplotlib tick ($\\mathdefault{4\\times10^{1}}$) -> plotly HTML (4×10<sup>1</sup>)"""
        s = re.sub(r"\$\\mathdefault\{(.*)\}\$", r"\1", s).replace(r"\times", "×")
        return re.sub(r"\^\{([^}]*)\}", r"<sup>\1</sup>", s).replace("-", "−")

    c_ticks = [(v, mathtext(s)) for v, s in geo["c_ticks"]]                        # labelled ticks (major and minor)
    c_minor = [v for v in geo["c_minor"] if v not in [t[0] for t in c_ticks]
               and log_x_c[0] <= np.log10(v) <= log_x_c[1]]
    add_axes("C", dict(type="log", range=log_x_c, tickvals=[t[0] for t in c_ticks], ticktext=[t[1] for t in c_ticks],
                       minor=dict(tickvals=c_minor, ticklen=pt(2), tickwidth=pt(0.6), tickcolor="black", ticks="outside",
                                  showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8)), **x_shift,
                       **{**axis_style, "tickformatstops": None}),
             dict(range=ranges["C"][1], nticks=nticks(ranges["C"][1], geo["C"]["ax"], "y"), **y_shift, **axis_style))
    linear_axes("D")
    linear_axes("Ed", grid_alpha=0.3, x_extra=dict(showticklabels=False))
    linear_axes("En", grid_alpha=0.3, x_extra=dict(matches="x5"), y_extra=dict(matches="y5"))
    linear_axes("F", x_extra=dict(matches="x4"))
    linear_axes("G")
    linear_axes("H")

    # Colorbars: own axes with a one-column map (ticks on the right; seaborn ones without edge)
    def colorbar(k, yrange, z, colorscale, outline, zmin=None, zmax=None, tickpos=None):
        ticks = geo["cb" + k]["ticks"]
        frame = dict(showline=outline, mirror=True, linecolor="black", linewidth=pt(0.8), fixedrange=True)
        add_axes("cb" + k, dict(range=[0, 1], showticklabels=False, ticks="", showgrid=False, zeroline=False, **frame),
                 dict(range=yrange, side="right", tickvals=[tickpos(v) if tickpos else v for v, _ in ticks],
                      ticktext=[s for _, s in ticks], **{**heat_style, **frame}))
        y_edges = np.linspace(*yrange, len(z) + 1)
        fig.add_trace(go.Heatmap(z=np.asarray(z)[:, None], x=[0, 1], y=y_edges, colorscale=colorscale, zmin=zmin,
                                 zmax=zmax, showscale=False, hoverinfo="skip", xaxis=f"x{ID['cb' + k]}",
                                 yaxis=f"y{ID['cb' + k]}"))

    fig = go.Figure()
    shapes, annotations = [], []
    lin256 = (np.arange(256) + 0.5) / 256

    # --- Legends ---
    # One plotly legend per item, at the position of the matplotlib text (the text of a plotly item starts 40 px
    # to the right of the x of the legend), a frame with rounded corners and the title in bold
    legends, n_leg = {}, 0

    def legend_items(key, title=None):
        nonlocal n_leg
        ids = {}
        for name, t in zip(legend_labels[key], geo[key]["legend_texts"]):
            n_leg += 1
            lid = "legend" if n_leg == 1 else f"legend{n_leg}"
            legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)
            ids[name] = lid
        if title:
            annotations.append(ann(f"<b>{title}</b>", geo[key]["legend_title"], 10))
        b = geo[key]["legend"]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                           line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H)))
        return ids

    def ann(s, b, size, rotate=False):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size)), textangle=-90 if rotate else 0)

    leg = {k: legend_items(k, title) for k, title in (("C", None), ("D", "SNR"), ("F", None), ("G", "Median ± IQR"),
                                                      ("H", None))}
    snr_group = {lvl: f"snr_{lvl:g}" for lvl in lv}                                 # D, E and F: one group per SNR
    snr_legend = {lvl: leg["D"][name] for lvl, name in zip(lv, snr_labels)}

    def ref_line(axis, value, vertical, color, lw, pattern, name=None, legend=None, opacity=1):
        """Reference line across the whole panel (axhline / axvline), over the curves; with a name, in the legend"""
        i = ID[axis]
        kw = dict(name=name, showlegend=True, legend=legend, legendgroup=name) if name else {}
        pos = dict(x0=value, x1=value, y0=0, y1=1, xref=f"x{i}", yref=f"y{i} domain") if vertical else \
              dict(x0=0, x1=1, y0=value, y1=value, xref=f"x{i} domain", yref=f"y{i}")
        shapes.append(dict(type="line", layer="above", opacity=opacity,
                           line=dict(color=color, width=pt(lw), dash=pattern(lw) if pattern else None), **pos, **kw))

    def band_and_line(axis, tab, color, name, group, legend, show, dashed=False, hover=""):
        """Band (zorder 1) and curve with markers (zorder 2) of an est / lo / hi / n table; returns both traces"""
        i = ID[axis]
        x = tab.index.to_numpy(float)
        ok = np.isfinite(tab["lo"]).to_numpy()
        common = dict(legend=legend, legendgroup=group, xaxis=f"x{i}", yaxis=f"y{i}")
        band = go.Scatter(x=np.r_[x[ok], x[ok][::-1]], y=np.r_[tab["hi"][ok], tab["lo"][ok][::-1]], mode="lines",
                          fill="toself", fillcolor=rgba(color, 0.2), line=dict(width=0), hoverinfo="skip",
                          showlegend=False, **common)
        line = go.Scatter(x=x, y=tab["est"], mode="lines+markers", name=name,
                          line=dict(color=color, width=pt(2), dash=dash(2) if dashed else None),
                          marker=dict(size=pt(6), color=color, line=dict(color="#000000", width=pt(0.75))),
                          customdata=tab[["lo", "hi", "n"]].to_numpy(), hovertemplate=hover, showlegend=show, **common)
        return band, line

    # --- A and B: artifact × SNR heatmaps ---
    def heatmap(k, values, colorscale, zmin, zmax, fmt, cmap, hover):
        i = ID[k]
        z = values.to_numpy(float)
        art_snr = np.dstack(np.meshgrid([f"{v:g}" for v in lv], list(art_rows))[::-1])   # (row, column, [art, snr])
        fig.add_trace(go.Heatmap(z=z, x=centers_x, y=centers_y, colorscale=colorscale, zmin=zmin, zmax=zmax,
                                 showscale=False, customdata=art_snr, hoverongaps=False, hovertemplate=hover,
                                 xaxis=f"x{i}", yaxis=f"y{i}"))
        # White edges of the cells (linewidths=0.3, linecolor="white": matplotlib draws them 2 px wide)
        n_c, n_r = len(lv), len(art_rows)
        xs = sum(([v, v, None] for v in range(n_c + 1)), []) + sum(([0, n_c, None] for _ in range(n_r + 1)), [])
        ys = sum(([0, n_r, None] for _ in range(n_c + 1)), []) + sum(([v, v, None] for v in range(n_r + 1)), [])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#ffffff", width=dpi / 50),
                                 hoverinfo="skip", showlegend=False, xaxis=f"x{i}", yaxis=f"y{i}"))
        # Values of the cells, in dark gray or white depending on the luminance of the cell (as seaborn)
        rows, cols = np.nonzero(np.isfinite(z))
        colors = ["#262626" if relative_luminance(cmap(Normalize(zmin, zmax)(z[r, c]))) > 0.408 else "#ffffff"
                  for r, c in zip(rows, cols)]
        fig.add_trace(go.Scatter(x=centers_x[cols], y=centers_y[rows], mode="text",
                                 text=[format(z[r, c], fmt) for r, c in zip(rows, cols)],
                                 textfont=dict(size=pt(8), color=colors), hoverinfo="skip", showlegend=False,
                                 xaxis=f"x{i}", yaxis=f"y{i}"))

    heatmap("A", det_rate, mpl_colorscale("Purples"), 0, 1, ".2f", colormaps["Purples"],
            "Artifact %{customdata[0]}<br>SNR: %{customdata[1]} dB<br>Detection rate: %{z:.2f}<extra></extra>")

    # B: cells without detections, hatched (the matplotlib hatch is the axes background: 18 lines per inch)
    nan_r, nan_c = np.nonzero(np.isnan(ds_mean.to_numpy(float)))
    if len(nan_r):
        hx = sum(([c, c + 1, c + 1, c, c, None] for c in nan_c), [])
        hy = sum(([r, r, r + 1, r + 1, r, None] for r in nan_r), [])
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", fill="toself", line=dict(width=0), hoverinfo="skip",
                                 fillpattern=dict(shape="/", fgcolor=C_LIGHTGRAY, bgcolor="#ffffff", size=dpi / 18,
                                                  solidity=0.25),
                                 showlegend=False, xaxis="x2", yaxis="y2"))
    heatmap("B", ds_mean, mpl_colorscale("RdBu"), -ds_lim, ds_lim, ".1f", colormaps["RdBu"],
            "Artifact %{customdata[0]}<br>SNR: %{customdata[1]} dB<br>Mean ΔSNR: %{z:.1f} dB<extra></extra>")

    colorbar("A", [0, 1], lin256, mpl_colorscale("Purples"), outline=False, zmin=0, zmax=1)
    colorbar("B", [-ds_lim, ds_lim], -ds_lim + 2 * ds_lim * lin256, mpl_colorscale("RdBu"), outline=False,
             zmin=-ds_lim, zmax=ds_lim)

    # --- C: SNR at 50 % vs duration ---
    # On log axes, the shapes use the data value and the annotations its log10
    MK = {"o": ("circle", 1), "v": ("triangle-down", 1.075), "^": ("triangle-up", 1.075)}   # same area as matplotlib
    for mk, legend_color, name in c_markers:
        d = dur_pts[dur_pts["mk"] == mk]
        symbol, scale = MK[mk]
        snr_txt = [f"{v:.1f} dB" if np.isfinite(v) else ("≥ 50 % in the whole range" if v > 0 else
                                                         "< 50 % in the whole range") for v in d["snr50"]]
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"], mode="markers", name=name, xaxis="x3", yaxis="y3",
            marker=dict(symbol=symbol, size=pt(np.sqrt(60)) * scale, color=[art_colors[a] for a in d["art"]],
                        line=dict(color="#000000", width=pt(1))),
            customdata=np.c_[d["art"], snr_txt],
            hovertemplate="Artifact %{customdata[0]}<br>Duration: %{x:.0f} ms<br>SNR at 50%: %{customdata[1]}<extra></extra>",
            showlegend=False, legend=leg["C"][name], legendgroup=f"C_{mk}"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, xaxis="x3", yaxis="y3",
                                 marker=dict(symbol=symbol, size=pt(np.sqrt(60)) * scale, color=to_hex(legend_color),
                                             line=dict(color="#000000", width=pt(1))),
                                 legend=leg["C"][name], legendgroup=f"C_{mk}"))
    for _, p in dur_pts.iterrows():                                               # number of the artifact (+4, +3 pt)
        annotations.append(dict(text=str(p["art"]), x=np.log10(p["x"]), y=p["y"], xref="x3", yref="y3",
                                xanchor="left", yanchor="bottom", xshift=pt(4), yshift=pt(3) - 0.2 * pt(8),
                                showarrow=False, font=dict(size=pt(8)), borderpad=0))
    n_art = len(art_order)
    colorbar("C", [-0.5, n_art - 0.5], np.arange(n_art),
             sum(([[i / n_art, art_colors[a]], [(i + 1) / n_art, art_colors[a]]] for i, a in enumerate(art_order)), []),
             outline=True, zmin=-0.5, zmax=n_art - 0.5)

    # --- D and F: curves per SNR (bands below all of them) ---
    for axis, curves, what in (("D", curves_d, "Detection rate: %{y:.3f}"), ("F", curves_f, "Mean ΔSNR: %{y:.2f} dB")):
        pairs = [band_and_line(axis, curves.loc[lvl], snr_color[lvl], name, snr_group[lvl], snr_legend[lvl],
                               show=(axis == "D"),
                               hover=(f"SNR {name}<br>Local relative power: %{{x:.2f}} dB<br>{what}<br>"
                                      "CI 95%: (%{customdata[0]:.3f}, %{customdata[1]:.3f})<br>"
                                      "n = %{customdata[2]:.0f} mixtures<extra></extra>"))
                 for lvl, name in zip(lv, snr_labels) if lvl in curves.index.get_level_values(0)]
        fig.add_traces([b for b, _ in pairs] + [l for _, l in pairs])
    ref_line("D", 0, True, C_GRAY, 1.2, dot)                                       # mean level of the signal
    ref_line("F", 0, False, C_GRAY, 2, dash, name=legend_labels["F"][0], legend=leg["F"][legend_labels["F"][0]])

    # --- E: hexagons (color = SNR, opacity = density) ---
    for name, axis in (("Detected", "Ed"), ("Not detected", "En")):
        i = ID[axis]
        hover = {}                                                                  # hexagon -> n. of mixtures per SNR
        for lvl in sorted(sub_pts[name]["snr_db"].unique(), reverse=True):
            h = hex_e[name, lvl]
            for level in np.unique(h["level"]):
                m = h["level"] == level
                alpha = 0.15 + 0.85 * (level * (256 // E_LEVELS) + (256 // E_LEVELS - 1) / 2) / 255
                hx, hy = hex_polygons(h["off"][m], hexagon_e)
                fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", fill="toself", fillcolor=rgba(snr_color[lvl], alpha),
                                         line=dict(width=0), hoverinfo="skip", showlegend=False,
                                         legend=snr_legend[lvl], legendgroup=snr_group[lvl], xaxis=f"x{i}",
                                         yaxis=f"y{i}"))
            for (cx, cy), n, dens in zip(h["off"], h["n"], h["dens"]):
                hover.setdefault((cx, cy), []).append(f"SNR {lvl:g} dB: {n:.0f} ({dens:.2f})")
        pts = np.array(list(hover))
        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="markers", marker=dict(size=pt(6), opacity=0), showlegend=False,
            text=["<br>".join(v[::-1]) for v in hover.values()], xaxis=f"x{i}", yaxis=f"y{i}",
            hovertemplate=(f"{name}<br>Local relative power: %{{x:.1f}} dB<br>Global SNR: %{{y:.1f}} dB<br>"
                           "Mixtures (relative density):<br>%{text}<extra></extra>")))

        ref_line(axis, 0, False, C_RED, 2, dash)
        d = sub_pts[name]
        annotations += [
            dict(text=f"<b>{name} (n = {len(d)})</b>", x=0.01, y=0.97, xref=f"x{i} domain", yref=f"y{i} domain",
                 xanchor="left", yanchor="top", yshift=pt(2), showarrow=False, font=dict(size=pt(11)), borderpad=0),
            dict(text=f"Global SNR < 0 dB: {(d['snr_global_db'] < 0).mean():.1%}", x=0.99, y=0.03,
                 xref=f"x{i} domain", yref=f"y{i} domain", xanchor="right", yanchor="bottom", showarrow=False,
                 font=dict(size=em, color=C_RED), bgcolor="rgba(255, 255, 255, 0.9)", bordercolor=rgba(C_RED, 0.9),
                 borderwidth=pt(1), borderpad=pt(2), xshift=pt(2) + pt(1), yshift=-pt(2) - pt(1))]
    colorbar("E", [0, 1], lin256, [[v, f"rgba(0, 0, 0, {0.15 + 0.85 * v:.4f})"] for v in np.linspace(0, 1, 256)],
             outline=True, zmin=0, zmax=1, tickpos=lambda v: v ** GAMMA)

    # --- G: error of the global peak vs global SNR ---
    pairs = [band_and_line("G", curves_g[name], color, name, f"G_{n}", leg["G"][name], show=True, dashed=dashed,
                           hover=(f"{name}<br>Global SNR: %{{x:g}} dB<br>Median: %{{y:.1f}} %<br>"
                                  "IQR: (%{customdata[0]:.1f}, %{customdata[1]:.1f}) %<br>"
                                  "n = %{customdata[2]:.0f} mixtures<extra></extra>"))
             for n, (name, _, _, color, dashed) in enumerate(G_CURVES)]
    fig.add_traces([b for b, _ in pairs] + [l for _, l in pairs])

    # --- H: ΔSNR vs NMSE out of the zone ---
    blues = colormaps["Blues"]
    for idx in np.unique(idx_h):
        hx, hy = hex_polygons(off_h[idx_h == idx], hexagon_h)
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines", fill="toself", fillcolor=to_hex(blues(int(idx))),
                                 line=dict(width=0), hoverinfo="skip", showlegend=False, xaxis="x9", yaxis="y9"))
    fig.add_trace(go.Scatter(x=off_h[:, 0], y=off_h[:, 1], mode="markers", marker=dict(size=pt(6), opacity=0),
                             customdata=np.c_[cnt_h, dens_h], showlegend=False, xaxis="x9", yaxis="y9",
                             hovertemplate=("ΔSNR: %{x:.1f} dB<br>NMSE out of zone: %{y:.1f} dB<br>"
                                            "Mixtures: %{customdata[0]:.0f}<br>Relative density: %{customdata[1]:.3f}"
                                            "<extra></extra>")))
    lbl_h = legend_labels["H"][0]
    tab_h = pd.DataFrame({"est": tr_stats[0.5].to_numpy(), "lo": tr_stats[0.25].to_numpy(),
                          "hi": tr_stats[0.75].to_numpy(), "n": g.size()[tr_stats.index].to_numpy()}, index=x_mid)
    fig.add_traces(band_and_line("H", tab_h, C_RED, lbl_h, "H_median", leg["H"][lbl_h], show=True,
                                 hover=("ΔSNR: %{x:g} dB (bin center)<br>Median NMSE: %{y:.1f} dB<br>"
                                        "IQR: (%{customdata[0]:.1f}, %{customdata[1]:.1f}) dB<br>"
                                        "n = %{customdata[2]:.0f} mixtures<extra></extra>")))
    ref_line("H", 0, True, C_GRAY, 2, dash, name=legend_labels["H"][1], legend=leg["H"][legend_labels["H"][1]])
    colorbar("H", [0, 1], lin256, mpl_colorscale("Blues"), outline=True, zmin=0, zmax=1, tickpos=lambda v: v ** GAMMA)

    # --- Titles and labels ---
    TEXTS = {   # panel: (title, x label, y label); "" = no text
        "A": ("Detection rate", "SNR (dB)", "Artifact"),
        "B": ("Mean ΔSNR of detected artifacts", "SNR (dB)", ""),
        "C": ("SNR at 50 % detection vs artifact duration", "Artifact duration (ms)", "SNR at 50 % detection (dB)"),
        "D": ("Detection rate vs local signal power", "Local relative power (dB)", "Detection rate"),
        "Ed": ("Local power vs global SNR", "", "Global SNR (dB)"),
        "En": ("", "Local relative power (dB)", "Global SNR (dB)"),
        "F": ("ΔSNR of detected artifacts vs local signal power", "Local relative power (dB)",
              "Mean ΔSNR of detected (dB)"),
        "G": ("Global peak error before / after vs global SNR", "Global SNR (dB)", "Global RMS peak error (%)"),
        "H": ("Removal vs conservation (detected)", "ΔSNR in the artifact zone (dB)", "NMSE out of zone (dB)"),
    }
    CB_LABELS = dict(A=("Detection rate", 10), B=("Mean ΔSNR (dB)", 10), C=("Artifact", 11),
                     E=("Relative density (opacity)", 11), H=("Relative density", 11))

    annotations.append(ann("Factor analysis", geo["suptitle"], 16))
    for k, (title, xlabel, ylabel) in TEXTS.items():
        for name, s, size, rotate in (("title", f"<b>{title}</b>", 14, False), ("xlabel", xlabel, 12, False),
                                      ("ylabel", ylabel, 12, True)):
            if name in geo[k]:
                annotations.append(ann(s, geo[k][name], size, rotate))
    for k, (label, size) in CB_LABELS.items():
        annotations.append(ann(label, geo["cb" + k]["label"], size, rotate=True))

    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )
    return fig


# ====================================================================================================================
# 5. Main functions: experiment 2 (insertion of several artifacts + idempotence)
# ====================================================================================================================

# --------------------------------------------------------------------------------------------------------------------
# 5.1. Experiment 2 vs experiment 1 (section 6.4.1 of evaluation_denoise.ipynb)
# --------------------------------------------------------------------------------------------------------------------

def _comparison_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_experiment_comparison figure as matplotlib computes it (layout="constrained"): builds the same
    figure WITHOUT data (titles, labels, scales, limits, ticks and legends) and returns the boxes (px from the top-left
    corner of the image cropped as bbox_inches="tight") of the axes, texts and legends, and the ticks that matplotlib
    puts on each axis.
    spec: {panel: dict(title, title_pad, xlabel, ylabel, xlim, ylim, xticks, yticks, yscale, legend, legend_ls, loc),
           "C_legend": [...], "legend": [...], "suptitle": str}
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
    sub_det = gs[0, 0].subgridspec(3, 1, height_ratios=[2, 0.5, 0.5])
    sub_own = gs[0, 1].subgridspec(3, 1, height_ratios=[2, 0.5, 0.5])
    sub_ds = gs[1, 1].subgridspec(2, 1, height_ratios=[2, 1])
    sub_net = gs[2, 1].subgridspec(2, 1, height_ratios=[2, 1])
    cells = dict(A=sub_det[0], A1=sub_det[1], A2=sub_det[2], B=sub_own[0], B1=sub_own[1], B2=sub_own[2], C=gs[1, 0],
                 D=sub_ds[0], D1=sub_ds[1], E=gs[2, 0], F=sub_net[0], F1=sub_net[1])
    axs = {k: fig.add_subplot(c) for k, c in cells.items()}

    for k, ax in axs.items():
        s = spec[k]
        ax.set_title(s.get("title", ""), fontsize=14, fontweight="bold", pad=s.get("title_pad"))
        ax.set_xlabel(s.get("xlabel", ""), fontsize=12)
        ax.set_ylabel(s.get("ylabel", ""), fontsize=12)
        if s.get("yscale"):
            ax.set_yscale(*s["yscale"][:1], **s["yscale"][1])
        if s.get("xticks"):
            ax.set_xticks(*s["xticks"])
        if s.get("yticks"):
            ax.set_yticks(*s["yticks"])
        ax.set_xlim(*s["xlim"])
        ax.set_ylim(*s["ylim"])
        if s.get("legend"):                                                        # legend inside the axes (D)
            ax.legend([Line2D([], [], ls=ls) for ls in s["legend_ls"]], s["legend"], fontsize=10, loc=s["loc"])

    # Legend of C (above the axes) and common legend (below the suptitle)
    hC = [Line2D([], [], ls="", marker="o", ms=8), Line2D([], [], ls="", marker="o", ms=8), Line2D([], [], ls="--")]
    axs["C"].legend(hC, spec["C_legend"], fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    h = [Line2D([], [], marker="o", lw=2) for _ in spec["legend"][:-1]] + [Line2D([], [], ls="--")]
    fig.suptitle(spec["suptitle"], fontsize=16, y=1.04, va="top")
    leg = fig.legend(h, spec["legend"], loc="upper center", bbox_to_anchor=(0.5, 1.026), ncol=len(h), fontsize=10)

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    def ticks(axis):
        """Visible ticks (within the limits) and their labels"""
        lo, hi = sorted(axis.get_view_interval())
        tl = [(t.get_loc(), t.label1.get_text()) for t in axis.get_major_ticks()
              if lo - 1e-9*(hi - lo) <= t.get_loc() <= hi + 1e-9*(hi - lo)]
        return [v for v, _ in tl], [s for _, s in tl]

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle),
               legend=box(leg), legend_texts=[box(t) for t in leg.get_texts()])
    for k, ax in axs.items():
        out[k] = dict(ax=box(ax), xlabel=box(ax.xaxis.label), ylabel=box(ax.yaxis.label), title=box(ax.title),
                      xticks=ticks(ax.xaxis), yticks=ticks(ax.yaxis))
        if ax.get_legend():
            out[k]["legend"] = box(ax.get_legend())
            out[k]["legend_texts"] = [box(t) for t in ax.get_legend().get_texts()]
    return out


def plot_experiment_comparison(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_experiment_comparison (matplotlib/seaborn, section 6.4.1 of
    evaluation_denoise.ipynb): same layout (layout="constrained", figsize 14 x 12 in), sizes, fonts, colors and
    positions. The figure is cropped as the inline backend does (bbox_inches="tight"), so its size matches the
    matplotlib image.
    Experiment 2 (equal condition) vs experiment 1, per number of artifacts in the mixture (1, 2, 3):
    - A: detection rate vs SNR (95 % CI between clean signals); A': paired difference with exp. 1; A'': proportion of
      merged artifacts
    - B: own detection vs SNR; B': difference with exp. 1; B'': borrowed detection (total − own)
    - C: SNR at 50 % detection, total and own, with its 95 % CI (bootstrap over the clean signals)
    - D: ΔSNR of the detected artifacts vs SNR, with the "no change" reference (0 dB); D': difference with exp. 1
    - E: error of the local mean RMS after the filter vs SNR (median ± IQR; symlog axis, linear within ±10 %)
    - F: net SNR improvement of all the artifacts (not detected -> ΔSNR = 0) vs SNR; F': difference with exp. 1
    Interactivity: hover with the values (number of artifacts, SNR, value, CI...) in every panel; click on the common
    legend to hide/show a number of artifacts or experiment 1 in every panel together (lines and bands), on the legend
    of C for the total detection, the own one or the reference of experiment 1, and on the legend of D for each
    reference line; zoom in each panel: the SNR x axes of ALL the panels move together (also those of B, B' and B'',
    which in matplotlib only share their axis among themselves) and the shared y axes as sharey.
    The positions of the axes and texts and the ticks are taken from the constrained layout of matplotlib applied to
    the same figure without data (_comparison_geometry).
    Limits: plotly has no symlog scale, so E is drawn with the already transformed values (fixed ticks at ±10^k and 0;
    the hover gives the real value); the axes with a step of 2.5·10^k (B'', D', F) use a fixed step (plotly only picks
    1-2-5 steps), which does not adapt when zooming; the plotly grid always stays below the bands; the bootstrap CIs of
    the curves use a fixed seed (seaborn does not), so they can differ slightly from the matplotlib version (those of C
    are the same: fixed seed in both).

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 2 (read_results("experiment_2.csv")). It is not modified.
        Also reads results/experiments/experiment_1.csv (read_results)
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Data (as the matplotlib version) ===
    N_COLORS = {1: "#AFA9EC", 2: "#7F77DD", 3: "#3C3489"}                        # single-hue ramp: 1 → 3 artifacts
    C_EXP1 = "#808080"                                                             # "gray"
    N_ART = (1, 2, 3)

    mix2 = results_df[results_df["condition"] == "equal"]

    PER_ART = ["art_ids", "snrs_db", "k0", "detected", "merged", "dsnr_db", "err_rms_mean_loc_out"]
    art2 = mix2.explode(PER_ART).reset_index(drop=True)                            # copy: results_df is not modified
    for col in ("snrs_db", "k0", "dsnr_db", "err_rms_mean_loc_out"):
        art2[col] = pd.to_numeric(art2[col])
    for col in ("detected", "merged"):
        art2[col] = art2[col].astype(bool).astype(float)
    art2["n_art"] = art2["n_art"].astype(int)

    results1 = read_results("experiment_1.csv")
    results1["snr_db"] = pd.to_numeric(results1["snr_db"], errors="coerce")
    mix1 = results1[results1["art_id"] != "Control"]

    lv = np.sort(art2["snrs_db"].unique())

    # A: total detection; A': difference with exp. 1; A'': merged
    det_clean = art2.groupby(["clean_id", "snrs_db", "n_art"], as_index=False)["detected"].mean()
    det_exp1 = mix1.groupby(["clean_id", "snr_db"], as_index=False)["detected"].mean()
    diff_det = det_clean.merge(det_exp1.rename(columns={"snr_db": "snrs_db", "detected": "det_exp1"}),
                               on=["clean_id", "snrs_db"], how="inner")
    diff_det["diff"] = diff_det["detected"] - diff_det["det_exp1"]
    mrg_clean = (art2[art2["n_art"] > 1]
                 .groupby(["clean_id", "snrs_db", "n_art"], as_index=False)["merged"].mean())

    # B: own detection = detected and (not merged, or merged but also detected alone)
    KEY = ["clean_id", "art_ids", "k0", "snrs_db"]
    solo = (art2.loc[art2["n_art"] == 1, KEY + ["detected"]]
            .drop_duplicates(KEY).rename(columns={"detected": "det_solo"}))
    art2 = art2.merge(solo, on=KEY, how="left")
    art2["det_own"] = ((art2["detected"] == 1) & ((art2["merged"] == 0) | (art2["det_solo"] == 1))).astype(float)
    own_clean = art2.groupby(["clean_id", "snrs_db", "n_art"], as_index=False)["det_own"].mean()
    diff_own = own_clean.merge(det_exp1.rename(columns={"snr_db": "snrs_db", "detected": "det_exp1"}),
                               on=["clean_id", "snrs_db"], how="inner")
    diff_own["diff"] = diff_own["det_own"] - diff_own["det_exp1"]
    bor = det_clean.merge(own_clean, on=["clean_id", "snrs_db", "n_art"])
    bor["borrowed"] = bor["detected"] - bor["det_own"]

    # D: ΔSNR of the detected ones; D': difference with exp. 1
    ds_clean = (art2[art2["detected"] == 1]
                .groupby(["clean_id", "snrs_db", "n_art"], as_index=False)["dsnr_db"].mean())
    ds_exp1 = (mix1[mix1["detected"] == 1]
               .groupby(["clean_id", "snr_db"], as_index=False)["dsnr_db"].mean())
    diff_ds = ds_clean.merge(ds_exp1.rename(columns={"snr_db": "snrs_db", "dsnr_db": "ds_exp1"}),
                             on=["clean_id", "snrs_db"], how="inner")
    diff_ds["diff"] = diff_ds["dsnr_db"] - diff_ds["ds_exp1"]

    # F: net SNR improvement of all the artifacts (not detected -> ΔSNR = 0); F': difference with exp. 1
    net_clean = (art2.assign(dsnr=art2["dsnr_db"].where(art2["detected"] == 1, 0.0))
                 .groupby(["clean_id", "snrs_db", "n_art"], as_index=False)["dsnr"].mean())
    net_exp1 = (mix1.assign(dsnr=mix1["dsnr_db"].where(mix1["detected"] == 1, 0.0))
                .groupby(["clean_id", "snr_db"], as_index=False)["dsnr"].mean())
    diff_net = net_clean.merge(net_exp1.rename(columns={"snr_db": "snrs_db", "dsnr": "net_exp1"}),
                               on=["clean_id", "snrs_db"], how="inner")
    diff_net["diff"] = diff_net["dsnr"] - diff_net["net_exp1"]


    # --- Mean and 95 % CI per n_art and SNR (as seaborn) ---
    N_BOOT = 1000
    rng_ci = np.random.default_rng(1)

    def mean_ci(v):
        """Mean and bootstrap 95 % CI of the observations (errorbar=("ci", 95), n_boot=1000 of seaborn)"""
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        n = len(v)
        if n == 0:
            return (np.nan, np.nan, np.nan)
        if np.isin(v, (0, 1)).all():                                               # binary: the resampled mean
            boots = rng_ci.binomial(n, v.mean(), N_BOOT) / n                       # follows a binomial (= bootstrap)
        else:
            boots = v[rng_ci.integers(0, n, (N_BOOT, n))].mean(1)
        return (v.mean(), *np.percentile(boots, [2.5, 97.5]))

    def ci_table(df, col):
        """Mean, lower CI and upper CI of col per (n_art, SNR)"""
        gb = df.groupby(["n_art", "snrs_db"])[col]
        return pd.DataFrame([mean_ci(g) for _, g in gb], index=pd.MultiIndex.from_tuples(gb.groups.keys()),
                            columns=["mean", "lo", "hi"])

    t_det, t_ddet, t_mrg = ci_table(det_clean, "detected"), ci_table(diff_det, "diff"), ci_table(mrg_clean, "merged")
    t_own, t_down, t_bor = (ci_table(own_clean, "det_own"), ci_table(diff_own, "diff"),
                            ci_table(bor[bor["n_art"] > 1], "borrowed"))
    t_ds, t_dds = ci_table(ds_clean, "dsnr_db"), ci_table(diff_ds, "diff")
    t_net, t_dnet = ci_table(net_clean, "dsnr"), ci_table(diff_net, "diff")
    e_det = det_exp1.groupby("snr_db")["detected"].mean()                         # experiment 1: only the mean
    e_ds = ds_exp1.groupby("snr_db")["dsnr_db"].mean()
    e_net = net_exp1.groupby("snr_db")["dsnr"].mean()

    # E: median and 25-75 percentiles of all the artifacts (estimator="median", errorbar=("pi", 50))
    g = art2.groupby(["n_art", "snrs_db"])["err_rms_mean_loc_out"]
    t_rms = pd.DataFrame({"mean": g.median(), "lo": g.quantile(0.25), "hi": g.quantile(0.75)})
    e_rms = mix1.groupby("snr_db")["err_rms_mean_loc_out"].median()

    # symlog scale of E (matplotlib: base 10, linthresh 10, linscale 1)
    LINTHRESH = 10
    LINSCALE_ADJ = 1 / (1 - 1 / 10)

    def symlog(v):
        v = np.asarray(v, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sign(v) * LINTHRESH * (LINSCALE_ADJ - 1 + np.log10(np.abs(v)))
        return np.where(np.abs(v) <= LINTHRESH, v * LINSCALE_ADJ, out)


    # --- C: SNR at 50 % per n_art (same as matplotlib) ---
    def snr_at_50(rate, levels):
        """SNR at which the rate crosses 0.5 (linear interpolation). +inf / -inf: above / below the range"""
        above = rate >= 0.5
        if not above.any():
            return -np.inf
        i = np.where(above)[0].max()
        if i == len(rate) - 1:
            return np.inf
        r0, r1 = rate[i], rate[i + 1]
        return levels[i] + (0.5 - r0) * (levels[i + 1] - levels[i]) / (r1 - r0)

    def snr50_boot(df, col, snr_col, levels, n_boot=1000, seed=0):
        """SNR at 50 % of the column col and its 95 % CI (bootstrap over the clean signals, as in Detection)"""
        clean_ids = sorted(df["clean_id"].unique())
        g = df.groupby(["clean_id", snr_col])[col].agg(["sum", "count"])
        idx = pd.MultiIndex.from_product([clean_ids, levels])
        S = g["sum"].reindex(idx, fill_value=0).to_numpy().reshape(len(clean_ids), len(levels))
        Nn = g["count"].reindex(idx, fill_value=0).to_numpy().reshape(S.shape)
        W = np.random.default_rng(seed).multinomial(len(clean_ids), np.full(len(clean_ids), 1 / len(clean_ids)),
                                                    size=n_boot)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = snr_at_50(S.sum(0) / Nn.sum(0), levels)
            boot = np.array([snr_at_50(r, levels) for r in (W @ S) / (W @ Nn)])
        ci = np.percentile(np.clip(boot, levels.min(), levels.max()), [2.5, 97.5])
        return est, ci

    DY = 0.15                                                                      # vertical offset total / own
    s50 = {(n, col): snr50_boot(art2[art2["n_art"] == n], col, "snrs_db", lv)
           for n in N_ART for col in ("detected", "det_own")}
    est1, ci1 = snr50_boot(mix1, "detected", "snr_db", np.sort(mix1["snr_db"].unique()))


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dashed = lambda lw: mpl_dashes(DASH_PATTERNS["--"], lw, PT)
    dotted = lambda lw: mpl_dashes(DASH_PATTERNS[":"], lw, PT)
    lv_labels = [f"{v:g}" for v in lv]

    # Limits (matplotlib autoscale: data + 5 %; axhline/axvline/axvspan count)
    vals = lambda *tabs: [t[["mean", "lo", "hi"]].to_numpy().ravel() if isinstance(t, pd.DataFrame)
                          else np.asarray(t, float).ravel() for t in tabs]
    x_range = mpl_range(lv)
    y_AB = mpl_range(*vals(t_det, t_own, e_det, [0.5]))                            # A and B (sharey)
    y_AB1 = mpl_range(*vals(t_ddet, t_down, [0]))                                  # A' and B' (sharey)
    y_A2 = [0, mpl_range(*vals(t_mrg, [0]))[1]]                                    # set_ylim(bottom=0)
    y_B2 = mpl_range(*vals(t_bor, [0]))
    x_C = mpl_range(*[np.r_[e, ci] if np.isfinite(e) else [lv.min() if e < 0 else lv.max()]   # CI or arrow at the edge
                      for e, ci in s50.values()], np.r_[est1, ci1])
    y_C = [2.6, -0.6]                                                              # 1 artifact on top
    y_D = mpl_range(*vals(t_ds, e_ds, [0]))
    lim = np.max(np.abs(mpl_range(*vals(t_dds, [0]))))
    y_D1 = [-lim, lim]
    # E: the axhline before set_yscale("symlog") leaves the 5 % margins computed in LINEAR scale
    y_E = mpl_range(*vals(t_rms, e_rms, [0]))
    y_E_t = list(symlog(y_E))                                                      # range in the transformed space
    y_F = mpl_range(*vals(t_net, e_net, [0]))
    lim = np.max(np.abs(mpl_range(*vals(t_dnet, [0]))))                          # symmetric, as D'
    y_F1 = [-lim, lim]

    legend_C = ["Total detection", "Own detection", f"Experiment 1: {est1:.1f} dB"]
    legend_D = ["No change (ΔSNR = 0 dB)"]
    leg_names = [f"{n} artifact{'s' if n != 1 else ''}" for n in N_ART] + ["Experiment 1"]
    xt = (lv, lv_labels)
    spec = dict(
        suptitle="Experiment 2 vs Experiment 1", legend=leg_names, C_legend=legend_C,
        A=dict(title="Detection rate vs SNR", ylabel="Detection rate", xlim=x_range, ylim=y_AB, xticks=xt),
        A1=dict(ylabel="Δ vs exp. 1", xlim=x_range, ylim=y_AB1, xticks=xt),
        A2=dict(xlabel="SNR (dB)", ylabel="Merged", xlim=x_range, ylim=y_A2, xticks=xt),
        B=dict(title="Own detection rate vs SNR", xlim=x_range, ylim=y_AB, xticks=xt),
        B1=dict(xlim=x_range, ylim=y_AB1, xticks=xt),
        B2=dict(xlabel="SNR (dB)", ylabel="Total - Own", xlim=x_range, ylim=y_B2, xticks=xt),
        C=dict(title="SNR at 50 % detection", title_pad=28, xlabel="SNR at 50 % detection (dB)",
               ylabel="Artifacts per mixture", xlim=x_C, ylim=y_C, yticks=(range(3), ["1", "2", "3"])),
        D=dict(title="ΔSNR of detected artifacts vs SNR", ylabel="ΔSNR (dB)", xlim=x_range, ylim=y_D, xticks=xt,
               legend=legend_D, legend_ls=[":"], loc="lower left"),
        D1=dict(xlabel="SNR (dB)", ylabel="Δ vs exp. 1 (dB)", xlim=x_range, ylim=y_D1, xticks=xt),
        E=dict(title="Mean local RMS error after processing vs SNR", xlabel="SNR (dB)",
               ylabel="Mean local RMS error (%)", xlim=x_range, ylim=y_E, xticks=xt,
               yscale=("symlog", dict(linthresh=LINTHRESH))),
        F=dict(title="Net SNR improvement vs SNR (all artifacts)", ylabel="Mean ΔSNR (dB)", xlim=x_range, ylim=y_F,
               xticks=xt),
        F1=dict(xlabel="SNR (dB)", ylabel="Δ vs exp. 1 (dB)", xlim=x_range, ylim=y_F1, xticks=xt),
    )

    fig_w, fig_h = 14, 12
    geo = _comparison_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(3 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _comparison_geometry(fig_w, fig_h, dpi, spec)
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H
    dom_x = lambda b: [X(b["x0"]), X(b["x1"])]
    dom_y = lambda b: [Y(b["bottom"]), Y(b["top"])]

    def y_ticks(k, axis_range):
        """
        Ticks of the linear y axis with the matplotlib step. Steps 1-2-5·10^k: nticks (they adapt to the zoom; plotly
        takes the 1-2-5 step strictly greater than range/nticks). Steps 2.5·10^k (plotly does not pick them):
        fixed step
        """
        v, labels = geo[k]["yticks"]
        step = v[1] - v[0]
        mant = step / 10**np.floor(np.log10(step))
        if np.isclose(mant, 2.5):
            dec = max(len(s.split(".")[1]) if "." in s else 0 for s in labels)
            return dict(tick0=0, dtick=step, tickformat=f".{dec}f")
        return dict(nticks=int(np.floor(abs(axis_range[1] - axis_range[0]) / step)) + 1,
                    tickformatstops=MPL_TICKFORMAT)

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none")                                       # sns.despine: no top/right axes
    x_style = dict(**axis_style, ticklabelstandoff=round(-2.2 * PT))
    AX = dict(A=1, A1=2, A2=3, B=4, B1=5, B2=6, C=7, D=8, D1=9, E=10, F=11, F1=12)   # plotly axis number
    xa = lambda k: "x" if AX[k] == 1 else f"x{AX[k]}"
    ya = lambda k: "y" if AX[k] == 1 else f"y{AX[k]}"
    y_ranges = dict(A=y_AB, A1=y_AB1, A2=y_A2, B=y_AB, B1=y_AB1, B2=y_B2, D=y_D, D1=y_D1, F=y_F, F1=y_F1)
    MATCH_X = {k: "x" for k in AX if k not in ("A", "C")}                         # all the SNR axes linked
    MATCH_Y = dict(B="y", B1="y2")                                                 # sharey

    layout = {}
    for k, n in AX.items():
        b = geo[k]["ax"]
        xkey, ykey = ("xaxis", "yaxis") if n == 1 else (f"xaxis{n}", f"yaxis{n}")
        if k == "C":
            layout[xkey] = dict(domain=dom_x(b), anchor=ya(k), range=x_C, tickformatstops=MPL_TICKFORMAT,
                                nticks=int(np.floor((x_C[1] - x_C[0]) / np.diff(geo[k]["xticks"][0][:2])[0])) + 1,
                                **x_style)
            layout[ykey] = dict(domain=dom_y(b), anchor=xa(k), range=y_C, tickvals=[0, 1, 2], ticktext=["1", "2", "3"],
                                **{**axis_style, "showgrid": False})              # grid(axis="x")
            continue
        layout[xkey] = dict(domain=dom_x(b), anchor=ya(k), range=x_range, tickvals=lv, ticktext=lv_labels, **x_style)
        if k in MATCH_X:
            layout[xkey]["matches"] = MATCH_X[k]
        if k == "E":                                                               # symlog: transformed values
            dec = np.arange(0, 7)
            tv = np.r_[-symlog(10.0**dec[::-1]), 0, symlog(10.0**dec)]
            tt = [f"−10<sup>{d}</sup>" for d in dec[::-1]] + ["0"] + [f"10<sup>{d}</sup>" for d in dec]
            keep = (np.abs(tv) >= symlog(10)) | (tv == 0)                          # decades >= linthresh and 0
            layout[ykey] = dict(domain=dom_y(b), anchor=xa(k), range=y_E_t, tickvals=tv[keep],
                                ticktext=list(np.array(tt)[keep]), **axis_style)
        else:
            layout[ykey] = dict(domain=dom_y(b), anchor=xa(k), range=y_ranges[k], **y_ticks(k, y_ranges[k]),
                                **axis_style)
        if k in MATCH_Y:
            layout[ykey]["matches"] = MATCH_Y[k]

    # Legends: one plotly legend per item, at the position of the matplotlib text (the text of a plotly item
    # starts 40 px to the right of the x of the legend), and a common frame with rounded corners
    legends, shapes, n_leg = {}, [], 0

    def legend_items(g, names):
        nonlocal n_leg
        ids = []
        for name, t in zip(names, g["legend_texts"]):
            n_leg += 1
            lid = "legend" if n_leg == 1 else f"legend{n_leg}"
            legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)
            ids.append(lid)
        b = g["legend"]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                           line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H)))
        return dict(zip(names, ids))

    leg = legend_items(geo, leg_names)
    leg_C = legend_items(geo["C"], legend_C)
    leg_D = legend_items(geo["D"], legend_D)
    LEG_N = {n: leg[name] for n, name in zip(N_ART, leg_names)}
    LEG_E1 = leg["Experiment 1"]

    fig = go.Figure()

    def hline(k, y, color, lw, dash, name=None, legend=None):
        """
        axhline (zorder 0: below the grid and the data). name/legend: legend entry (empty trace with the same style; the
        line belongs to its group, so it is hidden with it)
        """
        shapes.append(dict(type="line", xref=f"{xa(k)} domain", yref=ya(k), x0=0, x1=1, y0=y, y1=y, layer="below",
                           line=dict(color=color, width=pt(lw), dash=dash),
                           **(dict(legend=legend, legendgroup=name) if name else {})))
        if name:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name, xaxis=xa(k), yaxis=ya(k),
                                     line=dict(color=color, width=pt(lw), dash=dash), legend=legend, legendgroup=name))

    def stat_lines(k, tab, what, fmt=".3f", ns=N_ART, ci="CI 95%", tr=lambda v: v):
        """
        Curves per n_art (sns.lineplot(hue="n_art", marker="o", lw=2)): bands (alpha 0.2) below all the lines. tr:
        transformation of the y axis (E: symlog); the hover gives the untransformed value
        """
        hover = (f"%{{fullData.name}}<br>SNR: %{{x:g}} dB<br>{what}: %{{customdata[0]:{fmt}}}<br>"
                 f"{ci}: (%{{customdata[1]:{fmt}}}, %{{customdata[2]:{fmt}}})<extra></extra>")
        for n in ns:
            if n not in tab.index.get_level_values(0):
                continue
            d = tab.loc[n]
            x = d.index.to_numpy(float)
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=tr(np.r_[d["hi"], d["lo"][::-1]]), mode="lines",
                                     fill="toself", fillcolor=rgba(N_COLORS[n], 0.2), line=dict(width=0),
                                     hoverinfo="skip", showlegend=False, xaxis=xa(k), yaxis=ya(k),
                                     legend=LEG_N[n], legendgroup=f"n{n}"))
        for n in ns:
            if n not in tab.index.get_level_values(0):
                continue
            d = tab.loc[n]
            fig.add_trace(go.Scatter(x=d.index.to_numpy(float), y=tr(d["mean"].to_numpy()), mode="lines+markers",
                                     name=leg_names[n - 1], line=dict(color=N_COLORS[n], width=pt(2)),
                                     marker=dict(size=pt(6), color=N_COLORS[n], line=dict(color="black", width=pt(0.75))),
                                     customdata=d[["mean", "lo", "hi"]].to_numpy(), hovertemplate=hover,
                                     showlegend=(k == "A"), xaxis=xa(k), yaxis=ya(k), legend=LEG_N[n],
                                     legendgroup=f"n{n}"))

    def exp1_line(k, s, what, fmt=".3f", tr=lambda v: v):
        """Experiment 1: dashed gray line without markers (over the curves)"""
        fig.add_trace(go.Scatter(x=s.index.to_numpy(float), y=tr(s.to_numpy(float)), mode="lines", name="Experiment 1",
                                 line=dict(color=C_EXP1, width=pt(1.5), dash=dashed(1.5)), customdata=s.to_numpy(),
                                 hovertemplate=(f"Experiment 1<br>SNR: %{{x:g}} dB<br>{what}: "
                                                f"%{{customdata:{fmt}}}<extra></extra>"),
                                 showlegend=(k == "A"), xaxis=xa(k), yaxis=ya(k), legend=LEG_E1, legendgroup="exp1"))

    # --- A, A', A'' ---
    stat_lines("A", t_det, "Detection rate")
    exp1_line("A", e_det, "Detection rate")
    hline("A", 0.5, C_EXP1, 1.5, dotted(1.5))
    stat_lines("A1", t_ddet, "Δ total vs exp. 1")
    hline("A1", 0, C_EXP1, 1.5, dashed(1.5))
    stat_lines("A2", t_mrg, "Merged", ns=(2, 3))
    hline("A2", 0, C_EXP1, 1.5, dotted(1.5))

    # --- B, B', B'' ---
    stat_lines("B", t_own, "Own detection rate")
    exp1_line("B", e_det, "Detection rate")
    hline("B", 0.5, C_EXP1, 1.5, dotted(1.5))
    stat_lines("B1", t_down, "Δ own vs exp. 1")
    hline("B1", 0, C_EXP1, 1.5, dashed(1.5))
    stat_lines("B2", t_bor, "Total − own", ns=(2, 3))
    hline("B2", 0, C_EXP1, 1.5, dotted(1.5))

    # --- C: SNR at 50 % ---
    # Experiment 1: CI band (zorder 0) and line (zorder 1), below the grid as in matplotlib
    shapes.append(dict(type="rect", xref=xa("C"), yref=f"{ya('C')} domain", x0=ci1[0], x1=ci1[1], y0=0, y1=1,
                       layer="below", fillcolor=rgba(C_EXP1, 0.15), line=dict(width=0), legend=leg_C[legend_C[2]],
                       legendgroup="exp1_C"))
    shapes.append(dict(type="line", xref=xa("C"), yref=f"{ya('C')} domain", x0=est1, x1=est1, y0=0, y1=1,
                       layer="below", line=dict(color=C_EXP1, width=pt(1.5), dash=dashed(1.5)),
                       legend=leg_C[legend_C[2]], legendgroup="exp1_C"))
    for row, n in enumerate(N_ART):
        for col, dy, filled, name in (("detected", -DY, True, legend_C[0]), ("det_own", DY, False, legend_C[1])):
            est, (lo, hi) = s50[(n, col)]
            marker = dict(size=pt(8), color=N_COLORS[n] if filled else "white",
                          line=dict(color="black" if filled else N_COLORS[n], width=pt(1.5)))
            kw = dict(name=name, xaxis=xa("C"), yaxis=ya("C"), showlegend=False, legend=leg_C[name],
                      legendgroup=name)
            if np.isfinite(est):
                fig.add_trace(go.Scatter(
                    x=[est], y=[row + dy], mode="markers", marker=marker,
                    error_x=dict(type="data", symmetric=False, array=[hi - est], arrayminus=[est - lo],
                                 color=N_COLORS[n], thickness=pt(1.5), width=pt(3)),
                    hovertemplate=(f"{leg_names[n - 1]} — {name}<br>SNR at 50 % detection: {est:.1f} dB<br>"
                                   f"CI 95%: ({lo:.1f}, {hi:.1f}) dB<extra></extra>"), **kw))
            else:                                                                  # out of range: arrow at the border
                side = "below" if est < 0 else "above"
                fig.add_trace(go.Scatter(
                    x=[lv.min() if est < 0 else lv.max()], y=[row + dy], mode="markers",
                    marker={**marker, "symbol": "triangle-left" if est < 0 else "triangle-right"},
                    hovertemplate=f"{leg_names[n - 1]} — {name}<br>50 % crossing {side} the SNR range<extra></extra>",
                    **kw))
    # Legend entries of C (marker shape; the colors are those of A)
    for name, mfc, mec in ((legend_C[0], C_EXP1, "black"), (legend_C[1], "white", C_EXP1)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, xaxis=xa("C"), yaxis=ya("C"),
                                 marker=dict(size=pt(8), color=mfc, line=dict(color=mec, width=pt(1.5))),
                                 legend=leg_C[name], legendgroup=name))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=legend_C[2], xaxis=xa("C"), yaxis=ya("C"),
                             line=dict(color=C_EXP1, width=pt(1.5), dash=dashed(1.5)), legend=leg_C[legend_C[2]],
                             legendgroup="exp1_C"))

    # --- D, D' ---
    stat_lines("D", t_ds, "ΔSNR (dB)", fmt=".2f")
    exp1_line("D", e_ds, "ΔSNR (dB)", fmt=".2f")
    hline("D", 0, C_EXP1, 1.5, dotted(1.5), name=legend_D[0], legend=leg_D[legend_D[0]])
    stat_lines("D1", t_dds, "Δ vs exp. 1 (dB)", fmt=".2f")
    hline("D1", 0, C_EXP1, 1.5, dashed(1.5))

    # --- E: error of the local mean RMS (symlog) ---
    stat_lines("E", t_rms, "Mean local RMS error (%) (median)", fmt=".1f", ci="IQR", tr=symlog)
    exp1_line("E", e_rms, "Mean local RMS error (%) (median)", fmt=".1f", tr=symlog)
    hline("E", 0, C_EXP1, 1.5, dotted(1.5))

    # --- F, F' ---
    stat_lines("F", t_net, "Net ΔSNR (dB)", fmt=".2f")
    exp1_line("F", e_net, "Net ΔSNR (dB)", fmt=".2f")
    hline("F", 0, C_EXP1, 1.5, dotted(1.5))
    stat_lines("F1", t_dnet, "Δ vs exp. 1 (dB)", fmt=".2f")
    hline("F1", 0, C_EXP1, 1.5, dashed(1.5))

    # --- Titles and labels ---
    def ann(s, b, size, rotate=False):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size)), textangle=-90 if rotate else 0)

    annotations = [ann(spec["suptitle"], geo["suptitle"], 16)]
    for k in AX:
        s = spec[k]
        if s.get("title"):
            annotations.append(ann(f"<b>{s['title']}</b>", geo[k]["title"], 14))
        if s.get("xlabel"):
            annotations.append(ann(s["xlabel"], geo[k]["xlabel"], 12))
        if s.get("ylabel"):
            annotations.append(ann(s["ylabel"], geo[k]["ylabel"], 12, rotate=True))

    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )
    return fig


# --------------------------------------------------------------------------------------------------------------------
# 5.2. Interaction between artifacts (section 6.4.2)
# --------------------------------------------------------------------------------------------------------------------

def _interaction_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_interaction_artifacts figure as matplotlib computes it (layout="constrained"): builds the same
    figure WITHOUT data (titles, labels, scales, limits, ticks, heatmaps with their colorbar and legends) and returns
    the boxes (px from the top-left corner of the image cropped as bbox_inches="tight") of the axes, texts, colorbars
    and legends, and the ticks that matplotlib puts on each axis.
    spec: {panel: dict(title, title_pad, xlabel, ylabel, xlim, ylim, xscale, xticks, legends, lines, heatmap),
           "legend": [...], "suptitle": str}
    - legends: [dict(names, loc, bbox, ncol, title)]; lines: curves [(x, y)] for loc="best"
    - heatmap: dict(xticklabels, yticklabels, lim, cbar_label) (as sns.heatmap with cbar_ax = inset_axes)
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])
    sub_mrg = gs[2, 1].subgridspec(2, 1, height_ratios=[2, 1])
    axs = dict(A=fig.add_subplot(gs[0, 0]))
    axs["B"] = fig.add_subplot(gs[0, 1], sharex=axs["A"])
    axs["C"] = fig.add_subplot(gs[1, 0])
    axs["D"] = fig.add_subplot(gs[1, 1], sharey=axs["C"])
    axs["E"] = fig.add_subplot(gs[2, 0], sharex=axs["A"])
    axs["F"] = fig.add_subplot(sub_mrg[0])
    axs["F1"] = fig.add_subplot(sub_mrg[1], sharex=axs["F"])
    fig.suptitle(spec["suptitle"], fontsize=16)

    cbars = {}
    for k, ax in axs.items():
        s = spec[k]
        for x, y in s.get("lines", []):
            ax.plot(x, y)
        if s.get("xscale"):
            ax.set_xscale(s["xscale"])
        if s.get("xticks"):
            ax.set_xticks(*s["xticks"])
            ax.minorticks_off()
        if s.get("heatmap"):                                                       # as sns.heatmap
            h = s["heatmap"]
            nr, nc = len(h["yticklabels"]), len(h["xticklabels"])
            ax.set_xlim(0, nc)
            ax.set_ylim(nr, 0)
            ax.set_xticks(np.arange(nc) + 0.5, h["xticklabels"])
            ax.set_yticks(np.arange(nr) + 0.5, h["yticklabels"], va="center")
            cax = ax.inset_axes([1.02, 0, 0.03, 1])
            cbars[k] = fig.colorbar(ScalarMappable(Normalize(-h["lim"], h["lim"]), cmap="RdBu"), cax=cax,
                                    label=h["cbar_label"])
        else:
            ax.set_xlim(*s["xlim"])
            ax.set_ylim(*s["ylim"])
        ax.set_title(s.get("title", ""), fontsize=14, fontweight="bold", pad=s.get("title_pad"))
        ax.set_xlabel(s.get("xlabel", ""), fontsize=12)
        ax.set_ylabel(s.get("ylabel", ""), fontsize=12)
        for lg in s.get("legends", []):
            leg = ax.legend([Line2D([], []) for _ in lg["names"]], lg["names"], fontsize=10, loc=lg["loc"],
                            bbox_to_anchor=lg.get("bbox"), ncol=lg.get("ncol", 1), title=lg.get("title"),
                            title_fontproperties=dict(weight="bold", size=10))
            ax.add_artist(leg)                                                     # several legends per panel (E)

    # Common legend: between the titles of A and B and the panels (position after the constrained layout)
    canvas.draw()
    pos_a, pos_b = axs["A"].get_position(), axs["B"].get_position()
    fig_leg = fig.legend([Line2D([], []) for _ in spec["legend"]], spec["legend"], loc="lower center",
                         bbox_to_anchor=((pos_a.x0 + pos_b.x1) / 2, pos_a.y1 + 0.003), bbox_transform=fig.transFigure,
                         ncol=len(spec["legend"]), fontsize=10)

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    def ticks(axis):
        """Visible ticks (within the limits) and their labels"""
        lo, hi = sorted(axis.get_view_interval())
        tl = [(t.get_loc(), t.label1.get_text()) for t in axis.get_major_ticks()
              if lo - 1e-9*(hi - lo) <= t.get_loc() <= hi + 1e-9*(hi - lo)]
        return [v for v, _ in tl], [s for _, s in tl]

    def legend_box(leg):
        title = leg.get_title()
        return dict(frame=box(leg), texts=[box(t) for t in leg.get_texts()],
                    title=box(title) if title.get_text() else None)

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle),
               legend=legend_box(fig_leg))
    for k, ax in axs.items():
        out[k] = dict(ax=box(ax), xlabel=box(ax.xaxis.label), ylabel=box(ax.yaxis.label), title=box(ax.title),
                      xticks=ticks(ax.xaxis), yticks=ticks(ax.yaxis),
                      legends=[legend_box(a) for a in ax.artists if isinstance(a, type(fig_leg))])
        if k in cbars:
            cb = cbars[k]
            out[k]["cbar"] = dict(ax=box(cb.ax), label=box(cb.ax.yaxis.label), ticks=ticks(cb.ax.yaxis))
    return out


def plot_interaction_artifacts(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_interaction_artifacts (matplotlib/seaborn, section 6.4.2 of
    evaluation_denoise.ipynb): same layout (layout="constrained", figsize 14 x 12 in), sizes, fonts, colors and
    positions. The figure is cropped as the inline backend does (bbox_inches="tight"), so its size matches the
    matplotlib image.
    Interaction between artifacts in experiment 2:
    - A: change in the detection (accompanied − alone), total and own, vs gap (95 % CI between clean signals)
    - B: proportion of merged artifacts vs gap
    - C, D: gap × SNR heatmaps (triplets) of the change in the detection (%) and in the ΔSNR (dB), with the values in
      each cell; the cells of D without data, hatched
    - E: change in the detection of the weak artifact next to a strong one (unequal condition) vs gap
    - F: ΔSNR of the detected artifacts merged, not merged and alone vs SNR; F': NMSE between the two artifacts of the
      pairs (median and 25-75 percentiles)
    In A, B and E, vertical lines at 2·TOL, REF_INNER and REF_OUTER.
    Interactivity: hover with the values (gap, SNR, value, CI...) in every panel, also in the cells of the heatmaps;
    click on the common legend to hide/show a number of artifacts (A and B) or a reference line (A, B and E), on the
    upper legend of E for each SNR of the weak artifact and on the legend of F for each group (F and F'); the "Total /
    Own detection" legends only show the line style. Zoom: A, B and E move the gap axis together (sharex), F and F' the
    SNR one, and the two heatmaps both axes (in matplotlib they only share the y axis).
    Uses FS, TOL (evaluation), REF_INNER, REF_OUTER (remove_noise) and SNR_STRONG (module constant).
    The positions of the axes, texts, colorbars and legends and the ticks are taken from the constrained layout of
    matplotlib applied to the same figure without data (_interaction_geometry).
    Limits: the y axis of F' has a step of 25 (plotly only picks 1-2-5 steps), so it is fixed and does not adapt when
    zooming; the plotly grid always stays below the bands; the bootstrap CIs use a fixed seed (seaborn does not), so
    they can differ slightly from the matplotlib version.

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 2 (read_results("experiment_2.csv")). It is not modified
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Data (as the matplotlib version) ===
    N_COLORS = {1: "#AFA9EC", 2: "#7F77DD", 3: "#3C3489"}                        # same colors as in figure 1
    SNR_WEAK_COLORS = {-10: "#085041", -5: "#1D9E75", 0: "#5DCAA5"}              # green ramp: stronger weak = darker
    C_MERGED, C_NOTMERGED, C_ALONE = "#D85A30", "#888780", N_COLORS[1]
    PAL_MERGE = {"Alone": C_ALONE, "Not merged": C_NOTMERGED, "Merged": C_MERGED}
    GRAY = "#808080"                                                               # "gray"
    RED = "#d62728"                                                                # "tab:red"

    mix2 = results_df[results_df["condition"] == "equal"]

    PER_ART = ["art_ids", "snrs_db", "k0", "detected", "merged", "dsnr_db", "attenuated", "err_rms_mean_loc_out"]
    art2 = mix2.explode(PER_ART).reset_index(drop=True)                            # copy: results_df is not modified
    for col in ("snrs_db", "k0", "dsnr_db", "err_rms_mean_loc_out"):
        art2[col] = pd.to_numeric(art2[col])
    for col in ("detected", "merged", "attenuated"):
        art2[col] = art2[col].astype(bool).astype(float)
    art2["n_art"] = art2["n_art"].astype(int)

    # Detection of the same artifact alone (same signal, artifact, position and SNR) and own detection
    KEY = ["clean_id", "art_ids", "k0", "snrs_db"]
    solo = (art2.loc[art2["n_art"] == 1, KEY + ["detected", "dsnr_db"]]
            .drop_duplicates(KEY).rename(columns={"detected": "det_solo", "dsnr_db": "dsnr_solo"}))
    art2 = art2.merge(solo, on=KEY, how="left")
    art2["det_own"] = ((art2["detected"] == 1) & ((art2["merged"] == 0) | (art2["det_solo"] == 1))).astype(float)

    # Accompanied artifacts and interaction effects (accompanied − alone)
    acc = art2[art2["n_art"] > 1].copy()
    acc["gap_ms"] = acc["gap_s"] * 1000
    acc["d_det"] = acc["detected"] - acc["det_solo"]
    acc["d_own"] = acc["det_own"] - acc["det_solo"]

    lv = np.sort(art2["snrs_db"].unique())
    gaps_ms = np.sort(acc["gap_ms"].unique())
    REF_LINES = {f"2·TOL = {2 * TOL / FS * 1000:g} ms": 2 * TOL / FS * 1000,
                 f"REF_INNER = {REF_INNER * 1000:g} ms": REF_INNER * 1000,
                 f"REF_OUTER = {REF_OUTER * 1000:g} ms": REF_OUTER * 1000}
    REF_DASHES = dict(zip(REF_LINES, (DASH_PATTERNS["--"], DASH_PATTERNS["-."], [5, 2, 1, 2])))

    # A, B: one row per clean signal, gap and n_art (mean over artifacts, positions and SNR)
    gdet = acc.groupby(["clean_id", "gap_ms", "n_art"], as_index=False)[["d_det", "d_own"]].mean()
    gmrg = acc.groupby(["clean_id", "gap_ms", "n_art"], as_index=False)["merged"].mean()

    # C, D: heatmaps of the triplets (mean over artifacts, positions and signals)
    tri = acc[acc["n_art"] == 3]
    hdet = tri.pivot_table(index="snrs_db", columns="gap_ms", values="d_det", aggfunc="mean") * 100
    tri_det = tri[(tri["detected"] == 1) & (tri["det_solo"] == 1)].assign(d_ds=lambda d: d["dsnr_db"] - d["dsnr_solo"])
    hds = (tri_det.pivot_table(index="snrs_db", columns="gap_ms", values="d_ds", aggfunc="mean")
           .reindex(index=hdet.index, columns=hdet.columns))

    # E: weak artifacts next to the strong one (unequal condition)
    mixU = results_df[results_df["condition"] == "unequal"]
    artU = mixU.explode(["art_ids", "snrs_db", "is_strong", "k0", "k1", "detected", "merged"]).reset_index(drop=True)
    for col in ("snrs_db", "k0", "k1"):
        artU[col] = pd.to_numeric(artU[col])
    for col in ("detected", "merged", "is_strong"):
        artU[col] = artU[col].astype(bool)
    strong = artU.loc[artU["is_strong"], ["id", "k0", "k1"]].rename(columns={"k0": "k0_s", "k1": "k1_s"})
    weak = artU[~artU["is_strong"]].merge(strong, on="id", how="inner")
    gap_n = (weak["gap_s"] * FS).round()
    dist = np.maximum(weak["k0"] - weak["k1_s"], weak["k0_s"] - weak["k1"])
    weak = weak[np.isclose(dist, gap_n)].copy()
    solo_all = (results_df[results_df["n_art"] == 1].explode(["art_ids", "snrs_db", "k0", "detected"])
                .astype({"snrs_db": float, "k0": float})
                .assign(det_solo=lambda d: d["detected"].astype(bool))
                .drop_duplicates(KEY)[KEY + ["det_solo"]])
    weak = weak.astype({"snrs_db": float, "k0": float}).merge(solo_all, on=KEY, how="left")
    weak["gap_ms"] = weak["gap_s"] * 1000
    weak["d_det"] = weak["detected"].astype(float) - weak["det_solo"].astype(float)
    weak["det_own"] = weak["detected"] & (~weak["merged"] | weak["det_solo"])
    weak["d_own"] = weak["det_own"].astype(float) - weak["det_solo"].astype(float)
    gmask = weak.groupby(["clean_id", "gap_ms", "snrs_db"], as_index=False)[["d_det", "d_own"]].mean()

    # F: ΔSNR of the detected ones per group (one row per signal); F': NMSE between the two artifacts of the pairs
    det_all = art2[art2["detected"] == 1].assign(
        group=lambda d: np.select([d["n_art"] == 1, d["merged"] == 1], ["Alone", "Merged"], "Not merged"))
    fds = det_all.groupby(["clean_id", "snrs_db", "group"], as_index=False)["dsnr_db"].mean()
    NMSE_FLOOR = -100                                                              # dB: below = no change
    pairs = mix2[mix2["n_art"] == 2].copy()
    pairs["snr"] = pairs["snrs_db"].map(lambda s: float(s[0]))
    pairs["any_merged"] = pairs["merged"].map(lambda m: bool(any(m)))
    pairs["nmse_btw"] = pairs["nmse_between_db"].map(lambda v: float(v[0])).clip(lower=NMSE_FLOOR)
    pairs["group"] = np.where(pairs["any_merged"], "Merged", "Not merged")

    # --- Estimator and band per (hue, x) (as seaborn) ---
    N_BOOT = 1000
    rng_ci = np.random.default_rng(1)

    def mean_ci(v):
        """Mean and bootstrap 95 % CI of the observations (errorbar=("ci", 95), n_boot=1000 of seaborn)"""
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        n = len(v)
        if n == 0:
            return (np.nan, np.nan, np.nan)
        boots = v[rng_ci.integers(0, n, (N_BOOT, n))].mean(1)
        return (v.mean(), *np.percentile(boots, [2.5, 97.5]))

    def median_pi(v):
        """Median and 25-75 percentiles (estimator="median", errorbar=("pi", 50))"""
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        return (np.median(v), *np.percentile(v, [25, 75])) if len(v) else (np.nan, np.nan, np.nan)

    def stat_table(df, hue, x, col, func=mean_ci):
        """Estimator, lower band and upper band of col per (hue, x)"""
        gb = df.dropna(subset=[col]).groupby([hue, x])[col]
        return pd.DataFrame([func(g) for _, g in gb], index=pd.MultiIndex.from_tuples(gb.groups.keys()),
                            columns=["mean", "lo", "hi"])

    t_det, t_own = stat_table(gdet, "n_art", "gap_ms", "d_det"), stat_table(gdet, "n_art", "gap_ms", "d_own")
    t_mrg = stat_table(gmrg, "n_art", "gap_ms", "merged")
    t_mdet, t_mown = stat_table(gmask, "snrs_db", "gap_ms", "d_det"), stat_table(gmask, "snrs_db", "gap_ms", "d_own")
    t_fds = stat_table(fds, "group", "snrs_db", "dsnr_db")
    t_btw = stat_table(pairs, "group", "snr", "nmse_btw", func=median_pi)


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dashes = lambda pattern, lw: mpl_dashes(pattern, lw, PT)

    # Limits (matplotlib autoscale: data + 5 %; axhline/axvline count; gap in log scale)
    vals = lambda *tabs: [t[["mean", "lo", "hi"]].to_numpy().ravel() if isinstance(t, pd.DataFrame)
                          else np.asarray(t, float).ravel() for t in tabs]
    x_gap = list(10 ** np.array(mpl_range(np.log10(np.r_[gaps_ms, list(REF_LINES.values())]))))
    y_A = mpl_range(*vals(t_det, t_own, [0]))
    y_B = mpl_range(*vals(t_mrg))
    y_E = mpl_range(*vals(t_mdet, t_mown, [0]))
    x_F = mpl_range(lv)
    y_F = mpl_range(*vals(t_fds))
    y_F1 = mpl_range(*vals(t_btw))
    lim_C = np.nanmax(np.abs(hdet.to_numpy()))                                     # symmetric scale around 0
    lim_D = np.nanmax(np.abs(hds.to_numpy()))

    gap_labels = [f"{g:g}" for g in gaps_ms]
    snr_labels = [f"{s:g}" for s in hdet.index]
    lv_labels = [f"{v:g}" for v in lv]
    leg_names = [f"{n} artifacts" for n in (2, 3)] + list(REF_LINES)
    style_names = ["Total detection", "Own detection"]
    weak_names = [f"Weak at {s:g} dB" for s in SNR_WEAK_COLORS]
    gap_x = dict(xscale="log", xlim=x_gap, xticks=(gaps_ms, gap_labels), xlabel="Gap between artifacts (ms)")
    hm = lambda lim, label: dict(xticklabels=gap_labels, yticklabels=snr_labels, lim=lim, cbar_label=label)
    spec = dict(
        suptitle="Interaction between artifacts", legend=leg_names,
        A=dict(title="Change in detection vs Artifact separation", title_pad=34, ylabel="Δ detection rate", ylim=y_A,
               legends=[dict(names=style_names, loc="lower right")], **gap_x),
        B=dict(title="Merged proportion vs Artifact separation", title_pad=34, ylabel="Merged proportion", ylim=y_B,
               **gap_x),
        C=dict(title="Change in detection (triplets)", xlabel="Gap between artifacts (ms)", ylabel="SNR (dB)",
               heatmap=hm(lim_C, "Δ detection rate (%)")),
        D=dict(title="Change in ΔSNR (triplets)", xlabel="Gap between artifacts (ms)",
               heatmap=hm(lim_D, "Δ ΔSNR (dB)")),
        E=dict(title="Change in detection vs Masking", title_pad=48, ylabel="Δ detection rate (weak)", ylim=y_E,
               legends=[dict(names=weak_names, loc="lower center", bbox=(0.5, 1.0), ncol=len(weak_names),
                             title=f"Next to a strong artifact ({SNR_STRONG} dB)"),
                        dict(names=style_names, loc="upper right")], **gap_x),
        F=dict(title="Effect of merging on removal", ylabel="ΔSNR (dB)", xlim=x_F, ylim=y_F, xticks=(lv, lv_labels),
               legends=[dict(names=list(PAL_MERGE), loc="best", title="Detected artifacts")],
               lines=[(t_fds.loc[g].index.to_numpy(float), t_fds.loc[g]["mean"].to_numpy()) for g in PAL_MERGE
                      if g in t_fds.index.get_level_values(0)]),
        F1=dict(xlabel="SNR (dB)", ylabel="NMSE between (dB)", xlim=x_F, ylim=y_F1, xticks=(lv, lv_labels)),
    )

    fig_w, fig_h = 14, 12
    geo = _interaction_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(3 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _interaction_geometry(fig_w, fig_h, dpi, spec)
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H
    dom_x = lambda b: [X(b["x0"]), X(b["x1"])]
    dom_y = lambda b: [Y(b["bottom"]), Y(b["top"])]

    def y_ticks(k, axis_range):
        """
        Ticks of the linear y axis with the matplotlib step. Steps 1-2-5·10^k: nticks (they adapt to the zoom; plotly
        takes the 1-2-5 step strictly greater than range/nticks). Steps 2.5·10^k (plotly does not pick them):
        fixed step
        """
        v, labels = geo[k]["yticks"]
        step = v[1] - v[0]
        mant = step / 10**np.floor(np.log10(step))
        if np.isclose(mant, 2.5):
            dec = max(len(s.split(".")[1]) if "." in s else 0 for s in labels)
            return dict(tick0=0, dtick=step, tickformat=f".{dec}f")
        return dict(nticks=int(np.floor(abs(axis_range[1] - axis_range[0]) / step)) + 1,
                    tickformatstops=MPL_TICKFORMAT)

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none")                                       # sns.despine: no top/right axes
    x_style = dict(**axis_style, ticklabelstandoff=round(-2.2 * PT))
    AX = dict(A=1, B=2, C=3, D=4, E=5, F=6, F1=7)                                  # plotly axis number
    xa = lambda k: "x" if AX[k] == 1 else f"x{AX[k]}"
    ya = lambda k: "y" if AX[k] == 1 else f"y{AX[k]}"
    MATCH_X = dict(B="x", E="x", D="x3", F1="x6")                                  # sharex (+ x of the heatmaps)
    MATCH_Y = dict(D="y3")                                                         # sharey

    layout = {}
    for k, n in AX.items():
        b = geo[k]["ax"]
        xkey, ykey = ("xaxis", "yaxis") if n == 1 else (f"xaxis{n}", f"yaxis{n}")
        s = spec[k]
        if "heatmap" in s:                                                         # heatmaps: cells 0..n, no grid
            nr, nc = hdet.shape
            layout[xkey] = dict(domain=dom_x(b), anchor=ya(k), range=[0, nc], tickvals=np.arange(nc) + 0.5,
                                ticktext=gap_labels, **{**x_style, "showgrid": False})
            layout[ykey] = dict(domain=dom_y(b), anchor=xa(k), range=[nr, 0], tickvals=np.arange(nr) + 0.5,
                                ticktext=snr_labels, **{**axis_style, "showgrid": False})
        else:
            if s.get("xscale") == "log":
                layout[xkey] = dict(type="log", range=list(np.log10(x_gap)), **x_style)
            else:
                layout[xkey] = dict(range=x_F, **x_style)
            layout[xkey].update(domain=dom_x(b), anchor=ya(k), tickvals=s["xticks"][0], ticktext=s["xticks"][1])
            layout[ykey] = dict(domain=dom_y(b), anchor=xa(k), range=s["ylim"], **y_ticks(k, s["ylim"]), **axis_style)
        if k in MATCH_X:
            layout[xkey]["matches"] = MATCH_X[k]
        if k in MATCH_Y:
            layout[ykey]["matches"] = MATCH_Y[k]

    # Legends: one plotly legend per item, at the position of the matplotlib text (the text of a plotly item
    # starts 40 px to the right of the x of the legend), and a common frame with rounded corners
    legends, shapes, annotations, n_leg = {}, [], [], 0

    def ann(s, b, size, rotate=False):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size)), textangle=-90 if rotate else 0)

    def legend_items(g, names, clickable=True):
        """g: legend of the geometry (frame, texts and title). clickable=False: it only shows the style"""
        nonlocal n_leg
        ids = []
        for name, t in zip(names, g["texts"]):
            n_leg += 1
            lid = "legend" if n_leg == 1 else f"legend{n_leg}"
            legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30,
                                **({} if clickable else dict(itemclick=False, itemdoubleclick=False)))
            ids.append(lid)
        b = g["frame"]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                           line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H)))
        if g["title"]:
            annotations.append(ann(f"<b>{g['title_text']}</b>", g["title"], 10))
        return dict(zip(names, ids))

    geo["E"]["legends"][0]["title_text"] = spec["E"]["legends"][0]["title"]
    geo["F"]["legends"][0]["title_text"] = spec["F"]["legends"][0]["title"]
    leg = legend_items(geo["legend"], leg_names)
    leg_A = legend_items(geo["A"]["legends"][0], style_names, clickable=False)
    leg_weak = legend_items(geo["E"]["legends"][0], weak_names)
    leg_E = legend_items(geo["E"]["legends"][1], style_names, clickable=False)
    leg_F = legend_items(geo["F"]["legends"][0], list(PAL_MERGE))

    fig = go.Figure()

    # --- Reference lines (zorder 0: below the grid) ---
    def hline(k, y):
        shapes.append(dict(type="line", xref=f"{xa(k)} domain", yref=ya(k), x0=0, x1=1, y0=y, y1=y, layer="below",
                           line=dict(color=GRAY, width=pt(1.5), dash=dashes(DASH_PATTERNS[":"], 1.5))))

    def gap_refs(k):
        """Vertical lines at the characteristic distances of the algorithm (group of their legend entry)"""
        for name, x in REF_LINES.items():
            shapes.append(dict(type="line", xref=xa(k), yref=f"{ya(k)} domain", x0=x, x1=x, y0=0, y1=1, layer="below",
                               line=dict(color=RED, width=pt(2), dash=dashes(REF_DASHES[name], 2)),
                               legend=leg[name], legendgroup=name))

    for name in REF_LINES:                                                         # entries (lw 1.2, alpha 0.7)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name, xaxis="x", yaxis="y",
                                 line=dict(color=rgba(RED, 0.7), width=pt(1.2), dash=dashes(REF_DASHES[name], 1.2)),
                                 legend=leg[name], legendgroup=name))

    # --- Curves with band (sns.lineplot, marker="o") ---
    def stat_lines(k, tabs, colors, groups, names, what, x_what, fmt=".3f", band="CI 95%", show=None, legend=None):
        """
        tabs: [(table, dash)]; colors/groups/names: per hue level. Bands (alpha 0.2) below all the lines.
        show: level -> legend entry (the first style); legend: level -> plotly legend
        """
        x_unit = " ms" if x_what == "Gap" else " dB"
        for tab, _ in tabs:
            for h in colors:
                if h not in tab.index.get_level_values(0):
                    continue
                d = tab.loc[h]
                x = d.index.to_numpy(float)
                fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[d["hi"], d["lo"][::-1]], mode="lines",
                                         fill="toself", fillcolor=rgba(colors[h], 0.2), line=dict(width=0),
                                         hoverinfo="skip", showlegend=False, xaxis=xa(k), yaxis=ya(k),
                                         legend=legend[h] if legend else None, legendgroup=groups[h]))
        for i, (tab, dash) in enumerate(tabs):
            for h in colors:
                if h not in tab.index.get_level_values(0):
                    continue
                d = tab.loc[h]
                label = names[h] + (f" — {style_names[i]}" if len(tabs) > 1 else "")
                hover = (f"{label}<br>{x_what}: %{{x:g}}{x_unit}<br>{what}: %{{customdata[0]:{fmt}}}<br>"
                         f"{band}: (%{{customdata[1]:{fmt}}}, %{{customdata[2]:{fmt}}})<extra></extra>")
                dash_h = dash(h) if callable(dash) else dash
                fig.add_trace(go.Scatter(x=d.index.to_numpy(float), y=d["mean"].to_numpy(), mode="lines+markers",
                                         name=names[h], line=dict(color=colors[h], width=pt(2), dash=dash_h),
                                         marker=dict(size=pt(6), color=colors[h],
                                                     line=dict(color="black", width=pt(0.75))),
                                         customdata=d[["mean", "lo", "hi"]].to_numpy(), hovertemplate=hover,
                                         showlegend=bool(show) and i == 0 and h in show, xaxis=xa(k), yaxis=ya(k),
                                         legend=legend[h] if legend else None, legendgroup=groups[h]))

    def style_entries(k, lg, color):
        """Total / Own detection entries (style only: they do not hide anything)"""
        for name, dash in zip(style_names, ("solid", dashes(DASH_PATTERNS["--"], 2))):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name, xaxis=xa(k), yaxis=ya(k),
                                     line=dict(color=color, width=pt(2), dash=dash), legend=lg[name],
                                     legendgroup=f"{k}_{name}"))

    N_GROUP = {n: f"n{n}" for n in (2, 3)}
    N_NAMES = {n: f"{n} artifacts" for n in (2, 3)}
    N_LEG = {n: leg[N_NAMES[n]] for n in (2, 3)}
    own_dash = dashes(DASH_PATTERNS["--"], 2)

    # --- A: Δ detection vs gap ---
    stat_lines("A", [(t_det, "solid"), (t_own, own_dash)], {n: N_COLORS[n] for n in (2, 3)}, N_GROUP, N_NAMES,
               "Δ detection rate", "Gap", show=N_NAMES, legend=N_LEG)
    hline("A", 0)
    gap_refs("A")
    style_entries("A", leg_A, "black")

    # --- B: proportion of merged artifacts vs gap ---
    stat_lines("B", [(t_mrg, "solid")], {n: N_COLORS[n] for n in (2, 3)}, N_GROUP, N_NAMES, "Merged proportion",
               "Gap", legend=N_LEG)
    gap_refs("B")

    # --- C, D: gap × SNR heatmaps (triplets) ---
    cmap = colormaps["RdBu"]
    nr, nc = hdet.shape
    yc, xc = np.meshgrid(np.arange(nr) + 0.5, np.arange(nc) + 0.5, indexing="ij")  # centers of the cells
    cell_h = (geo["C"]["ax"]["bottom"] - geo["C"]["ax"]["top"]) / nr               # px per cell (height)
    for k, tab, lim, what, unit, cax in (("C", hdet, lim_C, "Δ detection rate", "%", "coloraxis"),
                                         ("D", hds, lim_D, "Δ ΔSNR", "dB", "coloraxis2")):
        z = tab.to_numpy(float)
        custom = np.dstack(np.broadcast_arrays(np.array(gap_labels)[None, :], np.array(snr_labels)[:, None]))
        fig.add_trace(go.Heatmap(x=np.arange(nc + 1), y=np.arange(nr + 1), z=z, coloraxis=cax, xgap=1, ygap=1,
                                 customdata=custom, xaxis=xa(k), yaxis=ya(k),
                                 hovertemplate=(f"Gap: %{{customdata[0]}} ms<br>SNR: %{{customdata[1]}} dB<br>"
                                                f"{what}: %{{z:.2f}} {unit}<extra></extra>")))
        ok = ~np.isnan(z)
        if (~ok).any():                                                            # no data: hatched (patch.set_hatch)
            r, c = np.where(~ok)
            fig.add_trace(go.Bar(x=c + 0.5, y=np.ones(len(r)), base=r, width=1, xaxis=xa(k), yaxis=ya(k),
                                 marker=dict(color="white", line=dict(width=0),
                                             pattern=dict(shape="/", size=dpi / 17, solidity=0.3, fgcolor="#d3d3d3",
                                                          bgcolor="white")),
                                 hoverinfo="skip", showlegend=False))
        # Values (annot=True, fmt=".2f"): dark or white text depending on the luminance of the cell, as seaborn
        rgba_cells = cmap(Normalize(-lim, lim)(z[ok]))
        fig.add_trace(go.Scatter(x=xc[ok], y=yc[ok] - 0.1 * em / cell_h, mode="text",
                                 text=[f"{v:.2f}" for v in z[ok]], xaxis=xa(k), yaxis=ya(k), hoverinfo="skip",
                                 showlegend=False,
                                 textfont=dict(size=em, color=["#262626" if relative_luminance(c) > .408 else "white"
                                                               for c in rgba_cells])))
        # Colorbar (inset_axes [1.02, 0, 0.03, 1], without edge) and its label
        cb = geo[k]["cbar"]
        layout[cax] = dict(cmin=-lim, cmax=lim, colorscale=mpl_colorscale("RdBu"), colorbar=dict(
            x=X(cb["ax"]["x0"]), xanchor="left", y=Y(cb["ax"]["bottom"]), yanchor="bottom", xpad=0, ypad=0,
            thickness=cb["ax"]["x1"] - cb["ax"]["x0"], thicknessmode="pixels",
            len=cb["ax"]["bottom"] - cb["ax"]["top"], lenmode="pixels", outlinewidth=0,
            ticks="outside", ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
            tickvals=cb["ticks"][0], ticktext=["\u2009" + t for t in cb["ticks"][1]]))    # thin space: matplotlib pad
        annotations.append(ann(spec[k]["heatmap"]["cbar_label"], cb["label"], 10, rotate=True))

    # --- E: masking of the weak artifact vs gap ---
    W_GROUP = {s: f"w{s:g}" for s in SNR_WEAK_COLORS}
    W_NAMES = dict(zip(SNR_WEAK_COLORS, weak_names))
    stat_lines("E", [(t_mdet, "solid"), (t_mown, own_dash)], SNR_WEAK_COLORS, W_GROUP, W_NAMES,
               "Δ detection rate (weak)", "Gap", show=W_NAMES, legend={s: leg_weak[W_NAMES[s]] for s in W_NAMES})
    hline("E", 0)
    gap_refs("E")
    style_entries("E", leg_E, GRAY)

    # --- F: ΔSNR of merged / not merged / alone ---
    G_GROUP = {g: f"g_{g}" for g in PAL_MERGE}
    G_NAMES = {g: g for g in PAL_MERGE}
    G_LEG = {g: leg_F[g] for g in PAL_MERGE}
    alone_dash = lambda g: dashes([4, 2], 2) if g == "Alone" else "solid"
    stat_lines("F", [(t_fds, alone_dash)], PAL_MERGE, G_GROUP, G_NAMES, "ΔSNR (dB)", "SNR", fmt=".2f",
               show=G_NAMES, legend=G_LEG)

    # --- F': NMSE between the two artifacts (pairs) ---
    stat_lines("F1", [(t_btw, "solid")], {g: PAL_MERGE[g] for g in ("Not merged", "Merged")}, G_GROUP, G_NAMES,
               "NMSE between (dB) (median)", "SNR", fmt=".1f", band="IQR", legend=G_LEG)

    # --- Titles and labels ---
    annotations.append(ann(spec["suptitle"], geo["suptitle"], 16))
    for k in AX:
        s = spec[k]
        if s.get("title"):
            annotations.append(ann(f"<b>{s['title']}</b>", geo[k]["title"], 14))
        if s.get("xlabel"):
            annotations.append(ann(s["xlabel"], geo[k]["xlabel"], 12))
        if s.get("ylabel"):
            annotations.append(ann(s["ylabel"], geo[k]["ylabel"], 12, rotate=True))

    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )
    return fig


# --------------------------------------------------------------------------------------------------------------------
# 5.3. Mixture-level performance (section 6.4.3)
# --------------------------------------------------------------------------------------------------------------------

def _mixture_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_mixture_performance figure as matplotlib computes it (layout="constrained"): builds the same
    figure WITHOUT data (titles, labels, limits, ticks, legends, numbers above the bars and inset) and returns the boxes
    (px from the top-left corner of the image cropped as bbox_inches="tight") of the axes, texts and legends, and the
    ticks that matplotlib puts on each axis.
    spec: {panel: dict(title, title_pad, xlabel, ylabel, xlim, ylim, xticks, legend, texts), "I": dict(pos, xlim, ylim),
           "suptitle": str}
    - legend: dict(names, ncol, title) (loc="lower center", bbox_to_anchor=(0.5, 1.0)); texts: [(x, y, text)]
    - I: inset of C (inset_axes at the position pos, fraction of the axes of C)
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(spec["suptitle"], fontsize=16)
    axs = dict(A=fig.add_subplot(gs[0, 0]), B=fig.add_subplot(gs[0, 1]), C=fig.add_subplot(gs[1, 0]))
    axs["D"] = fig.add_subplot(gs[1, 1], sharex=axs["C"])

    texts = {}
    for k, ax in axs.items():
        s = spec[k]
        ax.set_xlim(*s["xlim"])
        ax.set_ylim(*s["ylim"])
        if s.get("xticks"):
            ax.set_xticks(*s["xticks"])
        ax.set_title(s["title"], fontsize=14, fontweight="bold", pad=s.get("title_pad"))
        ax.set_xlabel(s["xlabel"], fontsize=12)
        ax.set_ylabel(s["ylabel"], fontsize=12)
        if s.get("legend"):
            lg = s["legend"]
            ax.legend([Line2D([], []) for _ in lg["names"]], lg["names"], fontsize=10, loc="lower center",
                      bbox_to_anchor=(0.5, 1.0), ncol=lg["ncol"], title=lg.get("title"),
                      title_fontproperties=dict(weight="bold", size=10))
        texts[k] = [ax.text(x, y, t, ha="center", va="bottom", fontsize=8, fontweight="bold")
                    for x, y, t in s.get("texts", [])]
    axs["D"].tick_params(labelbottom=True)
    axs["I"] = axs["C"].inset_axes(spec["I"]["pos"])
    axs["I"].set_xlim(*spec["I"]["xlim"])
    axs["I"].set_ylim(*spec["I"]["ylim"])
    axs["I"].tick_params(labelsize=8)

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    def ticks(axis):
        """Visible ticks (within the limits) and their labels"""
        lo, hi = sorted(axis.get_view_interval())
        tl = [(t.get_loc(), t.label1.get_text()) for t in axis.get_major_ticks()
              if lo - 1e-9*(hi - lo) <= t.get_loc() <= hi + 1e-9*(hi - lo)]
        return [v for v, _ in tl], [s for _, s in tl]

    def legend_box(leg):
        if leg is None:
            return None
        title = leg.get_title()
        return dict(frame=box(leg), texts=[box(t) for t in leg.get_texts()],
                    title=box(title) if title.get_text() else None)

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle))
    for k, ax in axs.items():
        out[k] = dict(ax=box(ax), xlabel=box(ax.xaxis.label), ylabel=box(ax.yaxis.label), title=box(ax.title),
                      xticks=ticks(ax.xaxis), yticks=ticks(ax.yaxis), legend=legend_box(ax.get_legend()),
                      texts=[box(t) for t in texts.get(k, [])])
    return out


def plot_mixture_performance(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_mixture_performance (matplotlib/seaborn, section 6.4.3 of
    evaluation_denoise.ipynb): same layout (layout="constrained", figsize 14 x 8 in), sizes, fonts, colors and
    positions. The figure is cropped as the inline backend does (bbox_inches="tight"), so its size matches the
    matplotlib image.
    Mixture-level performance in experiment 2 (equal condition):
    - A: proportion of mixtures by how many artifacts are detected (stacked bars per SNR and number of artifacts)
    - B: proportion of mixtures with all the artifacts detected vs SNR of each artifact (95 % CI between clean
      signals), the expected one without interaction (dashed) and all the mixtures together (black)
    - C: global RMS error peak before (orange) and after (blue) the algorithm vs global SNR of the mixture (median and
      IQR), with an inset from ZOOM_FROM dB
    - D: as B, vs global SNR of the mixture
    Interactivity: hover with the values in every panel (in A, the percentage of each bar segment); click on the legend
    of A to hide/show a category of the bars, on the legend of B for each number of artifacts, all the mixtures or the
    expected lines (B and D) and on the legend of C for each curve (C and its inset). Zoom: C and D move the global SNR
    axis together (sharex); the inset has its own zoom. The y axis of A is fixed (proportion from 0 to 1).
    The positions of the axes, texts and legends and the ticks are taken from the constrained layout of matplotlib
    applied to the same figure without data (_mixture_geometry).
    Limits: the plotly grid always stays below the bands (in matplotlib, above); in A it is drawn as shapes so that it
    stays over the bars, as in matplotlib; the red rectangle of C follows the zoom of C, but the arms that join it to
    the inset are fixed; the bootstrap CIs use a fixed seed (seaborn does not), so they can differ slightly from the
    matplotlib version.

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 2 (read_results("experiment_2.csv")). It is not modified
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Data (as the matplotlib version) ===
    N_COLORS = {1: "#AFA9EC", 2: "#7F77DD", 3: "#3C3489"}                        # same colors as in figures 1 and 2
    MISS_COLORS = {0: "#378ADD", 1: "#85B7EB", 2: "#B5D4F4"}                     # by how many are left undetected
    MISS_LABELS = {0: "All detected", 1: "All but one", 2: "All but two"}
    C_NONE = "#D3D1C7"
    BEFORE_COLORS = {1: "#FAC775", 2: "#EF9F27", 3: "#BA7517"}                   # orange: before (darker: more n_art)
    AFTER_COLORS = {1: "#85B7EB", 2: "#378ADD", 3: "#185FA5"}                    # blue: after
    GRAY, RED, BLACK = "#808080", "#ff0000", "#000000"                             # "gray", "red", "black"
    BIN_W = 5                                                                      # dB, global SNR bins
    MIN_N = 1                                                                      # min. mixtures per bin and n_art
    ZOOM_FROM = -5                                                                 # dB: the inset goes from here on
    INSET_POS = [0.36, 0.28, 0.61, 0.66]                                           # inset position (axes fraction)

    mix2 = results_df[results_df["condition"] == "equal"].copy()                  # same SNR for all the artifacts
    mix2["snr_db"] = pd.to_numeric(mix2["snr_db"])
    mix2["n_art"] = mix2["n_art"].astype(int)
    mix2["n_detected"] = mix2["n_detected"].astype(int)
    lv = np.sort(mix2["snr_db"].unique())

    # A: proportion of mixtures per number of detected artifacts (row: SNR; column: detected)
    W_BAR = 0.26                                                                   # width of each bar
    x_pos = np.arange(len(lv))
    offsets = {1: -W_BAR, 2: 0, 3: W_BAR}
    fracs = {n: pd.crosstab(d["snr_db"], d["n_detected"], normalize="index").reindex(index=lv, columns=range(n + 1),
                                                                                     fill_value=0)
             for n, d in mix2.groupby("n_art")}

    # B: all detected; expected without interaction = each artifact detected when alone (same signal, position, SNR)
    mix2["all_det"] = (mix2["n_detected"] == mix2["n_art"]).astype(float)
    KEY = ["clean_id", "art_ids", "k0", "snrs_db"]
    art_m = mix2.explode(["art_ids", "snrs_db", "k0", "detected"]).astype({"snrs_db": float, "k0": float})
    solo = (art_m.loc[art_m["n_art"] == 1, KEY + ["detected"]].drop_duplicates(KEY)
            .assign(det_solo=lambda d: d["detected"].astype(bool))[KEY + ["det_solo"]])
    expected = (art_m.merge(solo, on=KEY, how="left")
                .groupby("id")["det_solo"].all().astype(float).rename("all_det_solo"))
    mix2 = mix2.merge(expected, left_on="id", right_index=True, how="left")
    all_clean = mix2.groupby(["clean_id", "snr_db", "n_art"], as_index=False)[["all_det", "all_det_solo"]].mean()
    all_pooled = mix2.groupby(["clean_id", "snr_db"], as_index=False)["all_det"].mean()

    # C: global SNR bins
    pk = mix2.dropna(subset=["snr_global_mix_db"]).copy()
    edges = np.arange(np.floor(pk["snr_global_mix_db"].min() / BIN_W) * BIN_W,
                      pk["snr_global_mix_db"].max() + BIN_W, BIN_W)
    pk["g_center"] = pd.cut(pk["snr_global_mix_db"], edges).apply(lambda b: b.mid).astype(float)
    pk = pk[pk.groupby(["g_center", "n_art"])["id"].transform("size") >= MIN_N]   # almost empty bins dropped

    # D: same bins as C; one row per clean signal, bin and n_art
    allg_clean = pk.groupby(["clean_id", "g_center", "n_art"], as_index=False)[["all_det", "all_det_solo"]].mean()
    allg_pooled = pk.groupby(["clean_id", "g_center"], as_index=False)["all_det"].mean()

    # --- Estimator and band per (hue, x) (as seaborn) ---
    N_BOOT = 1000
    rng_ci = np.random.default_rng(1)

    def mean_ci(v):
        """Mean and bootstrap 95 % CI of the observations (errorbar=("ci", 95), n_boot=1000 of seaborn)"""
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        n = len(v)
        if n == 0:
            return (np.nan, np.nan, np.nan)
        boots = v[rng_ci.integers(0, n, (N_BOOT, n))].mean(1)
        return (v.mean(), *np.percentile(boots, [2.5, 97.5]))

    def median_pi(v):
        """Median and 25-75 percentiles (estimator="median", errorbar=("pi", 50))"""
        v = np.asarray(v, float)
        v = v[~np.isnan(v)]
        return (np.median(v), *np.percentile(v, [25, 75])) if len(v) else (np.nan, np.nan, np.nan)

    def stat_table(df, hue, x, col, func=mean_ci):
        """Estimator, lower band and upper band of col per (hue, x)"""
        gb = df.dropna(subset=[col]).groupby([hue, x])[col]
        return pd.DataFrame([func(g) for _, g in gb], index=pd.MultiIndex.from_tuples(gb.groups.keys()),
                            columns=["mean", "lo", "hi"])

    def expected_table(df, x):
        """Expected line without interaction (errorbar=None: only the mean), mixtures of 2 and 3 artifacts"""
        return df[df["n_art"] > 1].groupby(["n_art", x])["all_det_solo"].mean().to_frame("mean")

    POOL = "all"
    t_B = stat_table(all_clean, "n_art", "snr_db", "all_det")
    e_B = expected_table(all_clean, "snr_db")
    p_B = stat_table(all_pooled.assign(h=POOL), "h", "snr_db", "all_det")
    t_bef = stat_table(pk, "n_art", "g_center", "err_rms_max_glob_mix", func=median_pi)
    t_aft = stat_table(pk, "n_art", "g_center", "err_rms_max_glob_out", func=median_pi)
    t_D = stat_table(allg_clean, "n_art", "g_center", "all_det")
    e_D = expected_table(allg_clean, "g_center")
    p_D = stat_table(allg_pooled.assign(h=POOL), "h", "g_center", "all_det")

    # Natural zoom from ZOOM_FROM dB: limits with the IQRs of the zoomed range
    x0_z, x1_z = ZOOM_FROM, pk["g_center"].max() + BIN_W / 2
    zoom = pk[pk["g_center"] >= ZOOM_FROM]
    q = zoom.groupby(["g_center", "n_art"])[["err_rms_max_glob_mix", "err_rms_max_glob_out"]].quantile([0.25, 0.75])
    y_lo, y_hi = q.min().min(), q.max().max()
    pad = 0.1 * (y_hi - y_lo if y_hi > y_lo else 1)
    y0_z, y1_z = y_lo - pad, y_hi + pad


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dashes = lambda pattern, lw: mpl_dashes(pattern, lw, PT)

    # Limits (matplotlib autoscale: data + 5 %; the bars start at 0 without margin; axhline counts)
    vals = lambda *tabs: [t.to_numpy().ravel() if isinstance(t, pd.DataFrame) else np.asarray(t, float).ravel()
                          for t in tabs]
    half = W_BAR + W_BAR * 0.92 / 2
    x_A = mpl_range([x_pos[0] - half, x_pos[-1] + half])
    y_A = [0, 1.05 * max(f.sum(axis=1).max() for f in fracs.values())]
    x_B = mpl_range(lv)
    y_B = mpl_range(*vals(t_B, e_B, p_B))
    x_C = mpl_range(pk["g_center"].unique())
    y_C = mpl_range(*vals(t_bef, t_aft, [0]))
    y_D = mpl_range(*vals(t_D, e_D, p_D))

    lv_labels = [f"{v:g}" for v in lv]
    N_NAMES = {n: f"{n} artifact{'s' if n > 1 else ''}" for n in (1, 2, 3)}
    A_NAMES = [MISS_LABELS[m] for m in (0, 1, 2)] + ["None"]
    POOL_NAME, EXP_NAME = "All mixtures", "Expected without interaction"
    B_NAMES = [N_NAMES[1], POOL_NAME, N_NAMES[2], EXP_NAME, N_NAMES[3]]            # column by column
    C_NAMES = [f"{s} · {n}" for n in (1, 2, 3) for s in ("Before", "After")]       # column by column: B1, A1, B2...
    TXT_Y = 1.01                                                                   # numbers above the bars
    spec = dict(
        suptitle="Mixture-level performance",
        A=dict(title="Number of detected artifacts per mixture", title_pad=42, xlabel="SNR (dB)",
               ylabel="Proportion of mixtures", xlim=x_A, ylim=y_A, xticks=(x_pos, lv_labels),
               legend=dict(names=A_NAMES, ncol=len(A_NAMES), title="Artifacts detected"),
               texts=[(x + offsets[n], TXT_Y, str(n)) for n in (1, 2, 3) for x in x_pos[:1]]),
        B=dict(title="All detected rate vs SNR", title_pad=45, xlabel="SNR (dB)", ylabel="All artifacts detected",
               xlim=x_B, ylim=y_B, xticks=(lv, lv_labels), legend=dict(names=B_NAMES, ncol=3)),
        C=dict(title="Global Max RMS before / after processing vs Global SNR", title_pad=58,
               xlabel="Global SNR of the mixture (dB)", ylabel="Global RMS peak error (%)", xlim=x_C, ylim=y_C,
               legend=dict(names=C_NAMES, ncol=3, title="Median ± IQR · artifacts per mixture")),
        D=dict(title="All detected rate vs Global SNR", xlabel="Global SNR of the mixture (dB)",
               ylabel="All artifacts detected", xlim=x_C, ylim=y_D),
        I=dict(pos=INSET_POS, xlim=[x0_z, x1_z], ylim=[y0_z, y1_z]),
    )

    fig_w, fig_h = 14, 8
    geo = _mixture_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(3 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _mixture_geometry(fig_w, fig_h, dpi, spec)
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H
    dom_x = lambda b: [X(b["x0"]), X(b["x1"])]
    dom_y = lambda b: [Y(b["bottom"]), Y(b["top"])]

    def auto_ticks(ticks, axis_range):
        """
        Automatic ticks with the matplotlib step. Steps 1-2-5·10^k: nticks (they adapt to the zoom; plotly
        takes the 1-2-5 step strictly greater than range/nticks). Steps 2.5·10^k (plotly does not pick them):
        fixed step
        """
        v, labels = ticks
        step = v[1] - v[0]
        mant = step / 10**np.floor(np.log10(step))
        if np.isclose(mant, 2.5):
            dec = max(len(s.split(".")[1]) if "." in s else 0 for s in labels)
            return dict(tick0=0, dtick=step, tickformat=f".{dec}f")
        return dict(nticks=int(np.floor(abs(axis_range[1] - axis_range[0]) / step)) + 1,
                    tickformatstops=MPL_TICKFORMAT)

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none")                                       # sns.despine: no top/right axes
    x_style = dict(**axis_style, ticklabelstandoff=round(-2.2 * PT))
    AX = dict(A=1, B=2, C=3, D=4, I=5)                                             # plotly axis number (I: inset)
    xa = lambda k: "x" if AX[k] == 1 else f"x{AX[k]}"
    ya = lambda k: "y" if AX[k] == 1 else f"y{AX[k]}"

    x_axes = dict(A=dict(range=x_A, tickvals=x_pos, ticktext=lv_labels, showgrid=False),   # grid(axis="y")
                  B=dict(range=x_B, tickvals=lv, ticktext=lv_labels),
                  C=dict(range=x_C, **auto_ticks(geo["C"]["xticks"], x_C)),
                  D=dict(range=x_C, **auto_ticks(geo["D"]["xticks"], x_C), matches="x3"),  # sharex with C
                  I=dict(range=spec["I"]["xlim"], **auto_ticks(geo["I"]["xticks"], spec["I"]["xlim"])))
    y_axes = dict(A=dict(range=y_A, tickvals=geo["A"]["yticks"][0], ticktext=geo["A"]["yticks"][1], showgrid=False,
                         fixedrange=True),
                  B=dict(range=y_B, **auto_ticks(geo["B"]["yticks"], y_B)),
                  C=dict(range=y_C, **auto_ticks(geo["C"]["yticks"], y_C)),
                  D=dict(range=y_D, **auto_ticks(geo["D"]["yticks"], y_D)),
                  I=dict(range=spec["I"]["ylim"], **auto_ticks(geo["I"]["yticks"], spec["I"]["ylim"])))
    inset_style = dict(mirror=True, linecolor=RED, linewidth=pt(1.2), tickfont=dict(size=pt(8)))  # red frame, ticks 8
    layout = {}
    for k, n in AX.items():
        b = geo[k]["ax"]
        xkey, ykey = ("xaxis", "yaxis") if n == 1 else (f"xaxis{n}", f"yaxis{n}")
        layout[xkey] = {**x_style, **(inset_style if k == "I" else {}), "domain": dom_x(b), "anchor": ya(k),
                        **x_axes[k]}
        layout[ykey] = {**axis_style, **(inset_style if k == "I" else {}), "domain": dom_y(b), "anchor": xa(k),
                        **y_axes[k]}

    # Legends: one plotly legend per item, at the position of the matplotlib text (the text of a plotly item
    # starts 40 px to the right of the x of the legend), and a common frame with rounded corners
    legends, shapes, annotations, n_leg = {}, [], [], 0

    def ann(s, b, size, rotate=False):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size)), textangle=-90 if rotate else 0)

    def legend_items(k, names):
        """Legend of panel k (frame, texts and title): name -> id of the plotly legend"""
        nonlocal n_leg
        g, ids = geo[k]["legend"], []
        for name, t in zip(names, g["texts"]):
            n_leg += 1
            lid = "legend" if n_leg == 1 else f"legend{n_leg}"
            legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)
            ids.append(lid)
        b = g["frame"]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                           line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H)))
        if g["title"]:
            annotations.append(ann(f"<b>{spec[k]['legend']['title']}</b>", g["title"], 10))
        return dict(zip(names, ids))

    leg_A, leg_B, leg_C = legend_items("A", A_NAMES), legend_items("B", B_NAMES), legend_items("C", C_NAMES)

    fig = go.Figure()

    # --- A: number of detected artifacts per mixture ---
    # Color by how many artifacts are left undetected (all detected at the bottom, none at the top)
    for name, color in zip(A_NAMES, [MISS_COLORS[m] for m in (0, 1, 2)] + [C_NONE]):        # entries (Patch)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name, line=dict(color=color, width=pt(8)),
                                 legend=leg_A[name], legendgroup=f"A_{name}", xaxis="x", yaxis="y"))
    for n, frac in fracs.items():
        bottom = np.zeros(len(lv))
        for k_det in range(n, -1, -1):                                             # n. of detected: from all to none
            name = "None" if k_det == 0 else MISS_LABELS[n - k_det]
            h = frac[k_det].to_numpy()
            fig.add_trace(go.Bar(x=x_pos + offsets[n], y=h, base=bottom.copy(), width=W_BAR * 0.92, name=name,
                                 marker=dict(color=C_NONE if k_det == 0 else MISS_COLORS[n - k_det],
                                             line=dict(color="white", width=pt(0.5))),
                                 customdata=np.c_[lv, h * 100], showlegend=False, legend=leg_A[name],
                                 legendgroup=f"A_{name}", xaxis="x", yaxis="y",
                                 hovertemplate=(f"SNR: %{{customdata[0]:g}} dB<br>{N_NAMES[n]}<br>"
                                                f"{name}: %{{customdata[1]:.1f}} %<extra></extra>")))
            bottom += h
    # Number of artifacts above each bar (matplotlib text centered in its box)
    b = geo["A"]["ax"]
    for n, t in zip((1, 2, 3), geo["A"]["texts"]):
        cy = (t["top"] + t["bottom"]) / 2 - 0.1 * pt(8)
        y_txt = y_A[0] + (b["bottom"] - cy) / (b["bottom"] - b["top"]) * (y_A[1] - y_A[0])
        fig.add_trace(go.Scatter(x=x_pos + offsets[n], y=np.full(len(lv), y_txt), mode="text", text=[str(n)] * len(lv),
                                 textfont=dict(size=pt(8), color=N_COLORS[n], weight="bold"), cliponaxis=False,
                                 hoverinfo="skip", showlegend=False, xaxis="x", yaxis="y"))
    # Horizontal grid over the bars (in matplotlib, zorder 1.5 > bars)
    for v in geo["A"]["yticks"][0]:
        shapes.append(dict(type="line", xref="x domain", yref="y", x0=0, x1=1, y0=v, y1=v, layer="above",
                           line=dict(color="#b0b0b0", width=pt(0.8))))

    # --- Curves with band (sns.lineplot) ---
    def curve(k, d, color, group, lg, name, what, x_what, fmt=".3f", band=None, lw=2, dash="solid", markers=True,
              showlegend=False):
        """d: table (index x; mean, and lo/hi if band) -> (band trace or None, line trace)"""
        x = d.index.to_numpy(float)
        common = dict(xaxis=xa(k), yaxis=ya(k), legend=lg, legendgroup=group)
        band_tr = None
        if band:
            band_tr = go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[d["hi"], d["lo"][::-1]], mode="lines", fill="toself",
                                 fillcolor=rgba(color, 0.2), line=dict(width=0), hoverinfo="skip", showlegend=False,
                                 **common)
        hover = f"{name}<br>{x_what}: %{{x:g}} dB<br>{what}: %{{customdata[0]:{fmt}}}"
        if band:
            hover += f"<br>{band}: (%{{customdata[1]:{fmt}}}, %{{customdata[2]:{fmt}}})"
        line_tr = go.Scatter(x=x, y=d["mean"].to_numpy(), mode="lines+markers" if markers else "lines", name=name,
                             line=dict(color=color, width=pt(lw), dash=dash),
                             marker=dict(size=pt(6), color=color, line=dict(color="black", width=pt(0.75))),
                             customdata=d[["mean", "lo", "hi"] if band else ["mean"]].to_numpy(),
                             hovertemplate=hover + "<extra></extra>", showlegend=showlegend, **common)
        return band_tr, line_tr

    def add(pairs):
        """Bands (zorder 1) below all the lines (zorder 2; the ones with a higher zorder, at the end)"""
        fig.add_traces([b for b, _ in pairs if b is not None] + [l for _, l in pairs])

    def hline(k, y):
        """axhline(zorder 0): below the grid"""
        shapes.append(dict(type="line", xref=f"{xa(k)} domain", yref=ya(k), x0=0, x1=1, y0=y, y1=y, layer="below",
                           line=dict(color=GRAY, width=pt(1.5), dash=dashes(DASH_PATTERNS[":"], 1.5))))

    def detected_panel(k, tab, exp, pool, x_what):
        """B, D: all detected per number of artifacts (95 % CI), expected without interaction and all the mixtures"""
        show = k == "B"
        what = "All artifacts detected"
        pairs = [curve(k, tab.loc[n], N_COLORS[n], f"n{n}", leg_B[N_NAMES[n]], N_NAMES[n], what, x_what,
                       band="CI 95%", showlegend=show) for n in (1, 2, 3)]
        pairs += [curve(k, exp.loc[n], N_COLORS[n], "exp", leg_B[EXP_NAME], f"{EXP_NAME} ({N_NAMES[n]})", what,
                        x_what, lw=1.5, dash=dashes(DASH_PATTERNS["--"], 1.5), markers=False) for n in (2, 3)]
        pairs.append(curve(k, pool.loc[POOL], BLACK, "pool", leg_B[POOL_NAME], POOL_NAME, what, x_what,
                           band="CI 95%", lw=2.5, showlegend=show))               # zorder 4: on top
        add(pairs)

    # --- B: all detected vs SNR of each artifact ---
    detected_panel("B", t_B, e_B, p_B, "SNR")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=EXP_NAME, legend=leg_B[EXP_NAME], legendgroup="exp",
                             line=dict(color=BLACK, width=pt(1.5), dash=dashes(DASH_PATTERNS["--"], 1.5)),
                             xaxis="x2", yaxis="y2"))

    # --- C: global RMS peak before / after (and inset) ---
    for k in ("C", "I"):
        add([curve(k, tab.loc[n], colors[n], f"{when}{n}", leg_C[f"{when} · {n}"], f"{when} · {n}",
                   "Global RMS peak error (%) (median)", "Global SNR", fmt=".1f", band="IQR", showlegend=k == "C")
             for tab, colors, when in ((t_bef, BEFORE_COLORS, "Before"), (t_aft, AFTER_COLORS, "After"))
             for n in (1, 2, 3)])
        hline(k, 0)

    # Rectangle of the zoomed area (follows the zoom of C) and arms (top corners of the rectangle -> bottom corners
    # of the inset; fixed, in figure coordinates)
    red = dict(color=RED, width=pt(1.2))
    shapes.append(dict(type="rect", xref="x3", yref="y3", x0=x0_z, x1=x1_z, y0=y0_z, y1=y1_z, layer="above", line=red))
    b, bi = geo["C"]["ax"], geo["I"]["ax"]
    px = lambda v: b["x0"] + (v - x_C[0]) / (x_C[1] - x_C[0]) * (b["x1"] - b["x0"])
    py = lambda v: b["bottom"] - (v - y_C[0]) / (y_C[1] - y_C[0]) * (b["bottom"] - b["top"])
    for x_data, x_ins in ((x0_z, bi["x0"]), (x1_z, bi["x1"])):
        shapes.append(dict(type="line", xref="paper", yref="paper", x0=X(px(x_data)), y0=Y(py(y1_z)), x1=X(x_ins),
                           y1=Y(bi["bottom"]), layer="above", line=red))

    # --- D: all detected vs global SNR of the mixture ---
    detected_panel("D", t_D, e_D, p_D, "Global SNR")

    # --- Titles and labels ---
    annotations.append(ann(spec["suptitle"], geo["suptitle"], 16))
    for k in ("A", "B", "C", "D"):
        annotations.append(ann(f"<b>{spec[k]['title']}</b>", geo[k]["title"], 14))
        annotations.append(ann(spec[k]["xlabel"], geo[k]["xlabel"], 12))
        annotations.append(ann(spec[k]["ylabel"], geo[k]["ylabel"], 12, rotate=True))

    # plot_bgcolor almost white: if it equals paper_bgcolor, plotly draws no backgrounds and the inset would be
    # transparent (set_facecolor("white") in matplotlib)
    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False), barmode="overlay",
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="#fffffe", hovermode="closest", template="none"
    )
    return fig


# --------------------------------------------------------------------------------------------------------------------
# 5.4. Conservation with several artifacts (section 6.4.4)
# --------------------------------------------------------------------------------------------------------------------

def _conservation_multiple_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_conservation_multiple figure as matplotlib computes it (layout="constrained"): builds the same
    figure WITHOUT data (titles, labels, limits, ticks and legends) and returns the boxes (px from the top-left corner
    of the image cropped as bbox_inches="tight") of the axes, texts and legends, and the ticks that matplotlib puts on
    each axis.
    spec: {panel: dict(title, xlabel, ylabel, xlim, ylim, xticks, legend), "suptitle": str}
    - legend: dict(names, loc, title)
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2)
    fig.suptitle(spec["suptitle"], fontsize=16)
    axs = dict(A=fig.add_subplot(gs[0, 0]))
    axs["B"] = fig.add_subplot(gs[0, 1], sharex=axs["A"])
    axs["C"], axs["D"] = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    for k, ax in axs.items():
        s = spec[k]
        ax.set_xlim(*s["xlim"])
        ax.set_ylim(*s["ylim"])
        if s.get("xticks"):
            ax.set_xticks(*s["xticks"])
        ax.set_title(s["title"], fontsize=14, fontweight="bold")
        ax.set_xlabel(s["xlabel"], fontsize=12)
        ax.set_ylabel(s["ylabel"], fontsize=12)
        lg = s["legend"]
        ax.legend([Line2D([], []) for _ in lg["names"]], lg["names"], fontsize=10, loc=lg["loc"],
                  title=lg.get("title"), title_fontproperties=dict(weight="bold", size=10))

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    def ticks(axis):
        """Visible ticks (within the limits) and their labels"""
        lo, hi = sorted(axis.get_view_interval())
        tl = [(t.get_loc(), t.label1.get_text()) for t in axis.get_major_ticks()
              if lo - 1e-9*(hi - lo) <= t.get_loc() <= hi + 1e-9*(hi - lo)]
        return [v for v, _ in tl], [s for _, s in tl]

    def legend_box(leg):
        title = leg.get_title()
        return dict(frame=box(leg), texts=[box(t) for t in leg.get_texts()],
                    title=box(title) if title.get_text() else None)

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle))
    for k, ax in axs.items():
        out[k] = dict(ax=box(ax), xlabel=box(ax.xaxis.label), ylabel=box(ax.yaxis.label), title=box(ax.title),
                      xticks=ticks(ax.xaxis), yticks=ticks(ax.yaxis), legend=legend_box(ax.get_legend()))
    return out


def plot_conservation_multiple(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_conservation_multiple (matplotlib/seaborn, section 6.4.4 of
    evaluation_denoise.ipynb): same layout (layout="constrained", figsize 14 x 8 in), sizes, fonts, colors and
    positions. The figure is cropped as the inline backend does (bbox_inches="tight"), so its size matches the
    matplotlib image.
    Conservation of the signal out of the artifact zones in experiment 2 (equal condition and controls):
    - A: ECDF of the NMSE out of the zones per number of artifacts (and control)
    - B: ECDF of the mixtures of 2 and 3 artifacts with and without some merge, and the one of 1 artifact as reference
    - C: NMSE out of the zones vs SNR per number of artifacts (median and IQR) and the control on the right
    - D: false positives per mixture vs SNR per number of artifacts (mean and 95 % CI between clean signals) and the
      control
    The dashed orange lines (A and B) mark a change of 0.01 % and 1 % of the power (-40 and -20 dB).
    Interactivity: hover with the values in every panel (in A and B, the cumulative proportion at each NMSE); click on
    the legend of each panel to hide/show a curve (with its band; in C and D, the control hides the point, its error
    bar and its reference line). Zoom: A and B move the NMSE axis together (sharex).
    The positions of the axes, texts and legends and the ticks are taken from the constrained layout of matplotlib
    applied to the same figure without data (_conservation_multiple_geometry).
    Limits: the plotly grid always stays below the bands and the reference line of the control (in matplotlib, above);
    the ECDFs with more than ECDF_EXACT values are evaluated on a grid of ECDF_STEP dB (the steps move at most
    ECDF_STEP dB, invisible without zoom) so that the HTML does not weigh tens of MB; the bootstrap CIs use a fixed
    seed (seaborn does not), so they can differ slightly from the matplotlib version.

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 2 (read_results("experiment_2.csv")). It is not modified
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Style constants ===
    N_COLORS = {1: "#AFA9EC", 2: "#7F77DD", 3: "#3C3489"}                        # same colors as the previous figures
    C_CTRL, ORANGE, LIGHTGRAY = MPL_COLORS["gray"], MPL_COLORS["tab:orange"], "#d3d3d3"
    C_MERGED, C_NOTMERGED = "#D85A30", "#5F5E5A"
    NMSE_FLOOR = -100                                                              # dB: below it (or −∞) = no change
    NMSE_REFS = ((-40, "0.01 %"), (-20, "1 %"))                                    # change of 0.01 % and 1 % of power
    ECDF_EXACT = 2000                                                              # exact ECDF up to this n. of values
    ECDF_STEP = 0.05                                                               # dB: grid of the largest ECDFs

    GROUP_LABELS = {0: "Control", 1: "1 artifact", 2: "2 artifacts", 3: "3 artifacts"}
    GROUP_COLORS = {"Control": C_CTRL, **{GROUP_LABELS[n]: c for n, c in N_COLORS.items()}}


    # === Data (as the matplotlib version) ===
    cons = results_df[results_df["condition"].isin(["equal", "Control"])].copy()
    cons["n_art"] = cons["n_art"].astype(int)                                      # 0 = control
    cons["snr_db"] = pd.to_numeric(cons["snr_db"], errors="coerce")                # control -> NaN
    cons["nmse_plot"] = cons["nmse_out_zone_db"]
    cons["group"] = cons["n_art"].map(GROUP_LABELS)

    mixc = cons[cons["n_art"] > 0]                                                 # mixtures (1, 2 and 3 artifacts)
    ctrl = cons[cons["n_art"] == 0]                                                # controls (one row per clean signal)

    # Mixtures with 2 and 3 artifacts, with and without some merge
    multi = cons[cons["n_art"] > 1].copy()
    multi["mgroup"] = np.where(multi["merged"].map(lambda m: bool(any(m))), "With merges", "Without merges")

    # x axis of C and D: SNR levels and the control on the right
    lv = np.sort(mixc["snr_db"].unique())
    step = np.min(np.diff(lv))
    x_ctrl = lv.max() + 1.5 * step
    x_sep = (lv.max() + x_ctrl) / 2                                                # separation SNRs / control
    x_ticks, x_labels = list(lv) + [x_ctrl], [f"{v:g}" for v in lv] + ["Control"]

    # D: one row per clean signal, SNR and n_art (mean over artifacts and positions); the control is already
    # one per signal
    fp_clean = mixc.groupby(["clean_id", "snr_db", "n_art"], as_index=False)["n_false"].mean()

    # --- Estimators (as seaborn) ---
    N_BOOT = 1000
    rng_ci = np.random.default_rng(1)

    def ecdf(v):
        """ECDF as sns.ecdfplot (steps after each value); with many values, evaluated on a grid"""
        v = np.sort(np.asarray(v, float))
        n = len(v)
        if n <= ECDF_EXACT:
            return v, np.arange(1, n + 1) / n
        grid = np.arange(np.floor(v[0] / ECDF_STEP), np.ceil(v[-1] / ECDF_STEP) + 1) * ECDF_STEP
        y = np.searchsorted(v, grid, side="right") / n
        keep = np.r_[True, np.diff(y) != 0]                                        # flat runs dropped
        return grid[keep], y[keep]

    def mean_ci(v):
        """Mean and bootstrap 95 % CI of the observations (errorbar=("ci", 95), n_boot=1000 of seaborn)"""
        v = np.asarray(v, float)
        boots = v[rng_ci.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
        return (v.mean(), *np.percentile(boots, [2.5, 97.5]))

    def median_pi(v):
        """Median and 25-75 percentiles (estimator="median", errorbar=("pi", 50))"""
        v = np.asarray(v, float)
        return (np.median(v), *np.percentile(v, [25, 75]))

    def stat_table(df, col, func):
        """Estimator, lower band and upper band of col per (n_art, snr_db)"""
        gb = df.dropna(subset=[col]).groupby(["n_art", "snr_db"])[col]
        return pd.DataFrame([func(g) for _, g in gb], index=pd.MultiIndex.from_tuples(gb.groups.keys()),
                            columns=["mean", "lo", "hi"])

    t_C = stat_table(mixc, "nmse_plot", median_pi)
    t_D = stat_table(fp_clean, "n_false", mean_ci)

    # Control: median and IQR (C) and mean ± 95 % CI between clean signals (D)
    q25, med, q75 = ctrl["nmse_plot"].quantile([0.25, 0.5, 0.75])
    m_fp = ctrl["n_false"].mean()
    ci_fp = 1.96 * ctrl["n_false"].std(ddof=1) / np.sqrt(len(ctrl))
    ctrl_C = dict(mean=med, lo=q25, hi=q75)
    ctrl_D = dict(mean=m_fp, lo=m_fp - ci_fp, hi=m_fp + ci_fp)


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dashes = lambda lw: mpl_dashes(DASH_PATTERNS["--"], lw, PT)

    # Limits: A and B (and C in y) fixed; C and D, matplotlib autoscale (data + 5 %; D starts at 0)
    x_NMSE = [NMSE_FLOOR - 2, 2]
    x_SNR = mpl_range([lv.min(), x_ctrl])
    y_D = [0, mpl_range(t_D.to_numpy().ravel(), list(ctrl_D.values()))[1]]

    n_per_group = cons["group"].value_counts()
    n_m = multi["mgroup"].value_counts()
    A_NAMES = [f"{g} (n = {n_per_group.get(g, 0)})" for g in GROUP_COLORS]
    B_NAMES = [f"Without merges (n = {n_m.get('Without merges', 0)})",
               f"With merges (n = {n_m.get('With merges', 0)})", "1 artifact (reference)"]
    N_NAMES = {n: f"{n} artifact{'s' if n > 1 else ''}" for n in (1, 2, 3)}
    CTRL_NAME = f"Control (n = {len(ctrl)})"
    CD_NAMES = [N_NAMES[n] for n in (1, 2, 3)] + [CTRL_NAME]
    NMSE_LABEL = "NMSE out of zones (dB)"
    spec = dict(
        suptitle="Conservation with multiple artifacts",
        A=dict(title="Distortion out of the artifact zones", xlabel=NMSE_LABEL, ylabel="Cumulative proportion",
               xlim=x_NMSE, ylim=[0, 1], legend=dict(names=A_NAMES, loc="upper left")),
        B=dict(title="Distortion out of zones: merged vs not merged", xlabel=NMSE_LABEL,
               ylabel="Cumulative proportion", xlim=x_NMSE, ylim=[0, 1],
               legend=dict(names=B_NAMES, loc="upper left", title="2 and 3 artifacts")),
        C=dict(title="NMSE out of zones vs SNR", xlabel="SNR (dB)", ylabel=NMSE_LABEL, xlim=x_SNR,
               ylim=[NMSE_FLOOR - 2, 2], xticks=(x_ticks, x_labels),
               legend=dict(names=CD_NAMES, loc="upper right", title="Median ± IQR")),
        D=dict(title="False positives vs SNR", xlabel="SNR (dB)", ylabel="False positives per mixture", xlim=x_SNR,
               ylim=y_D, xticks=(x_ticks, x_labels), legend=dict(names=CD_NAMES, loc="upper left",
                                                                 title="Mean ± CI 95 %")),
    )

    fig_w, fig_h = 14, 8
    geo = _conservation_multiple_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(3 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _conservation_multiple_geometry(fig_w, fig_h, dpi, spec)
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H

    def auto_ticks(ticks, axis_range):
        """
        Automatic ticks with the matplotlib step. Steps 1-2-5·10^k: nticks (they adapt to the zoom; plotly
        takes the 1-2-5 step strictly greater than range/nticks). Steps 2.5·10^k (plotly does not pick them):
        fixed step
        """
        v, labels = ticks
        step = v[1] - v[0]
        mant = step / 10**np.floor(np.log10(step))
        if np.isclose(mant, 2.5):
            dec = max(len(s.split(".")[1]) if "." in s else 0 for s in labels)
            return dict(tick0=0, dtick=step, tickformat=f".{dec}f")
        return dict(nticks=int(np.floor(abs(axis_range[1] - axis_range[0]) / step)) + 1,
                    tickformatstops=MPL_TICKFORMAT)

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none")                                       # sns.despine: no top/right axes
    x_style = dict(**axis_style, ticklabelstandoff=round(-2.2 * PT))
    AX = dict(A=1, B=2, C=3, D=4)                                                  # plotly axis number
    xa = lambda k: "x" if AX[k] == 1 else f"x{AX[k]}"
    ya = lambda k: "y" if AX[k] == 1 else f"y{AX[k]}"

    x_axes = dict(A=dict(range=x_NMSE, **auto_ticks(geo["A"]["xticks"], x_NMSE)),
                  B=dict(range=x_NMSE, **auto_ticks(geo["B"]["xticks"], x_NMSE), matches="x"),   # sharex with A
                  C=dict(range=x_SNR, tickvals=x_ticks, ticktext=x_labels),
                  D=dict(range=x_SNR, tickvals=x_ticks, ticktext=x_labels))
    y_axes = {k: dict(range=spec[k]["ylim"], **auto_ticks(geo[k]["yticks"], spec[k]["ylim"])) for k in AX}
    layout = {}
    for k, n in AX.items():
        b = geo[k]["ax"]
        xkey, ykey = ("xaxis", "yaxis") if n == 1 else (f"xaxis{n}", f"yaxis{n}")
        layout[xkey] = {**x_style, "domain": [X(b["x0"]), X(b["x1"])], "anchor": ya(k), **x_axes[k]}
        layout[ykey] = {**axis_style, "domain": [Y(b["bottom"]), Y(b["top"])], "anchor": xa(k), **y_axes[k]}

    # Legends: one plotly legend per item, at the position of the matplotlib text (the text of a plotly item
    # starts 40 px to the right of the x of the legend), and a common frame with rounded corners. The ids start
    # at legend2: with so many legends, plotly does not draw the default legend ("legend")
    legends, shapes, annotations, n_leg = {}, [], [], 1

    def ann(s, b, size, rotate=False):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size)), textangle=-90 if rotate else 0)

    def legend_items(k):
        """Legend of panel k (frame, texts and title): name -> id of the plotly legend"""
        nonlocal n_leg
        g, ids = geo[k]["legend"], []
        for t in g["texts"]:
            n_leg += 1
            lid = f"legend{n_leg}"
            legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)
            ids.append(lid)
        b = g["frame"]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                           line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H)))
        if g["title"]:
            annotations.append(ann(f"<b>{spec[k]['legend']['title']}</b>", g["title"], 10))
        return dict(zip(spec[k]["legend"]["names"], ids))

    leg = {k: legend_items(k) for k in AX}

    fig = go.Figure()

    # --- A and B: ECDF of the NMSE out of the zones ---
    def ecdf_trace(k, v, name, color, lw=2, dash="solid"):
        x, y = ecdf(v)
        return go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=pt(lw), dash=dash, shape="hv"),
                          name=name, legend=leg[k][name], legendgroup=f"{k}_{name}", xaxis=xa(k), yaxis=ya(k),
                          hovertemplate=f"{name}<br>NMSE ≤ %{{x:.2f}} dB<br>Proportion: %{{y:.4f}}<extra></extra>")

    def add_nmse_refs(k):
        """Vertical lines at -40 and -20 dB (over the curves, as in matplotlib) and their text at y = 0.02"""
        for thr, txt in NMSE_REFS:
            shapes.append(dict(type="line", xref=xa(k), yref=f"{ya(k)} domain", x0=thr, x1=thr, y0=0, y1=1,
                               layer="above", line=dict(color=ORANGE, width=pt(1.2), dash=dashes(1.2))))
            annotations.append(dict(text=txt, x=thr, y=0.02, xref=xa(k), yref=ya(k), xanchor="left",
                                    yanchor="bottom", xshift=0.318 * em, yshift=0.1 * em, showarrow=False,
                                    borderpad=0, borderwidth=0, font=dict(size=em, color=ORANGE)))

    # A: seaborn drawing order (hue levels reversed: the control stays on top)
    for g, name in list(zip(GROUP_COLORS, A_NAMES))[::-1]:
        fig.add_trace(ecdf_trace("A", cons.loc[cons["group"] == g, "nmse_plot"], name, GROUP_COLORS[g]))
    add_nmse_refs("A")

    # B: with / without merges (2 and 3 artifacts) and 1 artifact as reference
    for mg, name, color in (("With merges", B_NAMES[1], C_MERGED), ("Without merges", B_NAMES[0], C_NOTMERGED)):
        fig.add_trace(ecdf_trace("B", multi.loc[multi["mgroup"] == mg, "nmse_plot"], name, color))
    fig.add_trace(ecdf_trace("B", mixc.loc[mixc["n_art"] == 1, "nmse_plot"], B_NAMES[2], N_COLORS[1], lw=1.5,
                             dash=dashes(1.5)))
    add_nmse_refs("B")

    # --- C and D: value vs SNR per n_art (+ control) ---
    def snr_panel(k, tab, ctrl_v, what, band, fmt):
        """Curves per number of artifacts with their band (sns.lineplot) and the control on the right (errorbar)"""
        lg = leg[k]
        common = lambda name: dict(xaxis=xa(k), yaxis=ya(k), legend=lg[name], legendgroup=f"{k}_{name}")
        hover_band = f"{band}: (%{{customdata[1]:{fmt}}}, %{{customdata[2]:{fmt}}})"
        # Bands (zorder 1) below the control reference and all the lines
        for n in (1, 2, 3):
            d = tab.loc[n]
            x = d.index.to_numpy(float)
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[d["hi"], d["lo"][::-1]], mode="lines", fill="toself",
                                     fillcolor=rgba(N_COLORS[n], 0.2), line=dict(width=0), hoverinfo="skip",
                                     showlegend=False, **common(N_NAMES[n])))
        # Control reference (axhline, zorder 1): very long trace so that it is hidden with the legend
        span = 1e3 * (x_SNR[1] - x_SNR[0])
        fig.add_trace(go.Scatter(x=[x_SNR[0] - span, x_SNR[1] + span], y=[ctrl_v["mean"]] * 2, mode="lines",
                                 line=dict(color=C_CTRL, width=pt(2), dash=dashes(2)), hoverinfo="skip",
                                 showlegend=False, **common(CTRL_NAME)))
        # Lines with markers (zorder 2)
        for n in (1, 2, 3):
            d = tab.loc[n]
            fig.add_trace(go.Scatter(x=d.index.to_numpy(float), y=d["mean"], mode="lines+markers", name=N_NAMES[n],
                                     line=dict(color=N_COLORS[n], width=pt(2)),
                                     marker=dict(size=pt(6), color=N_COLORS[n], line=dict(color="black", width=pt(0.75))),
                                     customdata=d[["mean", "lo", "hi"]].to_numpy(),
                                     hovertemplate=(f"{N_NAMES[n]}<br>SNR: %{{x:g}} dB<br>{what}: "
                                                    f"%{{customdata[0]:{fmt}}}<br>{hover_band}<extra></extra>"),
                                     **common(N_NAMES[n])))
        # Control (zorder 3): error bar, caps and point; legend entry as the matplotlib one
        lo, hi = ctrl_v["lo"], ctrl_v["hi"]
        fig.add_trace(go.Scatter(x=[x_ctrl, x_ctrl], y=[lo, hi], mode="lines", line=dict(color=C_CTRL, width=pt(1.5)),
                                 hoverinfo="skip", showlegend=False, **common(CTRL_NAME)))
        fig.add_trace(go.Scatter(x=[x_ctrl, x_ctrl], y=[lo, hi], mode="markers", hoverinfo="skip", showlegend=False,
                                 marker=dict(symbol="line-ew", size=pt(8), line=dict(color=C_CTRL, width=pt(1))),
                                 **common(CTRL_NAME)))
        fig.add_trace(go.Scatter(x=[x_ctrl], y=[ctrl_v["mean"]], mode="markers", showlegend=False,
                                 marker=dict(size=pt(7), color=C_CTRL, line=dict(color="black", width=pt(1))),
                                 customdata=[[ctrl_v["mean"], lo, hi]],
                                 hovertemplate=(f"{CTRL_NAME}<br>{what}: %{{customdata[0]:{fmt}}}<br>{hover_band}"
                                                "<extra></extra>"), **common(CTRL_NAME)))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers", name=CTRL_NAME,
                                 line=dict(color=C_CTRL, width=pt(2), dash=dashes(2)),
                                 marker=dict(size=pt(6), color=C_CTRL, line=dict(color="black", width=pt(1))),
                                 **common(CTRL_NAME)))
        # Separation between the SNRs and the control (axvline, zorder 0: below the grid)
        shapes.append(dict(type="line", xref=xa(k), yref=f"{ya(k)} domain", x0=x_sep, x1=x_sep, y0=0, y1=1,
                           layer="below", line=dict(color=LIGHTGRAY, width=pt(1))))

    snr_panel("C", t_C, ctrl_C, "NMSE out of zones (median)", "IQR", ".1f")
    snr_panel("D", t_D, ctrl_D, "False positives per mixture (mean)", "CI 95%", ".3f")

    # --- Titles and labels ---
    annotations.append(ann(spec["suptitle"], geo["suptitle"], 16))
    for k in AX:
        annotations.append(ann(f"<b>{spec[k]['title']}</b>", geo[k]["title"], 14))
        annotations.append(ann(spec[k]["xlabel"], geo[k]["xlabel"], 12))
        annotations.append(ann(spec[k]["ylabel"], geo[k]["ylabel"], 12, rotate=True))

    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )
    return fig


# --------------------------------------------------------------------------------------------------------------------
# 5.5. Idempotence (section 6.4.5)
# --------------------------------------------------------------------------------------------------------------------

def _idempotence_geometry(fig_w: float, fig_h: float, dpi: float, spec: dict) -> dict:
    """
    Geometry of the plot_idempotence figure as matplotlib computes it (layout="constrained"): builds the same figure
    WITHOUT data (titles, labels, limits, ticks, legends and percentages above the bars of A; in B, the ECDFs, so that
    the loc="best" legend ends up at the same place) and returns the boxes (px from the top-left corner of the image
    cropped as bbox_inches="tight") of the axes, texts and legends, and the ticks that matplotlib puts on each axis.
    spec: {panel: dict(title, xlabel, ylabel, xlim, ylim, xticks, legend, texts, lines), "suptitle": str}
    - legend: dict(names, loc, ncol, title, patches); texts: [(x, y, text)]; lines: [(x, y)] (steps)
    - panels: A, B, C, D, E, E2 (below E) and F
    """
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5])
    fig.suptitle(spec["suptitle"], fontsize=16)
    sub_e = gs[2, 0].subgridspec(2, 1, height_ratios=[2, 0.6])
    axs = dict(A=fig.add_subplot(gs[0, 0]), B=fig.add_subplot(gs[0, 1]), C=fig.add_subplot(gs[1, 0]),
               D=fig.add_subplot(gs[1, 1]), E=fig.add_subplot(sub_e[0]), E2=fig.add_subplot(sub_e[1]),
               F=fig.add_subplot(gs[2, 1]))

    texts = {}
    for k, ax in axs.items():
        s = spec[k]
        for x, y in s.get("lines", []):
            ax.plot(x, y, drawstyle="steps-post")
        ax.set_xlim(*s["xlim"])
        ax.set_ylim(*s["ylim"])
        if s.get("xticks"):
            ax.set_xticks(*s["xticks"])
        if s.get("title"):
            ax.set_title(s["title"], fontsize=14, fontweight="bold")
        ax.set_xlabel(s.get("xlabel", ""), fontsize=12)
        ax.set_ylabel(s["ylabel"], fontsize=12)
        if s.get("legend"):
            lg = s["legend"]
            handles = [Patch() if lg.get("patches") else Line2D([], []) for _ in lg["names"]]
            ax.legend(handles, lg["names"], fontsize=10, loc=lg["loc"], ncol=lg.get("ncol", 1), title=lg.get("title"),
                      title_fontproperties=dict(weight="bold", size=10))
        texts[k] = [ax.text(x, y, t, ha="center", va="bottom", fontsize=10, fontweight="bold")
                    for x, y, t in s.get("texts", [])]

    canvas.draw()
    r = canvas.get_renderer()
    tight = fig.get_tightbbox(r)
    pad = 0.1*dpi                                                                # pad_inches of bbox_inches="tight"
    x0, y_top = tight.x0*dpi - pad, tight.y1*dpi + pad

    def box(artist):
        bb = artist.get_window_extent(r)
        return dict(x0=bb.x0 - x0, x1=bb.x1 - x0, top=y_top - bb.y1, bottom=y_top - bb.y0)

    def ticks(axis):
        """Visible ticks (within the limits), their labels and their boxes"""
        lo, hi = sorted(axis.get_view_interval())
        tl = [t for t in axis.get_major_ticks() if lo - 1e-9*(hi - lo) <= t.get_loc() <= hi + 1e-9*(hi - lo)]
        return [t.get_loc() for t in tl], [t.label1.get_text() for t in tl], [box(t.label1) for t in tl]

    def legend_box(leg):
        if leg is None:
            return None
        title = leg.get_title()
        return dict(frame=box(leg), texts=[box(t) for t in leg.get_texts()],
                    title=box(title) if title.get_text() else None)

    out = dict(size=(int(tight.width*dpi + 2*pad), int(tight.height*dpi + 2*pad)), suptitle=box(fig._suptitle))
    for k, ax in axs.items():
        out[k] = dict(ax=box(ax), xlabel=box(ax.xaxis.label), ylabel=box(ax.yaxis.label), title=box(ax.title),
                      xticks=ticks(ax.xaxis), yticks=ticks(ax.yaxis), legend=legend_box(ax.get_legend()),
                      texts=[box(t) for t in texts[k]])
    return out


def plot_idempotence(results_df: pd.DataFrame, width: int | None = None, dpi: float = 100) -> go.Figure:
    """
    Interactive (plotly) version of plot_idempotence (matplotlib/seaborn, section 6.4.5 of evaluation_denoise.ipynb):
    same layout (layout="constrained", figsize 14 x 12 in), sizes, fonts, colors and positions. The figure is cropped
    as the inline backend does (bbox_inches="tight"), so its size matches the matplotlib image.
    Idempotence of the algorithm in experiment 2 (equal condition and controls; each mixture is processed twice):
    - A: proportion of mixtures with clicks in the 2nd pass per group (only inside the artifact zones / also outside),
      with the total percentage above each bar
    - B: ECDF of the change introduced by the 2nd pass (NMSE between the outputs of the 1st and the 2nd pass) per
      group
    - C: proportion of mixtures with clicks in the 2nd pass vs SNR, inside (solid) and outside (dashed) the zones
      (mean ± 95 % CI between clean signals), and the control (any click) on the right
    - D: change introduced by the 2nd pass vs SNR (median ± IQR) and the control on the right
    - E: proportion of artifacts detected after the 1st pass (solid) and in the 1st or the 2nd (dashed) vs SNR, and
      below (E'), the artifacts recovered by the 2nd pass (percentage points)
    - F: extra attenuation of the artifacts in the 2nd pass (ΔSNR after the 2nd − after the 1st) vs SNR (median ± IQR)
    Interactivity: hover with the values in every panel (in A, the percentage of each bar segment; in B, the
    cumulative proportion at each NMSE); click on the legend to hide/show the bars of a category (A), the curve of a
    group (B) or all the curves of a number of artifacts of its panel (C, D, F; in E, also the one of E'); the control
    of C and D hides the point, its error bar and its reference line. The style items of the legends of C and E
    (inside / outside the zones, 1st / 1st or 2nd pass) only explain the line type and are not clickable. Zoom: C and
    D move the SNR axis together, as E, E' and F (sharex); A is fixed (categories).
    The positions of the axes, texts and legends and the ticks are taken from the constrained layout of matplotlib
    applied to the same figure without data (_idempotence_geometry).
    Limits: the plotly grid always stays below the bands and the reference lines of the control (in matplotlib,
    above); in A it is drawn as shapes so that it stays over the bars, as in matplotlib; the ECDFs with more than
    ECDF_EXACT values are evaluated on a grid of ECDF_STEP dB (the steps move at most ECDF_STEP dB, invisible without
    zoom) so that the HTML does not weigh tens of MB; the bootstrap CIs use a fixed seed (seaborn does not), so they
    can differ slightly from the matplotlib version.

    Parameters
    ----------
    results_df: pd.DataFrame
        Results of experiment 2 (read_results("experiment_2.csv")). It is not modified
    width: int or None
        Width of the figure (px). Only the width of the axes changes (the fonts and the margins do not). None -> as
        matplotlib
    dpi: float
        px per inch (matplotlib inline: 100). Every size (fonts, lines, margins) scales with it

    Returns
    -------
    fig: go.Figure
        Interactive figure
    """

    # === Style constants ===
    N_COLORS = {1: "#AFA9EC", 2: "#7F77DD", 3: "#3C3489"}                        # same colors as the previous figures
    C_CTRL, BLACK = MPL_COLORS["gray"], "#000000"
    C_IN, C_OUT = "#EF9F27", "#E24B4A"                                             # 2nd pass: in the zones / outside
    GROUP_LABELS = {0: "Control", 1: "1 artifact", 2: "2 artifacts", 3: "3 artifacts"}
    GROUP_COLORS = {0: C_CTRL, **N_COLORS}
    ECDF_EXACT = 2000                                                              # exact ECDF up to this n. of values
    ECDF_STEP = 0.05                                                               # dB: grid of the largest ECDFs


    # === Data (as the matplotlib version) ===
    idem = results_df[results_df["condition"].isin(["equal", "Control"])].copy()
    idem["n_art"] = idem["n_art"].astype(int)                                      # 0 = control
    idem["snr_db"] = pd.to_numeric(idem["snr_db"], errors="coerce")                # control -> NaN
    idem["clicks_in"] = idem["n_clicks_pass2_in_zones"].astype(int)
    idem["clicks_out"] = idem["n_clicks_pass2"].astype(int) - idem["clicks_in"]
    idem["outcome"] = np.select([idem["clicks_out"] > 0, idem["clicks_in"] > 0],
                                ["Out of zones", "In zones only"], "None")
    idem["idem_plot"] = pd.to_numeric(idem["idem_nmse_db"], errors="coerce")
    idem["has_in"] = (idem["clicks_in"] > 0).astype(float)
    idem["has_out"] = (idem["clicks_out"] > 0).astype(float)

    # A: proportion of mixtures per result of the 2nd pass and group
    groups = [0, 1, 2, 3]
    frac = (pd.crosstab(idem["n_art"], idem["outcome"], normalize="index")
            .reindex(index=groups, columns=["In zones only", "Out of zones", "None"], fill_value=0))
    n_group = idem["n_art"].value_counts()
    x_bar = np.arange(len(groups))
    tops = (frac["In zones only"] + frac["Out of zones"]).to_numpy()

    # x axis of C and D: SNR levels and a gap for the control on the right
    lv = np.sort(idem.loc[idem["n_art"] > 0, "snr_db"].dropna().unique())
    step = np.median(np.diff(lv))
    x_sep = lv[-1] + 0.75 * step
    x_ctrl = lv[-1] + 1.5 * step
    x_ticks = list(lv) + [x_ctrl]
    x_labels = [f"{v:g}" for v in lv] + [f'<span style="color:{C_CTRL}">Control</span>']
    lv_labels = [f"{v:g}" for v in lv]

    # C: mean per clean signal (then, mean ± 95 % CI between signals)
    art = idem[idem["n_art"] > 0]
    per_clean = art.groupby(["n_art", "snr_db", "clean_id"])[["has_in", "has_out"]].mean().reset_index()
    ctrl_any = (idem[idem["n_art"] == 0].assign(has_any=lambda d: (d["n_clicks_pass2"].astype(int) > 0).astype(float))
                .groupby("clean_id")["has_any"].mean())
    m_ctrl = ctrl_any.mean()
    ci_ctrl = 1.96 * ctrl_any.sem() if len(ctrl_any) > 1 else 0

    # D: control (median and IQR)
    ctrl_vals = idem.loc[idem["n_art"] == 0, "idem_plot"].dropna()
    q25, q50, q75 = ctrl_vals.quantile([0.25, 0.5, 0.75])

    # E: one row per artifact (equal condition: all with the SNR of the mixture), mean per clean signal
    det_long = art[["clean_id", "n_art", "snr_db", "detected", "detected_any"]].explode(["detected", "detected_any"])
    det_long[["detected", "detected_any"]] = det_long[["detected", "detected_any"]].astype(float)
    per_det = det_long.groupby(["n_art", "snr_db", "clean_id"])[["detected", "detected_any"]].mean().reset_index()
    per_det["d_rec"] = (per_det["detected_any"] - per_det["detected"]) * 100      # pp recovered by the 2nd pass

    # F: one row per artifact: what the 2nd pass adds to the ΔSNR of the 1st (= snr_out_pass2 - snr_out)
    dsnr_long = art[["clean_id", "n_art", "snr_db", "dsnr_db", "dsnr_db_pass2"]].explode(["dsnr_db", "dsnr_db_pass2"])
    dsnr_long[["dsnr_db", "dsnr_db_pass2"]] = dsnr_long[["dsnr_db", "dsnr_db_pass2"]].astype(float)
    dsnr_long["d_dsnr"] = dsnr_long["dsnr_db_pass2"] - dsnr_long["dsnr_db"]

    # --- Estimators (as seaborn) ---
    N_BOOT = 1000
    rng_ci = np.random.default_rng(1)

    def ecdf(v):
        """ECDF as sns.ecdfplot (steps after each value); with many values, evaluated on a grid"""
        v = np.sort(np.asarray(v, float))
        n = len(v)
        if n <= ECDF_EXACT:
            return v, np.arange(1, n + 1) / n
        grid = np.arange(np.floor(v[0] / ECDF_STEP), np.ceil(v[-1] / ECDF_STEP) + 1) * ECDF_STEP
        y = np.searchsorted(v, grid, side="right") / n
        keep = np.r_[True, np.diff(y) != 0]                                        # flat runs dropped
        return grid[keep], y[keep]

    def mean_ci(v):
        """Mean and bootstrap 95 % CI of the observations (errorbar=("ci", 95), n_boot=1000 of seaborn)"""
        v = np.asarray(v, float)
        boots = v[rng_ci.integers(0, len(v), (N_BOOT, len(v)))].mean(1)
        return (v.mean(), *np.percentile(boots, [2.5, 97.5]))

    def median_pi(v):
        """Median and 25-75 percentiles (estimator="median", errorbar=("pi", 50))"""
        v = np.asarray(v, float)
        return (np.median(v), *np.percentile(v, [25, 75]))

    def stat_table(df, col, func):
        """Estimator, lower band and upper band of col per (n_art, snr_db)"""
        gb = df.dropna(subset=[col]).groupby(["n_art", "snr_db"])[col]
        return pd.DataFrame([func(g) for _, g in gb], index=pd.MultiIndex.from_tuples(gb.groups.keys()),
                            columns=["mean", "lo", "hi"])

    t_in = stat_table(per_clean, "has_in", mean_ci)
    t_out = stat_table(per_clean, "has_out", mean_ci)
    t_D = stat_table(art, "idem_plot", median_pi)
    t_det = stat_table(per_det, "detected", mean_ci)
    t_any = stat_table(per_det, "detected_any", mean_ci)
    t_rec = stat_table(per_det, "d_rec", mean_ci)
    t_F = stat_table(dsnr_long, "d_dsnr", median_pi)
    ecdfs = {g: ecdf(idem.loc[idem["n_art"] == g, "idem_plot"].dropna()) for g in groups}
    ctrl_C = dict(mean=m_ctrl, lo=m_ctrl - ci_ctrl, hi=m_ctrl + ci_ctrl)
    ctrl_D = dict(mean=q50, lo=q25, hi=q75)


    # === Geometry (as matplotlib) ===
    PT = dpi / 72                                                                  # px per point
    pt = lambda v: v * PT
    em = pt(10)                                                                    # font size of ticks and legends (px)
    dashes = lambda lw: mpl_dashes(DASH_PATTERNS["--"], lw, PT)

    # Limits (matplotlib autoscale: data + 5 %; the bars and the ECDFs have fixed borders at 0 (and 1 in B))
    vals = lambda *tabs: [t.to_numpy().ravel() if isinstance(t, pd.DataFrame) else np.asarray(t, float).ravel()
                          for t in tabs]
    x_A = mpl_range([x_bar[0] - 0.3, x_bar[-1] + 0.3])
    y_A = [0, 1.05 * tops.max()]
    x_B = mpl_range(*[x for x, _ in ecdfs.values()])
    x_SNR = [lv[0] - 0.5 * step, x_ctrl + 0.5 * step]
    x_E = [lv[0] - 0.5 * step, lv[-1] + 0.5 * step]
    y_C = mpl_range(*vals(t_in, t_out, list(ctrl_C.values())))
    y_D = mpl_range(*vals(t_D, list(ctrl_D.values())))
    y_E = mpl_range(*vals(t_det, t_any))
    y_E2 = mpl_range(*vals(t_rec, [0]))
    y_F = mpl_range(*vals(t_F))

    A_NAMES = ["Only in the artifact zones", "Out of the zones"]
    G_NAMES = [GROUP_LABELS[g] for g in groups]
    N_NAMES = G_NAMES[1:]
    C_STYLES = ["In the artifact zones", "Out of the zones", "Control (any click)"]
    D_CTRL = "Control (median ± IQR)"
    E_STYLES = ["1st pass", "1st or 2nd pass"]
    A_TICKS = [(GROUP_LABELS[g], f"(n = {n_group.get(g, 0)})") for g in groups]
    SNR_LABEL = "SNR (dB)"
    spec = dict(
        suptitle="Idempotence",
        A=dict(title="Clicks detected in the second pass", xlabel="Group", ylabel="Proportion of mixtures", xlim=x_A,
               ylim=y_A, xticks=(x_bar, [f"{a}\n{b}" for a, b in A_TICKS]),
               legend=dict(names=A_NAMES, loc="upper left", title="Mixtures with clicks in the 2nd pass",
                           patches=True),
               texts=[(x, t, f"{t:.1%}") for x, t in zip(x_bar, tops)]),
        B=dict(title="Change introduced by the second pass", xlabel="NMSE between 1st and 2nd pass outputs (dB)",
               ylabel="Cumulative proportion of mixtures", xlim=x_B, ylim=[0, 1], lines=list(ecdfs.values()),
               legend=dict(names=G_NAMES, loc="best")),
        C=dict(title="Clicks in the second pass vs SNR", xlabel=SNR_LABEL, ylabel="Proportion of mixtures",
               xlim=x_SNR, ylim=y_C, xticks=(x_ticks, [f"{v:g}" for v in lv] + ["Control"]),
               legend=dict(names=N_NAMES + C_STYLES, loc="upper right", ncol=2)),
        D=dict(title="Change introduced by the second pass vs SNR", xlabel=SNR_LABEL,
               ylabel="NMSE between 1st and 2nd pass (dB)", xlim=x_SNR, ylim=y_D,
               xticks=(x_ticks, [f"{v:g}" for v in lv] + ["Control"]),
               legend=dict(names=N_NAMES + [D_CTRL], loc="upper right", ncol=2)),
        E=dict(title="Detection after one and two passes", ylabel="Proportion of artifacts detected", xlim=x_E,
               ylim=y_E, xticks=(lv, lv_labels), legend=dict(names=N_NAMES + E_STYLES, loc="upper right", ncol=2)),
        E2=dict(xlabel=SNR_LABEL, ylabel="Recovered (%)", xlim=x_E, ylim=y_E2, xticks=(lv, lv_labels)),
        F=dict(title="Extra attenuation of the artifacts in the second pass", xlabel=SNR_LABEL,
               ylabel="ΔSNR after 2nd − after 1st pass (dB)", xlim=x_E, ylim=y_F, xticks=(lv, lv_labels),
               legend=dict(names=N_NAMES, loc="upper right")),
    )

    fig_w, fig_h = 14, 12
    geo = _idempotence_geometry(fig_w, fig_h, dpi, spec)
    for _ in range(3 if width else 0):                                             # requested width: only axes change
        if geo["size"][0] == width:
            break
        fig_w += (width - geo["size"][0]) / dpi
        geo = _idempotence_geometry(fig_w, fig_h, dpi, spec)
    W, H = geo["size"]
    X = lambda v: v / W                                                            # px -> paper
    Y = lambda v: 1 - v / H

    def auto_ticks(ticks, axis_range):
        """
        Automatic ticks with the matplotlib step. Steps 1-2-5·10^k: nticks (they adapt to the zoom; plotly
        takes the 1-2-5 step strictly greater than range/nticks). Steps 2.5·10^k (plotly does not pick them):
        fixed step
        """
        v, labels = ticks[:2]
        step = v[1] - v[0]
        mant = step / 10**np.floor(np.log10(step))
        if np.isclose(mant, 2.5):
            dec = max(len(s.split(".")[1]) if "." in s else 0 for s in labels)
            return dict(tick0=0, dtick=step, tickformat=f".{dec}f")
        return dict(nticks=int(np.floor(abs(axis_range[1] - axis_range[0]) / step)) + 1,
                    tickformatstops=MPL_TICKFORMAT)

    axis_style = dict(showline=True, mirror=False, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none")                                       # sns.despine: no top/right axes
    x_style = dict(**axis_style, ticklabelstandoff=round(-2.2 * PT))
    AX = dict(A=1, B=2, C=3, D=4, E=5, E2=6, F=7)                                  # plotly axis number
    xa = lambda k: "x" if AX[k] == 1 else f"x{AX[k]}"
    ya = lambda k: "y" if AX[k] == 1 else f"y{AX[k]}"

    # A: fixed axes (categories); its horizontal grid and the two-line labels are drawn separately
    x_axes = dict(A=dict(range=x_A, tickvals=x_bar, showticklabels=False, showgrid=False, fixedrange=True),
                  B=dict(range=x_B, **auto_ticks(geo["B"]["xticks"], x_B)),
                  C=dict(range=x_SNR, tickvals=x_ticks, ticktext=x_labels),
                  D=dict(range=x_SNR, tickvals=x_ticks, ticktext=x_labels, matches="x3"),       # sharex with C
                  E=dict(range=x_E, tickvals=lv, ticktext=lv_labels),
                  E2=dict(range=x_E, tickvals=lv, ticktext=lv_labels, matches="x5"),            # sharex with E
                  F=dict(range=x_E, tickvals=lv, ticktext=lv_labels, matches="x5"))
    y_axes = {k: dict(range=spec[k]["ylim"], **auto_ticks(geo[k]["yticks"], spec[k]["ylim"])) for k in AX if k != "A"}
    y_axes["A"] = dict(range=y_A, tickvals=geo["A"]["yticks"][0], ticktext=geo["A"]["yticks"][1], showgrid=False,
                       fixedrange=True)
    layout = {}
    for k, n in AX.items():
        b = geo[k]["ax"]
        xkey, ykey = ("xaxis", "yaxis") if n == 1 else (f"xaxis{n}", f"yaxis{n}")
        layout[xkey] = {**x_style, "domain": [X(b["x0"]), X(b["x1"])], "anchor": ya(k), **x_axes[k]}
        layout[ykey] = {**axis_style, "domain": [Y(b["bottom"]), Y(b["top"])], "anchor": xa(k), **y_axes[k]}

    # Legends: one plotly legend per item, at the position of the matplotlib text (the text of a plotly item
    # starts 40 px to the right of the x of the legend), and a common frame with rounded corners (at the end, over
    # all the shapes). The ids start at legend2: with so many legends, plotly does not draw the default legend
    # ("legend")
    legends, shapes, frames, annotations, n_leg = {}, [], [], [], 1

    def ann(s, b, size, rotate=False, color="black"):
        """Text centered in the matplotlib box b (raised 0.1 em: Arial sits lower than DejaVu Sans)"""
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["top"] + b["bottom"]) / 2
        if rotate:
            cx -= 0.1 * pt(size)
        else:
            cy -= 0.1 * pt(size)
        return dict(text=s, x=X(cx), y=Y(cy), xref="paper", yref="paper", xanchor="center", yanchor="middle",
                    showarrow=False, font=dict(size=pt(size), color=color), textangle=-90 if rotate else 0)

    def legend_items(k, clickable):
        """Legend of panel k (frame, texts and title): name -> id of the plotly legend"""
        nonlocal n_leg
        g, ids = geo[k]["legend"], []
        for name, t in zip(spec[k]["legend"]["names"], g["texts"]):
            n_leg += 1
            lid = f"legend{n_leg}"
            click = name in clickable
            legends[lid] = dict(x=X(t["x0"] - 40), y=Y((t["top"] + t["bottom"]) / 2 - 0.1 * em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30,
                                itemclick="toggle" if click else False,
                                itemdoubleclick="toggleothers" if click else False)
            ids.append(lid)
        b = g["frame"]
        frames.append(dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                           line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(b["x0"]), X(b["x1"]), Y(b["bottom"]), Y(b["top"]), 0.2 * em, W, H)))
        if g["title"]:
            annotations.append(ann(f"<b>{spec[k]['legend']['title']}</b>", g["title"], 10))
        return dict(zip(spec[k]["legend"]["names"], ids))

    leg = dict(A=legend_items("A", A_NAMES), B=legend_items("B", G_NAMES),
               C=legend_items("C", N_NAMES + C_STYLES[2:]), D=legend_items("D", N_NAMES + [D_CTRL]),
               E=legend_items("E", N_NAMES), F=legend_items("F", N_NAMES))

    def style_entry(k, name, color, lw, dash="solid", symbol=None, mew=0.75):
        """Legend item without data (line style), not clickable"""
        return go.Scatter(x=[None], y=[None], mode="lines+markers" if symbol else "lines", name=name,
                          line=dict(color=color, width=pt(lw), dash=dash),
                          marker=dict(symbol=symbol or "circle", size=pt(6), color=color,
                                      line=dict(color="black", width=pt(mew))),
                          legend=leg[k][name], xaxis=xa(k), yaxis=ya(k), hoverinfo="skip")

    fig = go.Figure()

    # --- A: clicks in the 2nd pass per group ---
    # Legend entries (Patch): thick line as tall as the patch and its black edge as a shape (over the frame)
    for name, color, col, bottom in ((A_NAMES[0], C_IN, "In zones only", np.zeros(len(groups))),
                                     (A_NAMES[1], C_OUT, "Out of zones", frac["In zones only"].to_numpy())):
        h = frac[col].to_numpy()
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name, line=dict(color=color, width=pt(7)),
                                 legend=leg["A"][name], legendgroup=f"A_{name}", xaxis="x", yaxis="y"))
        fig.add_trace(go.Bar(x=x_bar, y=h, base=bottom, width=0.6, name=name, showlegend=False,
                             marker=dict(color=color, line=dict(color="black", width=pt(0.8))),
                             customdata=np.c_[h * 100, [n_group.get(g, 0) for g in groups]],
                             text=G_NAMES, textposition="none", legend=leg["A"][name], legendgroup=f"A_{name}",
                             xaxis="x", yaxis="y",
                             hovertemplate=(f"%{{text}} (n = %{{customdata[1]}})<br>{name}: %{{customdata[0]:.2f}} %"
                                            "<extra></extra>")))
        t = geo["A"]["legend"]["texts"][A_NAMES.index(name)]
        cy = (t["top"] + t["bottom"]) / 2 - 0.1 * em
        frames.append(dict(type="rect", xref="paper", yref="paper", x0=X(t["x0"] - 35), x1=X(t["x0"] - 5),
                           y0=Y(cy + pt(3.5)), y1=Y(cy - pt(3.5)), layer="above", line=dict(color="black", width=pt(0.8))))
    # Horizontal grid over the bars (in matplotlib, zorder 1.5 > bars), except the one at 0, below the axis
    for v in geo["A"]["yticks"][0][1:]:
        shapes.append(dict(type="line", xref="x domain", yref="y", x0=0, x1=1, y0=v, y1=v, layer="above",
                           line=dict(color="#b0b0b0", width=pt(0.8))))
    # Total percentage above each bar
    for x, top in zip(x_bar, tops):
        annotations.append(dict(text=f"<b>{top:.1%}</b>", x=x, y=top, xref="x", yref="y", xanchor="center",
                                yanchor="bottom", yshift=0.1 * em, showarrow=False, borderpad=0, borderwidth=0,
                                font=dict(size=em)))
    # x axis labels: two lines with the color of the group, at the position of each line in matplotlib (from the
    # first one, with the matplotlib line spacing: 1.2 x height of a line)
    line_h = geo["A"]["yticks"][2][0]["bottom"] - geo["A"]["yticks"][2][0]["top"]                 # height of a line
    for (l1, l2), b, g in zip(A_TICKS, geo["A"]["xticks"][2], groups):
        color = GROUP_COLORS[g]
        for i, s in enumerate((l1, l2)):
            top = b["top"] + i * 1.2 * line_h
            annotations.append(ann(s, dict(b, top=top, bottom=top + line_h), 10, color=color))

    # --- B: ECDF of the change in the 2nd pass per group ---
    for g in groups:
        x, y = ecdfs[g]
        name = GROUP_LABELS[g]
        lw_dash = dashes(2) if g == 0 else "solid"
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                                 line=dict(color=GROUP_COLORS[g], width=pt(2), dash=lw_dash, shape="hv"), name=name,
                                 legend=leg["B"][name], legendgroup=f"B_{g}", xaxis="x2", yaxis="y2",
                                 hovertemplate=(f"{name}<br>NMSE ≤ %{{x:.2f}} dB<br>Proportion: %{{y:.4f}}"
                                                "<extra></extra>")))

    # --- Curves with band (sns.lineplot) ---
    def band(k, d, color, alpha, group):
        x = d.index.to_numpy(float)
        return go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[d["hi"], d["lo"][::-1]], mode="lines", fill="toself",
                          fillcolor=rgba(color, alpha), line=dict(width=0), hoverinfo="skip", showlegend=False,
                          legendgroup=group, xaxis=xa(k), yaxis=ya(k))

    def curve(k, d, n, group, what, band_name, fmt, lw=2, dash="solid", symbol="circle", ms=6, name=None,
              legend=None):
        """Line with markers of d (index SNR; mean, lo, hi); with legend, it is also the legend item"""
        return go.Scatter(x=d.index.to_numpy(float), y=d["mean"], mode="lines+markers", name=name or N_NAMES[n - 1],
                          line=dict(color=N_COLORS[n], width=pt(lw), dash=dash),
                          marker=dict(symbol=symbol, size=pt(ms), color=N_COLORS[n],
                                      line=dict(color="black", width=pt(0.75))),
                          customdata=d[["mean", "lo", "hi"]].to_numpy(), showlegend=legend is not None,
                          legend=legend, legendgroup=group, xaxis=xa(k), yaxis=ya(k),
                          hovertemplate=(f"{N_NAMES[n - 1]}<br>SNR: %{{x:g}} dB<br>{what}: %{{customdata[0]:{fmt}}}"
                                         f"<br>{band_name}: (%{{customdata[1]:{fmt}}}, %{{customdata[2]:{fmt}}})"
                                         "<extra></extra>"))

    def control(k, ctrl_v, name, what, band_name, fmt):
        """Control on the right (errorbar, zorder 4) and its value as reference (axhline, zorder 1); separation"""
        common = dict(xaxis=xa(k), yaxis=ya(k), legend=leg[k][name], legendgroup=f"{k}_ctrl")
        lo, hi = ctrl_v["lo"], ctrl_v["hi"]
        span = 1e3 * (x_SNR[1] - x_SNR[0])                                         # very long reference: follows zoom
        ref = go.Scatter(x=[x_SNR[0] - span, x_SNR[1] + span], y=[ctrl_v["mean"]] * 2, mode="lines",
                         line=dict(color=C_CTRL, width=pt(1.2), dash=dashes(1.2)), hoverinfo="skip",
                         showlegend=False, **common)
        point = [go.Scatter(x=[x_ctrl, x_ctrl], y=[lo, hi], mode="lines", line=dict(color=C_CTRL, width=pt(1.5)),
                            hoverinfo="skip", showlegend=False, **common),
                 go.Scatter(x=[x_ctrl, x_ctrl], y=[lo, hi], mode="markers", hoverinfo="skip", showlegend=False,
                            marker=dict(symbol="line-ew", size=pt(8), line=dict(color=C_CTRL, width=pt(1))), **common),
                 go.Scatter(x=[x_ctrl], y=[ctrl_v["mean"]], mode="markers", showlegend=False,
                            marker=dict(size=pt(8), color=C_CTRL, line=dict(color="black", width=pt(1))),
                            customdata=[[ctrl_v["mean"], lo, hi]],
                            hovertemplate=(f"{name}<br>{what}: %{{customdata[0]:{fmt}}}<br>{band_name}: "
                                           f"(%{{customdata[1]:{fmt}}}, %{{customdata[2]:{fmt}}})<extra></extra>"),
                            **common),
                 go.Scatter(x=[None], y=[None], mode="lines+markers", name=name, hoverinfo="skip",
                            line=dict(color=C_CTRL, width=pt(1.5), dash=dashes(1.5)),
                            marker=dict(size=pt(6), color=C_CTRL, line=dict(color="black", width=pt(1))), **common)]
        shapes.append(dict(type="line", xref=xa(k), yref=f"{ya(k)} domain", x0=x_sep, x1=x_sep, y0=0, y1=1,
                           layer="above", line=dict(color="black", width=pt(0.8))))
        return ref, point

    # --- C: mixtures with clicks in the 2nd pass vs SNR ---
    what = "Proportion of mixtures"
    ref, point = control("C", ctrl_C, C_STYLES[2], "Mixtures with clicks (mean)", "CI 95%", ".3f")
    fig.add_traces([b for n in (1, 2, 3) for b in (band("C", t_in.loc[n], N_COLORS[n], 0.2, f"C_{n}"),
                                                   band("C", t_out.loc[n], N_COLORS[n], 0.1, f"C_{n}"))] + [ref])
    for n in (1, 2, 3):
        fig.add_trace(curve("C", t_in.loc[n], n, f"C_{n}", f"{what} with clicks in the zones", "CI 95%", ".3f"))
        fig.add_trace(curve("C", t_out.loc[n], n, f"C_{n}", f"{what} with clicks out of the zones", "CI 95%", ".3f",
                            lw=1.5, dash=dashes(1.5), symbol="x"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=N_NAMES[n - 1], legend=leg["C"][N_NAMES[n - 1]],
                                 legendgroup=f"C_{n}", line=dict(color=N_COLORS[n], width=pt(2)), xaxis="x3", yaxis="y3"))
    fig.add_traces(point)
    fig.add_trace(style_entry("C", C_STYLES[0], BLACK, 2, symbol="circle"))
    fig.add_trace(style_entry("C", C_STYLES[1], BLACK, 1.5, dash=dashes(1.5), symbol="x"))

    # --- D: change in the 2nd pass vs SNR ---
    ref, point = control("D", ctrl_D, D_CTRL, "NMSE between passes (median)", "IQR", ".1f")
    fig.add_traces([band("D", t_D.loc[n], N_COLORS[n], 0.2, f"D_{n}") for n in (1, 2, 3)] + [ref])
    fig.add_traces([curve("D", t_D.loc[n], n, f"D_{n}", "NMSE between passes (dB, median)", "IQR", ".1f",
                          legend=leg["D"][N_NAMES[n - 1]]) for n in (1, 2, 3)])
    fig.add_traces(point)

    # --- E and E': detection after one and two passes ---
    fig.add_traces([b for n in (1, 2, 3) for b in (band("E", t_det.loc[n], N_COLORS[n], 0.2, f"E_{n}"),
                                                   band("E", t_any.loc[n], N_COLORS[n], 0.1, f"E_{n}"),
                                                   band("E2", t_rec.loc[n], N_COLORS[n], 0.2, f"E_{n}"))])
    for n in (1, 2, 3):
        fig.add_trace(curve("E", t_det.loc[n], n, f"E_{n}", "Detected (1st pass)", "CI 95%", ".3f"))
        fig.add_trace(curve("E", t_any.loc[n], n, f"E_{n}", "Detected (1st or 2nd pass)", "CI 95%", ".3f", lw=1.5,
                            dash=dashes(1.5), symbol="x"))
        fig.add_trace(curve("E2", t_rec.loc[n], n, f"E_{n}", "Recovered by the 2nd pass (pp)", "CI 95%", ".2f",
                            lw=1.5, ms=4))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=N_NAMES[n - 1], legend=leg["E"][N_NAMES[n - 1]],
                                 legendgroup=f"E_{n}", line=dict(color=N_COLORS[n], width=pt(2)), xaxis="x5", yaxis="y5"))
    fig.add_trace(style_entry("E", E_STYLES[0], BLACK, 2, symbol="circle"))
    fig.add_trace(style_entry("E", E_STYLES[1], BLACK, 1.5, dash=dashes(1.5), symbol="x"))
    shapes.append(dict(type="line", xref="x6 domain", yref="y6", x0=0, x1=1, y0=0, y1=0, layer="above",
                       line=dict(color="black", width=pt(0.8))))                   # axhline(0) of E'

    # --- F: extra attenuation in the 2nd pass ---
    fig.add_traces([band("F", t_F.loc[n], N_COLORS[n], 0.2, f"F_{n}") for n in (1, 2, 3)])
    fig.add_traces([curve("F", t_F.loc[n], n, f"F_{n}", "ΔSNR 2nd − 1st pass (dB, median)", "IQR", ".4f",
                          legend=leg["F"][N_NAMES[n - 1]]) for n in (1, 2, 3)])

    # --- Titles and labels ---
    annotations.append(ann(spec["suptitle"], geo["suptitle"], 16))
    for k in AX:
        if spec[k].get("title"):
            annotations.append(ann(f"<b>{spec[k]['title']}</b>", geo[k]["title"], 14))
        if spec[k].get("xlabel"):
            annotations.append(ann(spec[k]["xlabel"], geo[k]["xlabel"], 12))
        annotations.append(ann(spec[k]["ylabel"], geo[k]["ylabel"], 12, rotate=True))

    fig.update_layout(
        **layout, **legends, width=W, height=H, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False), barmode="overlay",
        annotations=annotations, shapes=shapes + frames, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )
    return fig


# ====================================================================================================================
# 6. Entry point
# ====================================================================================================================

# Figures of each experiment: name of the files -> function (results_df -> go.Figure)
FIGURES = {
    1: {
        "detection": plot_detection,
        "elimination": plot_elimination,
        "conservation": plot_conservation,
        "amplitude": plot_amplitude,
        "factor_analysis": plot_factor_analysis,
    },
    2: {
        "experiment_comparison": plot_experiment_comparison,
        "interaction_artifacts": plot_interaction_artifacts,
        "mixture_performance": plot_mixture_performance,
        "conservation_multiple": plot_conservation_multiple,
        "idempotence": plot_idempotence,
    },
}


def save_figures(experiment: int, names: list[str] | None = None, scale: float = PNG_SCALE,
                 verbose: bool = True) -> None:
    """
    Builds the figures of an experiment from its results (RESULTS_DIR/experiment_{experiment}.csv) and saves each
    one in FIGURES_DIR/experiment_{experiment}/{name}.{png, svg, html}.

    Parameters
    ----------
    experiment: int
        Experiment: 1 (single artifact) or 2 (several artifacts + idempotence)
    names: list of str or None
        Figures to save (keys of FIGURES[experiment]). None -> all of them
    scale: float
        Scale factor of the .png images
    verbose: bool
        Prints each saved figure

    Examples
    --------
    >>> save_figures(1)                                   # Every figure of experiment 1
    >>> save_figures(2, names=["idempotence"])            # Only one figure of experiment 2
    """
    if experiment not in FIGURES:
        raise ValueError(f"Unknown experiment {experiment} (available: {list(FIGURES)})")
    figures = FIGURES[experiment]
    names = list(figures) if names is None else names
    unknown = [n for n in names if n not in figures]
    if unknown:
        raise ValueError(f"Unknown figures of experiment {experiment}: {unknown} (available: {list(figures)})")

    results_df = read_results(CSV_FILES[experiment])
    out_dir = FIGURES_DIR / f"experiment_{experiment}"
    for name in names:
        fig = figures[name](results_df)
        save_figure(fig, out_dir / name, scale=scale)
        if verbose:
            print(f"  results/figures/experiment_{experiment}/{name} (.png, .svg, .html)")
