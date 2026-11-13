"""
Summary plots of the noise removal (clicks, crackles) - SWT (db4): interactive (plotly) figures built from the
intermediate results of remove_noise (details, return_details=True). Copied from denoise_summary_plots.ipynb.
The algorithm constants that the figures need are passed in a dictionary (params, see summary_plots).
"""

# ====================================================================================================================
# 1. Imports
# ====================================================================================================================

import operator
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import scipy.ndimage as ndi
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import ImageFont
from plotly.subplots import make_subplots
from scipy.ndimage import minimum_filter1d
from scipy.signal import ShortTimeFFT, freqz, welch
from scipy.signal.windows import hann

from . import remove_noise as rn


# ====================================================================================================================
# 2. Constants
# ====================================================================================================================

# Figure size
DPI = 100  # px per inch (matplotlib inline: 100). Every size (fonts, lines, pads) scales with it
WIDTH = None  # Width (px) of the figures that reproduce matplotlib (None -> as matplotlib)
WIDTH_SUMMARY = 1450  # Width (px) of decomposition_summary_plot
LINK_X = True  # decomposition_summary_plot: zoom/pan in time linked in all the subplots
EPS = np.finfo(float).eps

# Colors of the matplotlib version
MPL_COLORS = {"gray": "#808080", "tab:green": "#2ca02c", "tab:red": "#d62728", "tab:orange": "#ff7f0e",
              "red": "#ff0000", "blue": "#0000ff", "black": "#000000", "tab:blue": "#1f77b4", "tab:cyan": "#17becf",
              "tab:purple": "#9467bd", "orange": "#ffa500", "sienna": "#a0522d", "tab:gray": "#7f7f7f",
              "brown": "#a52a2a"}
FONT = "Arial"

# Tick labels with fixed decimals depending on the tick step (as matplotlib: 0.0, 0.2, ... instead of 0, 0.2, ...)
MPL_TICKFORMAT = [dict(dtickrange=[None if k == -8 else 10**k*(1 - 1e-6), None if k == 5 else 10**(k + 1)*(1 - 1e-6)],
                       value=f".{max(0, -k)}f") for k in range(-8, 6)]
LEGEND_ANCHORS = ["NE", "NW", "SW", "SE", "E", "W", "E", "S", "N", "C"]  # loc 1-10: "upper right", "upper left"...
DASH_PATTERNS = {":": [1, 1.65], "--": [3.7, 1.6], "-.": [6.4, 1.6, 1, 1.6]}

# Click classification (plot_click_criteria)
GRAY = "rgb(180, 180, 180)"
GREEN = "rgb(0, 153, 51)"
RED = "red"
CLICK_FILL = "rgba(220, 0, 0, 0.18)"  # Candidate spans classified as click (attenuated)
KEPT_FILL = "rgba(0, 90, 200, 0.12)"  # Candidate spans that are not a click (kept)
PASS_FILL = "rgb(205, 237, 205)"  # Table cells: rule met
FAIL_FILL = "rgb(248, 248, 248)"  # Table cells: rule not met
COMPARE = {">": operator.gt, ">=": operator.ge, "<=": operator.le}  # Operators of the click rules

# Spectrograms (plot_noise_removal)
FMIN = 50  # Hz. Frequency range shown in the spectrograms (same as src/calculate_rtf.py)
FMAX = 1050
SPEC_SCALE = "linear"  # Color scale: "linear" (normalized power, 0-1) or "log" (dB relative to the maximum)
SPEC_DB_RANGE = 60  # dB. Dynamic range shown with SPEC_SCALE = "log"
# Colormap of the spectrograms (mapac2, MATLAB, 64 RGB rows in [0, 1]): white -> green -> yellow -> red -> black
MAPAC2 = np.array([[0.9451, 0.9686, 0.9490],[0.8101, 0.9160, 0.8420],[0.6751, 0.8633, 0.7350],[0.5401, 0.8106, 0.6280],[0.4050, 0.7580, 0.5210],[0.2700, 0.7053, 0.4140],[0.1350, 0.6527, 0.3070],[0.0000, 0.6000, 0.2000],[0.0417, 0.6167, 0.1917],[0.0833, 0.6333, 0.1833],[0.1250, 0.6500, 0.1750],[0.1667, 0.6667, 0.1667],[0.2083, 0.6833, 0.1583],[0.2500, 0.7000, 0.1500],[0.2917, 0.7167, 0.1417],[0.3333, 0.7333, 0.1333],[0.3750, 0.7500, 0.1250],[0.4167, 0.7667, 0.1167],[0.4583, 0.7833, 0.1083],[0.5000, 0.8000, 0.1000],[0.5417, 0.8167, 0.0917],[0.5833, 0.8333, 0.0833],[0.6250, 0.8500, 0.0750],[0.6667, 0.8667, 0.0667],[0.7083, 0.8833, 0.0583],[0.7500, 0.9000, 0.0500],[0.7917, 0.9167, 0.0417],[0.8333, 0.9333, 0.0333],[0.8750, 0.9500, 0.0250],[0.9167, 0.9667, 0.0167],[0.9583, 0.9833, 0.0083],[1.0000, 1.0000, 0.0000],[1.0000, 0.9778, 0.0111],[1.0000, 0.9556, 0.0222],[1.0000, 0.9333, 0.0333],[1.0000, 0.9111, 0.0444],[1.0000, 0.8889, 0.0556],[1.0000, 0.8667, 0.0667],[1.0000, 0.8444, 0.0778],[1.0000, 0.8222, 0.0889],[1.0000, 0.8000, 0.1000],[1.0000, 0.7778, 0.1111],[1.0000, 0.7556, 0.1222],[1.0000, 0.7333, 0.1333],[1.0000, 0.7111, 0.1444],[1.0000, 0.6889, 0.1556],[1.0000, 0.6667, 0.1667],[1.0000, 0.6444, 0.1778],[1.0000, 0.6222, 0.1889],[1.0000, 0.6000, 0.2000],[1.0000, 0.4500, 0.1500],[1.0000, 0.3000, 0.1000],[1.0000, 0.1500, 0.0500],[1.0000, 0.0000, 0.0000],[0.9000, 0.0000, 0.0000],[0.8000, 0.0000, 0.0000],[0.7000, 0.0000, 0.0000],[0.6000, 0.0000, 0.0000],[0.5000, 0.0000, 0.0000],[0.4000, 0.0000, 0.0000],[0.3000, 0.0000, 0.0000],[0.2000, 0.0000, 0.0000],[0.1000, 0.0000, 0.0000],[0.0000, 0.0000, 0.0000]])
SPEC_COLORSCALE = [[i / (len(MAPAC2) - 1), f"rgb({r*255:.0f}, {g*255:.0f}, {b*255:.0f})"] for i, (r, g, b) in enumerate(MAPAC2)]


# ====================================================================================================================
# 3. Helper functions
# ====================================================================================================================

def rgba(color: str, alpha: float) -> str:
    """Hex color (#rrggbb) -> rgba string"""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def gridspec_positions(nrows: int, ncols: int, height_ratios=None, width_ratios=None, hspace: float = 0.2,
                       wspace: float = 0.2, left: float = 0.125, right: float = 0.9, bottom: float = 0.11,
                       top: float = 0.88) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cell positions of a matplotlib GridSpec (same as GridSpec.get_grid_positions, default subplot parameters).
    Returns bottoms, tops (one per row), lefts, rights (one per column), in figure fraction.
    """
    def edges(n, ratios, space, total):
        ratios = np.ones(n) if ratios is None else np.asarray(ratios, float)
        cell = total / (n + space*(n - 1))
        sizes = cell*n*ratios/ratios.sum()
        seps = np.r_[0, np.full(n - 1, space*cell)]
        return np.cumsum(np.column_stack([seps, sizes]).ravel()).reshape(-1, 2)

    tops, bottoms = (top - edges(nrows, height_ratios, hspace, top - bottom)).T
    lefts, rights = (left + edges(ncols, width_ratios, wspace, right - left)).T
    return bottoms, tops, lefts, rights


@lru_cache
def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return None


def text_width(text: str, size: float) -> float:
    """Width (px) of a text in Arial (used to size the frame of the multi-column legend)"""
    font = _font(round(size))
    return font.getlength(text) if font else 0.55*size*len(text)


def mask_polygons(mask: np.ndarray, t: np.ndarray) -> tuple[list, list]:
    """
    Rectangles over the True runs of mask, spanning the whole height of the axes (y in [0, 1]); as
    fill_between(t, 0, 1, where=mask, transform=ax.get_xaxis_transform(), step="mid")
    """
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    x, y = [], []
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
        x += [t[s], t[e], t[e], t[s], t[s], None]
        y += [0, 0, 1, 1, 0, None]
    return x or [None], y or [None]


def mpl_nticks(axis_range: list, length_pt: float, label_factor: float, fontsize_pt: float = 10) -> int:
    """
    plotly nticks that reproduces the ticks of matplotlib (AutoLocator) for the initial range, while keeping the
    automatic ticks of plotly when zooming. matplotlib: nbins = axis length / (label_factor x font size), <= 9, and
    steps 1, 2, 2.5, 5, 10. plotly takes the smallest step in 1, 2, 5 >= range / nticks, so nticks is chosen to give
    the largest plotly step <= the matplotlib one (the same one, except 2.5 -> 2)
    """
    nbins = int(np.clip(length_pt // (label_factor*fontsize_pt), 1, 9))
    step = np.diff(MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(*axis_range)[:2])[0]
    exp = np.floor(np.log10(step))
    step_plotly = max(m*10**exp for m in (1, 2, 5) if m*10**exp <= step*(1 + 1e-9))
    return int(np.ceil((axis_range[1] - axis_range[0]) / step_plotly*(1 - 1e-9)))


def mpl_range(*arrays, margin: float = 0.05) -> list[float]:
    """Axis limits as matplotlib autoscale: data limits + 5 % margins"""
    lo = min(np.nanmin(np.where(np.isfinite(a), a, np.nan)) for a in arrays)
    hi = max(np.nanmax(np.where(np.isfinite(a), a, np.nan)) for a in arrays)
    span = (hi - lo) or 1
    return [lo - margin*span, hi + margin*span]


@lru_cache
def _renderer(dpi: float) -> RendererAgg:
    return RendererAgg(1, 1, dpi)


def mpl_text_size(text: str, size_pt: float, dpi: float = 100, weight: str = "normal",
                  ismath: bool = False) -> tuple[float, float, float]:
    """
    Size (px) of a text as matplotlib (Agg, default font DejaVu Sans) lays it out: width, ascent and descent (the
    height of a line is the one of "lp", as in matplotlib). Used to place the elements and crop the figure as the
    matplotlib version (bbox_inches="tight")
    """
    prop = FontProperties(size=size_pt, weight=weight)
    w = _renderer(dpi).get_text_width_height_descent(text, prop, ismath=ismath)[0]
    _, h, d = _renderer(dpi).get_text_width_height_descent("lp", prop, ismath=False)
    return w, h - d, d


@lru_cache
def mpl_text_extent(text: str, size_pt: float, dpi: float = 100, weight: str = "normal",
                    va: str = "baseline") -> tuple[float, float, float]:
    """
    Extent (px) of a text as matplotlib lays it out (Agg, DejaVu Sans): width and height above and below the
    anchor point for the vertical alignment va (titles: "baseline", x tick labels and x labels: "top", y tick
    labels: "center_baseline"). Exact (the heights are not the integer glyph metrics of the renderer)
    """
    fig = Figure(dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    b = fig.text(0, 0, text, fontsize=size_pt, fontweight=weight, va=va).get_window_extent(canvas.get_renderer())
    return b.width, b.y1, -b.y0


def rounded_rect(x0: float, x1: float, y0: float, y1: float, r: float, width: float, height: float) -> str:
    """SVG path (paper coordinates) of a rectangle with rounded corners (radius r px), as the frame of a matplotlib legend"""
    rx, ry = r/width, r/height
    return (f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
            f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z")


def mpl_ticks(axis_range: list, length_pt: float, label_factor: float, fontsize_pt: float = 10) -> np.ndarray:
    """Ticks of matplotlib (AutoLocator) inside axis_range (see mpl_nticks)"""
    nbins = int(np.clip(length_pt // (label_factor*fontsize_pt), 1, 9))
    ticks = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(*axis_range)
    eps = 1e-9*(axis_range[1] - axis_range[0])
    return ticks[(ticks >= axis_range[0] - eps) & (ticks <= axis_range[1] + eps)]


def mpl_tick_labels(ticks: np.ndarray) -> list[str]:
    """Labels of the ticks as ScalarFormatter (same decimals in all, unicode minus)"""
    step = np.diff(ticks).min() if len(ticks) > 1 else 1
    dec = 0
    while dec < 10 and abs(round(step, dec) - step) > 1e-9*step:
        dec += 1
    return [f"{0 if abs(v) < 1e-12 else v:.{dec}f}".replace("-", "−") for v in ticks]


def grid_contour(mask: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> tuple[list, list]:
    """
    Contour of a binary mask on a grid of rectangular cells (mask[j, i]: cell between x_edges[i:i+2] and
    y_edges[j:j+2], y_edges increasing): boundary between True and False cells, joined into polylines (so the
    dashes are continuous). As plt.contour, the borders of the grid are not drawn (the lines end there).
    Returns x, y with None between polylines
    """
    m = mask.astype(bool)
    adj = defaultdict(list)
    jj, ii = np.nonzero(m[:, 1:] != m[:, :-1])  # Vertical segments: between columns i and i+1
    for j, i in zip(jj.tolist(), ii.tolist()):
        a, b = (i + 1, j), (i + 1, j + 1)
        adj[a].append(b)
        adj[b].append(a)
    jj, ii = np.nonzero(m[1:, :] != m[:-1, :])  # Horizontal segments: between rows j and j+1
    for j, i in zip(jj.tolist(), ii.tolist()):
        a, b = (i, j + 1), (i + 1, j + 1)
        adj[a].append(b)
        adj[b].append(a)

    def walk(node):
        path = [node]
        while adj[node]:
            nxt = adj[node].pop()
            adj[nxt].remove(node)
            path.append(nxt)
            node = nxt
        return path

    # Open polylines (ending at the borders) first, then the closed ones
    paths = [walk(n) for n in [n for n in adj if len(adj[n]) % 2] if adj[n]]
    for n in list(adj):
        while adj[n]:
            paths.append(walk(n))

    x, y = [], []
    for p in paths:
        p = np.array(p)
        if len(p) > 2:  # Only the corners
            keep = np.r_[True, np.any(p[2:] - p[1:-1] != p[1:-1] - p[:-2], axis=1), True]
            p = p[keep]
        x += x_edges[p[:, 0]].tolist() + [None]
        y += y_edges[p[:, 1]].tolist() + [None]
    return x or [None], y or [None]


def grid_rectangles(mask: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> tuple[list, list]:
    """Rectangles (one per run of True cells in each row) of a binary mask on a grid (see grid_contour)"""
    x, y = [], []
    for j, row in enumerate(mask.astype(bool)):
        d = np.diff(np.r_[0, row.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            x += [x_edges[s], x_edges[e], x_edges[e], x_edges[s], x_edges[s], None]
            y += [y_edges[j], y_edges[j], y_edges[j + 1], y_edges[j + 1], y_edges[j], None]
    return x or [None], y or [None]


@lru_cache
def mpl_legend_layout(labels: tuple, ncol: int = 1, fontsize_pt: float = 10, dpi: float = 100) -> tuple:
    """
    Size (px) of the frame of a matplotlib legend with these labels (mathtext allowed) and the box of each text
    (x0, x1, top, bottom; px from the top-left of the frame). Measured on a hidden matplotlib figure: the size of
    the mathtext labels cannot be computed from the font metrics alone
    """
    fig = Figure(figsize=(12, 6), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    leg = ax.legend(handles=[Line2D([], []) for _ in labels], labels=list(labels), ncol=ncol, loc="center",
                    fontsize=fontsize_pt)
    canvas.draw()
    renderer = canvas.get_renderer()
    f = leg.get_window_extent(renderer)
    texts = [t.get_window_extent(renderer) for t in leg.get_texts()]
    return f.width, f.height, [(b.x0 - f.x0, b.x1 - f.x0, f.y1 - b.y1, f.y1 - b.y0) for b in texts]


def _path_hits_box(x: np.ndarray, y: np.ndarray, box: tuple) -> bool:
    """True if some segment of the polyline (x, y) (NaN -> gap) touches the rectangle box = (x0, y0, x1, y1)"""
    x0, y0, x1, y1 = box
    p = np.column_stack([x[:-1], y[:-1]])
    d = np.column_stack([x[1:], y[1:]]) - p
    ok = np.isfinite(p).all(axis=1) & np.isfinite(d).all(axis=1)
    p, d = p[ok], d[ok]
    t0, t1 = np.zeros(len(p)), np.ones(len(p))
    for k, lo, hi in ((0, x0, x1), (1, y0, y1)):  # Liang-Barsky clipping
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo - p[:, k]) / d[:, k], (hi - p[:, k]) / d[:, k]
        inside = (p[:, k] >= lo) & (p[:, k] <= hi)
        parallel = d[:, k] == 0
        t0 = np.maximum(t0, np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(ta, tb)))
        t1 = np.minimum(t1, np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(ta, tb)))
    return bool(np.any(t0 <= t1))


def mpl_best_legend(box: tuple, size: tuple, paths: list, em: float, rects: list = ()) -> tuple[float, float]:
    """
    Position (left, top; px) of a matplotlib legend with loc="best" (as Legend._find_best_position): of the 9
    locations inside the axes (borderaxespad = 0.5 font sizes), the first one with the lowest badness = vertices
    of the lines inside the legend + lines (and polygons) that touch it + rectangles (axvspan, bar) that overlap it.
    box: axes (left, top, right, bottom), px; size: legend (width, height), px; paths: list of (x, y) arrays, px;
    rects: list of (left, top, right, bottom), px
    """
    l, t, r, b = box[0] + 0.5*em, box[1] + 0.5*em, box[2] - 0.5*em, box[3] - 0.5*em
    w, h = size
    best = None
    for anchor in LEGEND_ANCHORS:
        x = l if "W" in anchor else r - w if "E" in anchor else (l + r - w)/2
        y = t if "N" in anchor else b - h if "S" in anchor else (t + b - h)/2
        badness = 0
        for px, py in paths:
            with np.errstate(invalid="ignore"):
                inside = int(((px > x) & (px < x + w) & (py > y) & (py < y + h)).sum())
            badness += inside + (inside > 0 or _path_hits_box(px, py, (x, y, x + w, y + h)))
        badness += sum(not (r1 <= x or r0 >= x + w or b1 <= y or t1 >= y + h) for r0, t1, r1, b1 in rects)
        if best is None or badness < best[0]:
            best = (badness, x, y)
        if badness == 0:
            break
    return best[1], best[2]


def mpl_colorscale(name: str, n: int = 256) -> list:
    """plotly colorscale sampled from a matplotlib colormap"""
    cmap = colormaps[name]
    return [[v, "rgb({:.0f}, {:.0f}, {:.0f})".format(*(255*np.array(cmap(v)[:3])))] for v in np.linspace(0, 1, n)]


def mpl_dashes(pattern: list, lw_pt: float, px_per_pt: float) -> str:
    """plotly dash of a matplotlib line style (pattern in units of the line width: ":" [1, 1.65], "--" [3.7, 1.6],
    "-." [6.4, 1.6, 1, 1.6])"""
    return ",".join(f"{v*lw_pt*px_per_pt:.2f}px" for v in pattern)


def format_condition(metric: str, value: float, op: str, threshold: float) -> str:
    """
    Text of a condition of the click rules: met (✓) or not (✗), value and threshold. Widths in ms (unit not
    written, to keep the table narrow).
    """
    met = COMPARE[op](value, threshold)
    scale, decimals = (1e3, 1) if metric.startswith("width") else (1, 2)
    value_txt = f"{value}" if isinstance(value, (int, np.integer)) else f"{value*scale:.{decimals}f}"
    symbol = {">": ">", ">=": "≥", "<=": "≤"}[op]

    return f"{'✓' if met else '✗'} {metric} = {value_txt} ({symbol} {threshold*scale:g})"

# ====================================================================================================================
# 4. Main functions
# ====================================================================================================================

def decomposition_summary_plot(signal: np.ndarray, fs: float, details: dict, params: dict) -> go.Figure:
    """
    Interactive (plotly) version of summary_plot: same layout (gridspec 15 x 3, figsize 18 x 26 in), same
    legends (one per subplot group, at the same place). The figure is cropped as the inline backend does
    (bbox_inches="tight"), so its size matches the matplotlib image.
    - Every subplot has two y axes: a hidden one (range [0, 1]) with the shaded masks (as the axes transform
      of fill_between, so they always span the whole height and stay below the lines) and, overlaid, the one
      with the data
    - Clicking a legend item toggles that series in all the rows of its column
    - WIDTH_SUMMARY: width of the figure (px). Only the axes area gets narrower (the space for the labels on the
      left, the fonts and the heights do not change). None -> as matplotlib (18 in, 1568 px with DPI = 100)
    - LINK_X: zoom/pan in time linked in all the subplots
    - DPI: px per inch (matplotlib inline: 100). Every size (fonts, lines, pads) scales with it
    - params: algorithm constants (G_FAC; see summary_plots)
    """
    width, dpi, link_x = WIDTH_SUMMARY, DPI, LINK_X
    G_FAC = params["G_FAC"]
    t = np.arange(0, len(signal))/fs
    PT = dpi/72  # px per point
    W, H = 18*dpi, 26*dpi  # figsize (18, 26)
    n_rows = 15
    bottoms, tops, lefts, rights = gridspec_positions(n_rows, 3, height_ratios=[1, 0.3] + [1]*10 + [0, 1, 1],
                                                      hspace=0.5, wspace=0.2)

    # Visible region (bbox_inches="tight", pad 0.1 in): from the supylabel to the right border of the axes,
    # from the supxlabel to the title of the original signal
    pad = 0.1*dpi
    X0, X1 = 0.04 - pad/W, rights[-1] + pad/W
    Y0, Y1 = 0.09 - pad/H, tops[0] + (6*PT + 0.75*16*PT + pad)/H  # Title: pad 6 pt + cap height
    # Horizontal: the margins (labels on the left, pad on the right) keep their size in px, the axes area is
    # scaled by s to fit the width
    left, right = lefts[0], rights[-1]
    margin_l, margin_r = (left - X0)*W, (X1 - right)*W
    width = round(W*(X1 - X0)) if width is None else width
    height = round(H*(Y1 - Y0))
    s = (width - margin_l - margin_r)/((right - left)*W)
    px = lambda f: ((f - X0)*W if f < left else margin_l + (f - left)*W*s)/width  # Figure fraction -> paper
    py = lambda f: (f - Y0)/(Y1 - Y0)

    lw = lambda pt: pt*PT  # Line width: points -> px
    fs_tick, fs_legend = 10*PT, 10*PT
    axis_style = dict(showline=True, mirror=True, linecolor="black", linewidth=0.8*PT, ticks="outside",
                      ticklen=3.5*PT, tickwidth=0.8*PT, tickcolor="black", tickfont=dict(size=fs_tick),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=0.8*PT, zeroline=False, automargin=False,
                      exponentformat="none")
    x_range = mpl_range(t)

    fig = go.Figure()
    layout, annotations, axes = {}, [], []
    key = lambda letter, n: f"{letter}axis{'' if n == 1 else n}"
    ref = lambda letter, n: f"{letter}{'' if n == 1 else n}"

    def add_axes(row: int, col, ylabel: str = None) -> dict:
        """Axes in cell (row, col) of the gridspec (col: int or slice). Returns its axis references"""
        c0, c1 = (col.start or 0, (col.stop or 3) - 1) if isinstance(col, slice) else (col, col)
        n = len(axes) + 1
        x, y_fill, y_data = ref("x", n), ref("y", 2*n - 1), ref("y", 2*n)
        box = (lefts[c0], rights[c1], bottoms[row], tops[row])
        # Length of the axes in points (for the number of ticks, see mpl_nticks)
        len_x, len_y = (box[1] - box[0])*18*72*s, (box[3] - box[2])*26*72
        layout[key("x", n)] = dict(domain=[px(box[0]), px(box[1])], anchor=y_fill, range=x_range, **axis_style,
                                   nticks=mpl_nticks(x_range, len_x, 3), ticklabelstandoff=round(-2.2*PT),
                                   **({"matches": "x"} if link_x and n > 1 else {}))
        layout[key("y", 2*n - 1)] = dict(domain=[py(box[2]), py(box[3])], anchor=x, range=[0, 1],
                                         fixedrange=True, visible=False)
        layout[key("y", 2*n)] = dict(overlaying=y_fill, anchor=x, side="left", **axis_style, _len_y=len_y,
                                     **({"title": dict(text=ylabel, font=dict(size=10*PT), standoff=4*PT)}
                                        if ylabel else {}))
        # Empty trace on the hidden axis (plotly drops axes without traces, and the overlaid one with it)
        fig.add_trace(go.Scatter(x=[None], y=[None], xaxis=x, yaxis=y_fill, showlegend=False, hoverinfo="skip"))
        ax = dict(x=x, y_fill=y_fill, y=y_data, n=n, box=box)
        axes.append(ax)
        return ax

    def set_ylim(ax: dict, ylim: list, matches: dict = None):
        y_axis = layout[key("y", 2*ax["n"])]
        y_axis.update(range=ylim, nticks=mpl_nticks(ylim, y_axis.pop("_len_y"), 2),
                      **({"matches": matches["y"]} if matches else {}))

    def line(ax: dict, y: np.ndarray, name: str, color: str, width_pt: float, legend: str = None,
             group: str = None, show: bool = False, opacity: float = 1):
        fig.add_trace(go.Scatter(
            x0=t[0], dx=1/fs, y=y, mode="lines", name=name, line=dict(color=color, width=lw(width_pt)),
            opacity=opacity, xaxis=ax["x"], yaxis=ax["y"], legend=legend or "legend", legendgroup=group,
            showlegend=show
        ))

    def shade(ax: dict, mask: np.ndarray, name: str, color: str, legend: str = None, group: str = None,
              show: bool = False):
        x, y = mask_polygons(mask, t)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", fill="toself", fillcolor=rgba(color, 0.3), line=dict(width=0),
            name=name, hoverinfo="skip", xaxis=ax["x"], yaxis=ax["y_fill"], legend=legend or "legend",
            legendgroup=group, showlegend=show
        ))

    def title(ax: dict, text: str, size_pt: float, pad_pt: float = 6):
        l, r, _, top = ax["box"]
        annotations.append(dict(text=f"<b>{text}</b>", x=px((l + r)/2), y=py(top) + pad_pt*PT/height,
                                xref="paper", yref="paper", xanchor="center", yanchor="bottom",
                                showarrow=False, font=dict(size=size_pt*PT)))

    legend_style = dict(bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#cccccc", borderwidth=1,
                        font=dict(size=fs_legend), groupclick="togglegroup", tracegroupgap=0)

    def legend_center(name: str, ax: dict, y_axes: float):
        """Horizontal legend centered at (0.5, y_axes) in axes coordinates (loc="center", bbox_to_anchor)"""
        l, r, b, top = ax["box"]
        layout[name] = dict(orientation="h", x=px((l + r)/2), y=py(b + y_axes*(top - b)), xanchor="center",
                            yanchor="middle", **legend_style)

    def legend_upper_right(name: str, ax: dict):
        """Legend inside the axes, upper right corner (borderaxespad = 0.5 font size)"""
        _, r, _, top = ax["box"]
        layout[name] = dict(x=px(r) - 0.5*fs_legend/width, y=py(top) - 0.5*fs_legend/height, xanchor="right",
                            yanchor="top", **legend_style)

    # --- Original ---
    ax_original = add_axes(0, slice(None))
    line(ax_original, details["original"], "Original", MPL_COLORS["gray"], 1, show=True)
    set_ylim(ax_original, mpl_range(details["original"]))
    legend_upper_right("legend", ax_original)

    # --- Details ---
    swt_cols = [["Original", "Filtered", "Threshold"], ["Envelope", "Clicks"], ["Non-Clicks", "Time-Protected"]]
    swt_legend = {name: f"legend{2 + k}" for k, names in enumerate(swt_cols) for name in names}
    axs = np.empty((10, 3), dtype=object)

    for i in range(10):
        band = details["bands"][i]
        subband = "a" if band["part"] == "upper" else "b" if band["part"] == "lower" else ""
        axs[i, 0] = add_axes(i + 2, 0, ylabel=f"D{band['level']}{subband}:<br>{band['f_low']} - {band['f_high']} Hz")
        axs[i, 1] = add_axes(i + 2, 1)
        axs[i, 2] = add_axes(i + 2, 2)
        show = i == 0  # The legends show the items of the first row (as in matplotlib)

        # Detail, filtered, threshold and envelope
        swt = lambda name: dict(legend=swt_legend[name], group=f"swt-{name}", show=show)
        line(axs[i, 0], details["detail"][i, :], "Original", MPL_COLORS["gray"], 1, **swt("Original"))
        line(axs[i, 0], details["detail_filtered"][i, :], "Filtered", MPL_COLORS["tab:green"], 1, **swt("Filtered"))
        line(axs[i, 0], details["threshold"][i, :], "Threshold", MPL_COLORS["red"], 1.5, **swt("Threshold"))
        line(axs[i, 0], details["envelope"][i, :], "Envelope", MPL_COLORS["blue"], 1.5, **swt("Envelope"))

        mask_clicks = details["mask_clicks"][i, :].astype(bool)
        mask_protected = details["mask_protected"][i, :].astype(bool) & ~mask_clicks
        mask_reference = details["mask_reference"][i, :].astype(bool) & ~details["mask_protected"][i, :].astype(bool)

        shade(axs[i, 0], mask_clicks, "Clicks", MPL_COLORS["tab:red"], **swt("Clicks"))
        shade(axs[i, 0], mask_protected, "Non-Clicks", MPL_COLORS["tab:green"], **swt("Non-Clicks"))
        shade(axs[i, 0], mask_reference, "Time-Protected", MPL_COLORS["tab:orange"], **swt("Time-Protected"))

        # Gain
        unmasked_gain = np.minimum(1, G_FAC*details["reference"][i, :] / details["envelope"][i, :])
        gain = lambda name: dict(legend="legend5", group=f"gain-{name}", show=show)
        line(axs[i, 1], unmasked_gain, "Unmasked gain", "#808080", 1.5, opacity=0.4, **gain("Unmasked gain"))
        line(axs[i, 1], details["gain"][i, :], "Gain", MPL_COLORS["black"], 2, **gain("Gain"))
        shade(axs[i, 1], mask_clicks, "Clicks", MPL_COLORS["tab:red"], **gain("Clicks"))
        set_ylim(axs[i, 1], mpl_range(unmasked_gain, details["gain"][i, :]))

        # Removed component (sharey with the decomposition)
        residual = details["detail"][i, :] - details["detail_filtered"][i, :]
        removed = lambda name: dict(legend="legend6", group=f"removed-{name}", show=show)
        line(axs[i, 2], residual, "Residual", MPL_COLORS["tab:red"], 1, **removed("Residual"))
        shade(axs[i, 2], mask_clicks, "Clicks", MPL_COLORS["tab:red"], **removed("Clicks"))

        ylim = mpl_range(details["detail"][i, :], details["detail_filtered"][i, :], details["threshold"][i, :],
                         details["envelope"][i, :], residual)
        set_ylim(axs[i, 0], ylim)
        set_ylim(axs[i, 2], ylim, matches=axs[i, 0])

    # --- Clean signal ---
    ax_clean = add_axes(13, slice(None))
    line(ax_clean, details["original"], "Original", MPL_COLORS["gray"], 1, legend="legend7", show=True)
    line(ax_clean, details["clean"], "Filtered", MPL_COLORS["tab:green"], 1, legend="legend7", show=True)

    mask_clicks = details["click_span"].astype(bool)
    mask_preselected = details["click_span_preselected"].astype(bool) & ~mask_clicks

    shade(ax_clean, mask_clicks, "Clicks", MPL_COLORS["tab:red"], legend="legend7", show=True)
    shade(ax_clean, mask_preselected, "Non-Clicks", MPL_COLORS["tab:green"], legend="legend7", show=True)

    # --- Residual (sharey with the clean signal) ---
    ax_residual = add_axes(14, slice(None))
    removed_total = details["original"] - details["clean"]
    line(ax_residual, removed_total, "Residual", MPL_COLORS["tab:red"], 1, legend="legend8", show=True)
    shade(ax_residual, mask_clicks, "Clicks", MPL_COLORS["tab:red"], legend="legend8", show=True)

    ylim = mpl_range(details["original"], details["clean"], removed_total)
    set_ylim(ax_clean, ylim)
    set_ylim(ax_residual, ylim, matches=ax_clean)

    # --- Formatting ---
    title(ax_original, "Original signal", 16)
    title(ax_clean, "Filtered signal", 14, pad_pt=30)
    title(ax_residual, "Removed component", 14)
    title(axs[0, 0], "SWT decomposition", 14, pad_pt=60)
    title(axs[0, 1], "Gain", 14, pad_pt=30)
    title(axs[0, 2], "Removed component", 14, pad_pt=30)

    # Legends: same places as in matplotlib, one per subplot group
    # SWT legend (ncol=3): one vertical legend per column (legend2-4) inside a common frame
    # Size (px) of a vertical plotly legend without border: symbol (40) + text + 4 wide, item height
    # (max(1.3 x font size, 16) + 3) per item + 10.8 high (measured)
    col_gap = fs_legend  # Extra space between columns (columnspacing = 2 font sizes, part of it is already in the item)
    col_w = [44 + max(text_width(n, fs_legend) for n in names) for names in swt_cols]
    frame_w = sum(col_w) + col_gap*(len(col_w) - 1)
    frame_h = (max(1.3*fs_legend, 16) + 3)*max(len(names) for names in swt_cols) + 10.8
    l, r, b, top = axs[0, 0]["box"]
    cx, cy = px((l + r)/2), py(b + 1.4*(top - b))
    fx0, fy1 = cx - frame_w/2/width, cy + frame_h/2/height
    for k, w in enumerate(col_w):
        layout[f"legend{2 + k}"] = dict(x=fx0 + (sum(col_w[:k]) + k*col_gap)/width, y=fy1, xanchor="left",
                                        yanchor="top",
                                        **{**legend_style, "bgcolor": "rgba(0, 0, 0, 0)", "borderwidth": 0})
    frame = dict(type="rect", xref="paper", yref="paper", x0=fx0, x1=cx + frame_w/2/width,
                 y0=cy - frame_h/2/height, y1=fy1, line=dict(color="#cccccc", width=1),
                 fillcolor="rgba(255, 255, 255, 0.8)", layer="above")

    legend_center("legend5", axs[0, 1], 1.2)
    legend_center("legend6", axs[0, 2], 1.2)
    legend_center("legend7", ax_clean, 1.2)
    legend_upper_right("legend8", ax_residual)

    # supxlabel, supylabel
    annotations.append(dict(text="Time (s)", x=px(0.5), y=py(0.09), xref="paper", yref="paper",
                            xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=14*PT)))
    annotations.append(dict(text="Amplitude (A.U.)", x=px(0.04), y=py(0.5), xref="paper", yref="paper",
                            xanchor="left", yanchor="middle", textangle=-90, showarrow=False,
                            font=dict(size=14*PT)))

    fig.update_layout(
        **layout, width=width, height=height, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=[frame], font=dict(family=FONT, size=fs_tick, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )

    return fig


def plot_detection_map(signal: np.ndarray, fs: float, details: dict, params: dict) -> go.Figure:
    """
    Interactive (plotly) version of plot_detection_map: same layout (2 x 1 subplots, figsize 14 x 8 in), sizes,
    fonts, colors and positions. The figure is cropped as the inline backend does (bbox_inches="tight"), so its
    size matches the matplotlib image.
    - Top: original and filtered signal, with the click spans shaded (hidden y axis [0, 1], as in summary_plot)
    - Bottom: envelope / reference (dB) per band (log frequency axis), with the contours of the band masks and of
      the detection threshold. Hovering shows the time, frequency and envelope / reference
    - Legend: one plotly legend per item, at the place of the matplotlib item, inside a common frame. Clicking an
      item toggles that category in both subplots. The time axes are linked
    - Colorbar: drawn as a small axes (a plotly colorbar is drawn above the shapes and hides the threshold line)
    - WIDTH: width of the figure (px). Only the axes area gets narrower/wider (the labels, the fonts and the
      heights do not change). None -> as matplotlib
    - DPI: px per inch (matplotlib inline: 100). Every size (fonts, lines, pads) scales with it
    - params: algorithm constants (K; see summary_plots)
    The vertical grid of the map is drawn at the initial ticks (plotly draws its grid below the heatmap): it does
    not follow the ticks when zooming.
    """
    width, dpi = WIDTH, DPI
    K = params["K"]
    R_db = 20*np.log10(details["envelope"] / details["reference"])  # bands x time
    n = R_db.shape[1]
    t = np.arange(0, len(signal))/fs
    bands = details["bands"]
    f_edges = np.array([b["f_low"] for b in bands[::-1]] + [bands[0]["f_high"]], float)  # Increasing
    t_edges = np.arange(n + 1) / fs
    level = 20*np.log10(K)

    PT = dpi/72  # px per point
    pt = lambda v: v*PT  # points -> px
    em = pt(10)  # Font size of the ticks and the legend (px)
    W, H = 14*dpi, 8*dpi  # figsize (14, 8)
    bottoms, tops, lefts, rights = gridspec_positions(2, 1, hspace=0.2)  # plt.subplots(2, 1)
    ax_h = (tops[0] - bottoms[0])*H

    # Text sizes as matplotlib (px): tick labels (10 pt), axis labels (12 pt), colorbar label (10 pt), title (14 pt)
    text = lambda s, size, **kw: mpl_text_size(s, size, dpi, **kw)  # width, ascent, descent
    line_h = lambda size: sum(text("lp", size)[1:])  # Height of a line
    x_range = [t_edges[0], t_edges[-1]]  # sticky edges of pcolormesh (no margins)
    y_range = mpl_range(details["original"], details["clean"])
    f_range = [f_edges[0], f_edges[-1]]
    w_ytick = max(text(s, 10)[0] for s in mpl_tick_labels(mpl_ticks(y_range, ax_h/PT, 2)))
    decades = [k for k in range(-1, 7) if f_range[0] <= 10**k <= f_range[1]]
    w_ftick = max([text(f"$10^{{{k}}}$", 10, ismath=True)[0] for k in decades] or [0])
    cb_ticks = np.arange(0, 31, 5)
    w_cbtick = max(text(str(v), 10)[0] for v in cb_ticks)
    tick_space = pt(3.5) + pt(3.5)  # Tick length + pad
    pad = 0.1*dpi  # bbox_inches="tight" pad

    # Crop (px): left -> y labels, right -> colorbar label, top -> title, bottom -> x label
    margin_l = pad + line_h(12) + pt(4) + max(w_ytick, w_ftick) + tick_space
    right_extra = tick_space + w_cbtick + pt(4) + line_h(10) + pad  # From the right edge of the colorbar
    ax_w = (W*(rights[0] - lefts[0]) if width is None
            else (width - margin_l - right_extra) / 1.025)  # Colorbar: inset_axes [1.01, 0, 0.015, 1]
    width = round(margin_l + 1.025*ax_w + right_extra)
    title_asc, title_desc = text("Detection map", 14, weight="bold")[1:]
    margin_t = pad + title_asc + pt(30)
    margin_b = tick_space + line_h(10) + pt(4) + line_h(12) + pad
    y_top = lambda f: margin_t + (tops[0] - f)*H  # Figure fraction (from the bottom) -> px from the top
    height = round(y_top(bottoms[1]) + margin_b)
    X = lambda v: v/width  # px -> paper
    Y = lambda v: 1 - v/height
    ax_l, ax_r = margin_l, margin_l + ax_w
    ax0_t, ax0_b, ax1_t, ax1_b = y_top(tops[0]), y_top(bottoms[0]), y_top(tops[1]), y_top(bottoms[1])
    cb_l, cb_r = ax_r + 0.01*ax_w, ax_r + 0.025*ax_w

    axis_style = dict(showline=True, mirror=True, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none")
    x_style = dict(**axis_style, range=x_range, nticks=mpl_nticks(x_range, ax_w/PT, 3),
                   tickformatstops=MPL_TICKFORMAT, ticklabelstandoff=round(-2.2*PT))
    dash = lambda lw: f"{pt(3.7*lw):.1f}px,{pt(1.6*lw):.1f}px"  # Dashed line ("--") of width lw (pt)
    layout = dict(
        xaxis=dict(domain=[X(ax_l), X(ax_r)], anchor="y", **x_style),
        yaxis=dict(domain=[Y(ax0_b), Y(ax0_t)], anchor="x", range=[0, 1], fixedrange=True, visible=False),
        yaxis2=dict(overlaying="y", anchor="x", side="left", **axis_style, range=y_range,
                    nticks=mpl_nticks(y_range, ax_h/PT, 2), tickformatstops=MPL_TICKFORMAT),
        xaxis2=dict(domain=[X(ax_l), X(ax_r)], anchor="y3", matches="x", **{**x_style, "showgrid": False}),
        yaxis3=dict(domain=[Y(ax1_b), Y(ax1_t)], anchor="x2", type="log", range=np.log10(f_range).tolist(),
                    **{**axis_style, "showgrid": False}, tickvals=[10**k for k in decades],
                    ticktext=[f"10<sup>{k}</sup>" for k in decades],
                    minor=dict(tickvals=[m*10**k for k in range(-1, 7) for m in range(2, 10)], ticks="outside",
                               ticklen=pt(2), tickwidth=pt(0.6), tickcolor="black", showgrid=False)),
        # Colorbar
        xaxis3=dict(domain=[X(cb_l), X(cb_r)], anchor="y4", range=[0, 1], fixedrange=True, showline=True,
                    mirror=True, linecolor="black", linewidth=pt(0.8), showticklabels=False, ticks="",
                    showgrid=False, zeroline=False),
        yaxis4=dict(domain=[Y(ax1_b), Y(ax1_t)], anchor="x3", side="right", range=[0, 30], fixedrange=True,
                    tickvals=cb_ticks, **{**axis_style, "showgrid": False}),
    )

    # Legend (ncol=6, loc="center", bbox_to_anchor=(0.5, 1.06)): items and their legend
    threshold_name = f"Threshold K = {K} ({level:.0f} dB)"
    items = ["Original", "Filtered", "Clicks", "Non-Clicks", "Time-Protected", threshold_name]
    legend_of = {name: "legend" if k == 0 else f"legend{k + 1}" for k, name in enumerate(items)}
    legend_of["Threshold"] = legend_of[threshold_name]

    fig = go.Figure()
    # Empty trace on the hidden axis (plotly drops axes without traces, and the overlaid one with it)
    fig.add_trace(go.Scatter(x=[None], y=[None], xaxis="x", yaxis="y", showlegend=False, hoverinfo="skip"))

    # --- Signal ---
    categories = {"Clicks": "tab:red", "Non-Clicks": "tab:green", "Time-Protected": "tab:orange"}
    for name, y, color in [("Original", details["original"], "gray"), ("Filtered", details["clean"], "tab:green")]:
        fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=y, mode="lines", name=name, xaxis="x", yaxis="y2",
                                 line=dict(color=MPL_COLORS[color], width=pt(1)), legend=legend_of[name]))

    mask_clicks = details["click_span"].astype(bool)
    mask_preselected = details["click_span_preselected"].astype(bool) & ~mask_clicks
    mask_reference = details["click_span_reference"].astype(bool) & ~details["click_span_preselected"].astype(bool)
    for (name, color), mask in zip(categories.items(), [mask_clicks, mask_preselected, mask_reference]):
        x, y = mask_polygons(mask, t)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=rgba(MPL_COLORS[color], 0.3),
                                 line=dict(width=0), name=name, hoverinfo="skip", xaxis="x", yaxis="y",
                                 legend=legend_of[name], legendgroup=name))

    # --- Detection map ---
    fig.add_trace(go.Heatmap(
        x=t_edges, y=f_edges, z=np.nan_to_num(R_db[::-1], neginf=-1e3, posinf=1e3, nan=np.nan),
        coloraxis="coloraxis", xaxis="x2", yaxis="y3", name="Envelope / Reference",
        hovertemplate="t = %{x:.4f} s<br>f = %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>"
    ))

    mc = details["mask_clicks"].astype(bool)[::-1, :n]  # Increasing frequency (as f_edges)
    mp = details["mask_protected"].astype(bool)[::-1, :n]
    mr = details["mask_reference"].astype(bool)[::-1, :n]
    band_masks = {"Time-Protected": mr & ~mp, "Non-Clicks": mp & ~mc, "Clicks": mc}  # Clicks on top
    for name, mask in band_masks.items():  # contourf (alpha 0.25)
        x, y = grid_rectangles(mask, t_edges, f_edges)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", line=dict(width=0),
                                 fillcolor=rgba(MPL_COLORS[categories[name]], 0.25), name=name, hoverinfo="skip",
                                 xaxis="x2", yaxis="y3", legend=legend_of[name], legendgroup=name, showlegend=False))

    # Vertical grid (grid(axis="x")) at the initial ticks: plotly draws its grid below the heatmap
    x_grid = mpl_ticks(x_range, ax_w/PT, 3)
    fig.add_trace(go.Scatter(x=np.repeat(x_grid, 3), y=np.tile([f_range[0], f_range[1], np.nan], len(x_grid)),
                             mode="lines", line=dict(color="#b0b0b0", width=pt(0.8)), xaxis="x2", yaxis="y3",
                             hoverinfo="skip", showlegend=False))
    # Band borders (axhline(f_low), black, lw 0.6, alpha 0.6)
    f_lines = [b["f_low"] for b in bands[:-1]]
    fig.add_trace(go.Scatter(x=np.tile([x_range[0], x_range[1], np.nan], len(f_lines)), y=np.repeat(f_lines, 3),
                             mode="lines", line=dict(color="rgba(0, 0, 0, 0.6)", width=pt(0.6)), xaxis="x2",
                             yaxis="y3", hoverinfo="skip", showlegend=False))

    for name, mask in band_masks.items():  # contour (lw 1.5)
        x, y = grid_contour(mask, t_edges, f_edges)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=MPL_COLORS[categories[name]], width=pt(1.5)),
                                 name=name, hoverinfo="skip", xaxis="x2", yaxis="y3", legend=legend_of[name],
                                 legendgroup=name, showlegend=False))
    x, y = grid_contour(np.nan_to_num(R_db[::-1], nan=-np.inf) >= level, t_edges, f_edges)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=MPL_COLORS["blue"], width=pt(1), dash=dash(1)),
                             name=threshold_name, hoverinfo="skip", xaxis="x2", yaxis="y3",
                             legend=legend_of["Threshold"], legendgroup="Threshold"))

    # --- Colorbar (vmin 0, vmax 30) with the threshold (axhline, blue, lw 1.5, dashed) ---
    cb_edges = np.linspace(0, 30, 257)
    fig.add_trace(go.Heatmap(x=[0, 1], y=cb_edges, z=((cb_edges[1:] + cb_edges[:-1])/2)[:, None],
                             coloraxis="coloraxis", xaxis="x3", yaxis="y4", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[level, level], mode="lines", xaxis="x3", yaxis="y4", hoverinfo="skip",
                             line=dict(color=MPL_COLORS["blue"], width=pt(1.5), dash=dash(1.5)), showlegend=False,
                             legend=legend_of["Threshold"], legendgroup="Threshold"))

    # --- Labels and title ---
    ann = lambda s, x, y, size, **kw: dict(text=s, x=X(x), y=Y(y), xref="paper", yref="paper", xanchor="center",
                                           yanchor="middle", showarrow=False, font=dict(size=pt(size)), **kw)
    annotations = [
        ann("Amplitude (A.U)", ax_l - tick_space - w_ytick - pt(4) - line_h(12)/2, (ax0_t + ax0_b)/2, 12,
            textangle=-90),
        ann("Frequency (Hz)", ax_l - tick_space - w_ftick - pt(4) - line_h(12)/2, (ax1_t + ax1_b)/2, 12,
            textangle=-90),
        ann("Time (s)", (ax_l + ax_r)/2, ax1_b + tick_space + line_h(10) + pt(4) + line_h(12)/2, 12),
        ann("Envelope / Reference (dB)", cb_r + tick_space + w_cbtick + pt(4) + line_h(10)/2, (ax1_t + ax1_b)/2,
            10, textangle=-90),
        ann("<b>Detection map</b>", (ax_l + ax_r)/2, ax0_t - pt(30) - (title_asc - title_desc)/2 - 0.1*pt(14), 14),
    ]

    # --- Legend ---
    # matplotlib: borderpad 0.4, handlelength 2, handletextpad 0.8, columnspacing 2 (font sizes); frame with
    # rounded corners (0.2 font sizes), white with alpha 0.8, edge 0.8 gray
    # plotly (single item, no border): the symbol spans [5, 35] px and the text starts at 40 px from the legend x.
    # The title and the legend are raised 0.1 font sizes: the text of Arial (plotly) sits lower than DejaVu Sans
    w_items = [text(name, 10)[0] for name in items]
    leg_w = 2*0.4*em + sum(2.8*em + w for w in w_items) + 2*em*(len(items) - 1)
    leg_h = 2*0.4*em + line_h(10)
    leg_cx, leg_cy = (ax_l + ax_r)/2, ax0_t - 0.06*ax_h
    leg_l = leg_cx - leg_w/2
    legends = {}
    item_x = leg_l + 0.4*em
    for name, w in zip(items, w_items):
        # Text at the matplotlib position (item x + 2.8 font sizes)
        legends[legend_of[name]] = dict(x=X(item_x + 2.8*em - 40), y=Y(leg_cy - 0.1*em), xanchor="left", yanchor="middle",
                                        orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                        font=dict(size=em), groupclick="togglegroup", itemwidth=30)
        item_x += 2.8*em + w + 2*em
    shapes = [dict(type="path", xref="paper", yref="paper", layer="above", fillcolor="rgba(255, 255, 255, 0.8)",
                   line=dict(color="#cccccc", width=pt(1)),
                   path=rounded_rect(X(leg_l), X(leg_l + leg_w), Y(leg_cy + leg_h/2), Y(leg_cy - leg_h/2), 0.2*em,
                                     width, height))]

    purples = colormaps["Purples"]
    colorscale = [[v, "rgb({:.0f}, {:.0f}, {:.0f})".format(*(255*np.array(purples(v)[:3])))]
                  for v in np.linspace(0, 1, 64)]
    fig.update_layout(
        **layout, **legends, width=width, height=height, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        coloraxis=dict(colorscale=colorscale, cmin=0, cmax=30, showscale=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )

    return fig


def plot_impulsiveness(signal: np.ndarray, fs: float, details: dict, params: dict) -> go.Figure:
    """
    Interactive (plotly) version of plot_impulsiveness: same layout (gridspec 5 x 2, figsize 14 x 12 in,
    layout="constrained"), sizes, fonts, colors and positions. The figure is cropped as the inline backend does
    (bbox_inches="tight"), so its size matches the matplotlib image.
    - Layout: the margins of constrained_layout are computed from the decorations of each axes (tick labels,
      labels, titles, legends, colorbar), with the same pads and rules (shared margins per row/column, equal
      margins inside the PSD axes, which spans 4 rows). The pole diagram keeps the aspect ratio 1
    - Legends: one plotly legend per item, at the place of the matplotlib item, inside a common frame. loc="best"
      is computed as matplotlib does (lines and spans that fall inside each candidate position). Clicking an
      item toggles that series; Clicks / Non-Clicks toggle the spans in the 3 time subplots
    - The 4 time axes are linked. Hover: values of the curves, the weights, the z_max of each candidate and the
      frequency of each pole
    - WIDTH: width of the figure (px). Only the axes area gets narrower/wider (the labels, the fonts and the
      heights do not change). None -> as matplotlib
    - DPI: px per inch (matplotlib inline: 100). Every size (fonts, lines, pads) scales with it
    - params: algorithm constants (Z_CLICK, Z_TOLERANCE, AR_ORDER; see summary_plots)
    Limits: the x ticks of the Bode diagram (step 250, which plotly does not use) are fixed (they do not
    change when zooming); plotly draws the grid below the spans
    """
    width, dpi = WIDTH, DPI
    Z_CLICK, Z_TOLERANCE, AR_ORDER = params["Z_CLICK"], params["Z_TOLERANCE"], params["AR_ORDER"]
    PT = dpi/72  # px per point
    pt = lambda v: v*PT  # points -> px
    em = pt(10)  # Font size of the ticks and the legends (px)
    text = lambda s, size, **kw: mpl_text_extent(s, size, dpi, **kw)  # width, above and below the anchor (va)
    text_h = lambda s, size, **kw: sum(text(s, size, **kw)[1:])
    tick_space = pt(3.5) + pt(3.5)  # Tick length + pad
    pad = 0.1*dpi  # bbox_inches="tight" pad
    w_pad = h_pad = 3/72*dpi  # constrained_layout pads (rcParams figure.constrained_layout.w_pad/h_pad)

    # --- Data (as the matplotlib version) ---
    t = np.arange(0, len(signal))/fs
    pred, err, scale = details["prediction_ar"], details["prediction_error_ar"], details["local_scale_ar"]
    z = details["impulsiveness"]
    mask_clicks = details["click_span"].astype(bool)
    mask_preselected = details["click_span_preselected"].astype(bool) & ~mask_clicks

    # Huber weights aligned with t (w[j] -> sample j + AR_ORDER); minimum per pixel so short clicks are seen
    w = np.asarray(details["sample_weights_ar"], dtype=float)
    w_full = np.r_[np.full(len(t) - len(w), np.nan), w]
    w_plot = minimum_filter1d(np.where(~np.isnan(w_full), w_full, 1.0), max(1, len(t) // 800))

    # Maximum of z of each candidate (= features["z_max"])
    tol = round(Z_TOLERANCE*fs)
    peaks = []
    for k, c in enumerate(details["candidates"], 1):
        a, b = max(0, round(c["start"]*fs) - tol), min(len(z), round(c["end"]*fs) + tol)
        i = a + np.argmax(z[a:b])
        peaks.append((k, i, z[i] > Z_CLICK, c))

    # H(z) = 1 / (1 + sum a_k z^-k), a_k = -coefficients of the code
    a_k = -details["coefficients_ar"]
    f_h = np.linspace(0, fs/2, 2048)
    _, H = freqz(1, np.r_[1, a_k], worN=f_h, fs=fs)
    H_db = 20*np.log10(np.abs(H) + EPS)
    poles = np.roots(np.r_[1, a_k])
    outside = np.abs(poles) >= 1
    any_outside = bool(outside.any())
    f_marks = np.arange(0, fs/2 + 1, 500)

    x_ar = signal - signal.mean()  # Same signal as the AR sees (in remove_noise)
    ok = slice(AR_ORDER, None)  # Without the transient (NaN)
    kw = dict(fs=fs, nperseg=min(512, len(x_ar) - AR_ORDER), window="hann")
    f_psd, P_x = welch(x_ar[ok], **kw)
    psd = [10*np.log10(P_x)] + [10*np.log10(welch(v[ok], **kw)[1]) for v in (pred, err)]

    # Ranges (matplotlib autoscale: + 5 %; the time axes are shared with imshow -> no margins)
    ranges = dict(t=[t[0], t[-1]], sig=mpl_range(signal, pred), err=mpl_range(err, scale, -scale), z=mpl_range(z),
                  f_psd=mpl_range(f_psd), psd=mpl_range(*psd), f_h=[0, fs/2], h=mpl_range(H_db, [0]),
                  pz=[-1.4, 1.4])
    z_line = bool(ranges["z"][0] <= Z_CLICK <= ranges["z"][1])

    # Legends: matplotlib labels, plotly names (HTML) and ncol
    legend_specs = {
        "sig": (["Original", "Prediction"], ["Original", "Prediction"], 1),
        "spans": (["Clicks", "Non-Clicks"], ["Clicks", "Non-Clicks"], 2),
        "err": (["Error", r"$\pm\sigma_{loc}$"], ["Error", "±<i>σ</i><sub><i>loc</i></sub>"], 1),
        "z": (["Impulsiveness"] + [f"$Z_{{CLICK}}$ = {Z_CLICK}"]*z_line
              + [r"$z_{max} \geq$ threshold", r"$z_{max} <$ threshold"],
              ["Impulsiveness"] + [f"<i>Z</i><sub><i>CLICK</i></sub> = {Z_CLICK}"]*z_line
              + ["<i>z</i><sub><i>max</i></sub> ≥ threshold", "<i>z</i><sub><i>max</i></sub> < threshold"], 1),
        "psd": (["Original", "Prediction", "Error"], ["Original", "Prediction", "Error"], 1),
        "bode": ([r"$|He^{j\omega}|$"], ["|<i>He</i><sup><i>jω</i></sup>|"], 1),
        "pz": (["Unit Circle (frequency in Hz)", "Poles"] + ["|p| ≥ 1"]*any_outside,
               ["Unit Circle (frequency in Hz)", "Poles"] + ["|p| ≥ 1"]*any_outside, 2),
    }
    leg_layout = {k: mpl_legend_layout(tuple(mpl), ncol, 10, dpi) for k, (mpl, _, ncol) in legend_specs.items()}

    titles = {"sig": ("Original signal vs Prediction", 30), "err": ("Prediction Error", 6),
              "z": ("Impulsiveness", 6), "psd": ("Original vs Prediction vs Error - PSD", 6),
              "bode": ("Autoregressive Model - Bode Diagram", 6), "pz": ("Autoregressive Model - Pole Diagram", 25)}
    title_ext = {name: text(s, 14, weight="bold") for name, (s, _) in titles.items()}  # Baseline at pad pt
    suptitle = f"Impulsiveness calculation - Robust AR({AR_ORDER})"
    ylabels = {"sig": "Amplitude (A.U.)", "err": "Amplitude (A.U.)", "z": "z - Impulsiveness", "psd": "PSD (dB/Hz)",
               "bode": "|H| (dB)", "pz": "Im(z)"}
    xlabels = {"z": "Time (s)", "psd": "Frequency (Hz)", "bode": "Frequency (Hz)", "pz": "Re(z)"}
    y_of = dict(sig="sig", err="err", z="z", psd="psd", bode="h", pz="pz")
    x_of = dict(sig="t", err="t", w="t", z="t", psd="f_psd", bode="f_h", pz="pz")
    cb_labels = ["0", "0,5", "1"]
    w_cb = max(text(s, 7)[0] for s in cb_labels)
    cb_top, cb_bottom = text("1", 7, va="center_baseline")[1], text("0", 7, va="center_baseline")[2]

    def xlabel_stack(name, length):
        """From the axes bottom to the bottom of the x label (tick, pad, tick labels, labelpad, label)"""
        labels = mpl_tick_labels(mpl_ticks(ranges[x_of[name]], length/PT, 3))
        return (tick_space + max(text_h(s, 10, va="top") for s in labels) + pt(4)
                + text_h(xlabels[name], 12, va="top"))

    def ytick_labels(name, length):
        """Width of the y tick labels and how much they stick out of the axes (top, bottom)"""
        r = ranges[y_of[name]]
        ticks = mpl_ticks(r, length/PT, 2)
        pos = (r[1] - ticks)/(r[1] - r[0])*length  # px from the top. Labels: va="center_baseline"
        ext = np.array([text(s, 10, va="center_baseline") for s in mpl_tick_labels(ticks)])
        return ext[:, 0].max(), max(0, -(pos - ext[:, 1]).min()), max(0, (pos + ext[:, 2]).max() - length)

    def xtick_overhang(name, length):
        """How much the x tick labels stick out of the axes (left, right)"""
        r = ranges[x_of[name]]
        ticks = mpl_ticks(r, length/PT, 3)
        pos = (ticks - r[0])/(r[1] - r[0])*length
        half = np.array([text(s, 10)[0] for s in mpl_tick_labels(ticks)])/2
        return max(0, -(pos - half).min()), max(0, (pos + half).max() - length)

    def decorations(name, w, h):
        """Space (px) of the decorations of an axes of size w x h outside its box: left, right, top, bottom"""
        if name == "w":  # Weight label (rotation 0) and colorbar (inset [1.01, 0, 0.015, 1], labels 7 pt)
            return pt(4) + text("Weight", 10)[0], 0.025*w + tick_space + w_cb, cb_top, cb_bottom
        if name == "pz":  # Aspect 1: the box shrinks to a centered square
            s = min(w, h)
            dx, dy = (w - s)/2, (h - s)/2
            w_lab, top, bottom = ytick_labels(name, s)
            o_l, o_r = xtick_overhang(name, s)
            leg_w, leg_h, _ = leg_layout["pz"]  # Centered at (0.5, 1.06)
            return (max(tick_space + w_lab + pt(4) + text_h(ylabels[name], 12), o_l) - dx,
                    max(o_r, (leg_w - s)/2) - dx, max(pt(25) + title_ext[name][1], 0.06*s + leg_h/2, top) - dy,
                    max(xlabel_stack(name, s), bottom) - dy)
        w_lab, top, bottom = ytick_labels(name, h)
        left, right = tick_space + w_lab + pt(4) + text_h(ylabels[name], 12), 0
        top = max(top, pt(titles[name][1]) + title_ext[name][1])
        if name == "sig":  # Spans legend centered at (0.5, 1.1)
            top = max(top, 0.1*h + leg_layout["spans"][1]/2)
        if name in xlabels:
            o_l, o_r = xtick_overhang(name, w)
            left, right, bottom = max(left, o_l), o_r, max(bottom, xlabel_stack(name, w))
        return left, right, top, bottom

    # --- constrained_layout (rows: sig, err, w, z, bode; columns: [sig, err, w, z, bode], [psd (rows 0-3), pz]) ---
    rows = {"sig": 0, "err": 1, "w": 2, "z": 3, "bode": 4, "pz": 4}
    width_ratios, height_ratios = np.array([2.2, 1]), np.array([1, 1, 0.12, 1, 1.5])
    sup_margin = text_h(suptitle, 16, va="top") + 2*h_pad

    def solve(W, H, col_w, row_h):
        """Axes boxes (left, top, right, bottom; px from the top-left of the figure) for the sizes of the
        previous iteration (the decorations depend on the size of the axes)"""
        size = {name: (col_w[0], row_h[r]) for name, r in rows.items() if name != "pz"}
        size["psd"] = (col_w[1], row_h[:4].sum())  # Approximate: only its x tick labels depend on it
        size["pz"] = (col_w[1], row_h[4])
        deco = {name: decorations(name, *s) for name, s in size.items()}
        col0 = ["sig", "err", "w", "z", "bode"]
        L = np.array([max(deco[n][0] for n in col0), max(deco["psd"][0], deco["pz"][0], 0)])
        R = np.array([max(deco[n][1] for n in col0), max(deco["psd"][1], deco["pz"][1], 0)])
        T = np.array([max(deco["sig"][2], deco["psd"][2]), deco["err"][2], deco["w"][2], deco["z"][2],
                      max(deco["bode"][2], deco["pz"][2], 0)])
        B = np.array([deco["sig"][3], deco["err"][3], deco["w"][3], max(deco["z"][3], deco["psd"][3]),
                      max(deco["bode"][3], deco["pz"][3], 0)])
        T[1:4], B[0:3] = T[1:4].max(), B[0:3].max()  # Margins inside the PSD axes: all the same
        w_cb, h_cb = max(w_pad, 0.02/2/2*W), max(h_pad, 0.02/2/5*H)  # Padding (inside the grid: w/hspace)
        Lcb, Rcb = np.array([w_pad, w_cb]), np.array([w_cb, w_pad])
        Tcb, Bcb = np.array([h_pad] + [h_cb]*4), np.array([h_cb]*4 + [h_pad])
        col_w = width_ratios*(W - (L + R + Lcb + Rcb).sum())/width_ratios.sum()
        row_h = height_ratios*(H - sup_margin - (T + B + Tcb + Bcb).sum())/height_ratios.sum()
        lefts = np.cumsum(np.r_[0, (Lcb + L + col_w + R + Rcb)[:-1]]) + Lcb + L
        tops = sup_margin + np.cumsum(np.r_[0, (Tcb + T + row_h + B + Bcb)[:-1]]) + Tcb + T
        boxes = {name: (lefts[0], tops[r], lefts[0] + col_w[0], tops[r] + row_h[r]) for name, r in rows.items()}
        boxes["psd"] = (lefts[1], tops[0], lefts[1] + col_w[1], tops[3] + row_h[3])
        s = min(col_w[1], row_h[4])  # Pole diagram: square, centered in its cell
        cx, cy = lefts[1] + col_w[1]/2, tops[4] + row_h[4]/2
        boxes["pz"] = (cx - s/2, cy - s/2, cx + s/2, cy + s/2)
        return boxes, col_w, row_h

    def layout_for(W):
        H = 12*dpi
        col_w, row_h = width_ratios*0.4*W, height_ratios*0.15*H
        for _ in range(4):  # Fixed point (matplotlib: 2 passes, the difference is < 0.1 px)
            boxes, col_w, row_h = solve(W, H, col_w, row_h)
        # Bounding box of every element (bbox_inches="tight"): x0, x1, top, bottom
        sup_w = text(suptitle, 16)[0]
        ext = [(W/2 - sup_w/2, W/2 + sup_w/2, h_pad, h_pad + text_h(suptitle, 16, va="top"))]
        for name, (l, tp, r, b) in boxes.items():
            ext.append((l, r, tp, b))
            if name in titles:
                tw = title_ext[name][0]
                ext.append(((l + r - tw)/2, (l + r + tw)/2, tp - pt(titles[name][1]) - title_ext[name][1], tp))
            if name in ylabels:
                ext.append((l - (tick_space + ytick_labels(name, b - tp)[0] + pt(4) + text_h(ylabels[name], 12)), l,
                            tp, b))
            if name in xlabels:
                ext.append((l - xtick_overhang(name, r - l)[0], r + xtick_overhang(name, r - l)[1], b,
                            b + xlabel_stack(name, r - l)))
        l, tp, r, b = boxes["w"]
        ext.append((l - pt(4) - text("Weight", 10)[0], r + 0.025*(r - l) + tick_space + w_cb, tp - cb_top,
                    b + cb_bottom))
        x0, x1 = min(e[0] for e in ext) - pad, max(e[1] for e in ext) + pad
        y0, y1 = min(e[2] for e in ext) - pad, max(e[3] for e in ext) + pad
        return boxes, x0, y0, int(x1 - x0), int(y1 - y0)

    W = 14*dpi
    boxes, crop_x, crop_y, img_w, img_h = layout_for(W)
    for _ in range(4 if width is not None else 0):  # Only the axes area changes with width
        W += width - img_w
        boxes, crop_x, crop_y, img_w, img_h = layout_for(W)
    width, height = img_w, img_h
    boxes = {k: (l - crop_x, tp - crop_y, r - crop_x, b - crop_y) for k, (l, tp, r, b) in boxes.items()}
    W_fig = W
    X = lambda v: v/width  # px -> paper
    Y = lambda v: 1 - v/height

    # Data -> px of each axes (for loc="best")
    def to_px(name, x, y, y_name=None):
        l, tp, r, b = boxes[name]
        rx, ry = ranges[x_of[name]], ranges[y_name or y_of[name]]
        return (l + (np.asarray(x, float) - rx[0])/(rx[1] - rx[0])*(r - l),
                b - (np.asarray(y, float) - ry[0])/(ry[1] - ry[0])*(b - tp))

    def span_paths(name):
        """Rectangles of the spans (fill_between with the axes transform), one path per run"""
        l, tp, r, b = boxes[name]
        paths = []
        for mask in (mask_clicks, mask_preselected):
            d = np.diff(np.r_[0, mask.astype(np.int8), 0])
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
                xs = to_px(name, [t[s], t[e]], [0, 0])[0]
                paths.append((np.array([xs[0], xs[1], xs[1], xs[0], xs[0]]), np.array([b, b, tp, tp, b])))
        return paths

    def finite_runs(x, y1, y2):
        """Polygons of fill_between(x, y1, y2) (one per run of finite values)"""
        good = np.isfinite(y1) & np.isfinite(y2)
        d = np.diff(np.r_[0, good.astype(np.int8), 0])
        return [(np.r_[x[s:e], x[s:e][::-1], x[s]], np.r_[y2[s:e], y1[s:e][::-1], y2[s]])
                for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]

    z_l, _, z_r, _ = boxes["z"]
    paths = {
        "sig": [to_px("sig", t, signal), to_px("sig", t, pred)] + span_paths("sig"),
        "err": ([to_px("err", t, err)] + [to_px("err", *p) for p in finite_runs(t, -scale, scale)]
                + span_paths("err")),
        "z": ([to_px("z", t, z)] + [to_px("z", [t[i]], [z[i]]) for _, i, _, _ in peaks]
              + ([(np.array([z_l, z_r]), to_px("z", [0, 0], [Z_CLICK]*2)[1])] if z_line else []) + span_paths("z")),
        "psd": [to_px("psd", f_psd, p) for p in psd],
        "bode": [to_px("bode", f_h, H_db), (np.array(boxes["bode"][0:3:2]), to_px("bode", [0, 0], [0, 0])[1])],
    }

    # Legend frames (left, top) and plotly legends (one per item)
    frames, legend_of, legends = {}, {}, {}
    for key, (mpl_labels, names, ncol) in legend_specs.items():
        leg_w, leg_h, item_boxes = leg_layout[key]
        if key in ("spans", "pz"):  # loc="center", bbox_to_anchor=(0.5, 1.1) / (0.5, 1.06)
            l, tp, r, b = boxes["sig" if key == "spans" else "pz"]
            cy = tp - (0.1 if key == "spans" else 0.06)*(b - tp)
            frames[key] = ((l + r - leg_w)/2, cy - leg_h/2)
        else:
            frames[key] = mpl_best_legend(boxes[key], (leg_w, leg_h), paths[key], em)
        fl, ft = frames[key]
        for k, (x0, _, top, bottom) in enumerate(item_boxes):
            lid = "legend" if not legends else f"legend{len(legends) + 1}"
            legend_of[(key, k)] = lid
            # Text at the matplotlib position (plotly: symbol at [5, 35] px, text at 40 px from the legend x).
            # Raised 0.1 font sizes: Arial (plotly) sits lower than DejaVu Sans
            legends[lid] = dict(x=X(fl + x0 - 40), y=Y(ft + (top + bottom)/2 - 0.1*em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)

    def item(key, k):
        """Legend arguments of the item k of the legend key"""
        return dict(name=legend_specs[key][1][k], legend=legend_of[(key, k)], legendgroup=f"{key}{k}")

    # --- Axes ---
    axis_style = dict(showline=True, mirror=True, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none", tickformatstops=MPL_TICKFORMAT)

    def ticks_of(r, length, label_factor):
        """matplotlib ticks: nticks (plotly keeps its automatic ticks when zooming) or, for steps of 2.5 x 10^k
        (plotly only uses 1, 2, 5), fixed ticks"""
        ticks = mpl_ticks(r, length/PT, label_factor)
        step = ticks[1] - ticks[0]
        if np.isclose(step/10**np.floor(np.log10(step)), 2.5):
            return dict(tickmode="linear", tick0=ticks[0], dtick=step)
        return dict(nticks=mpl_nticks(r, length/PT, label_factor))

    def x_axis(name, labels=True, grid=True, **kw):
        l, tp, r, b = boxes[name]
        rng = ranges[x_of[name]]
        return dict(domain=[X(l), X(r)], range=rng, **{**axis_style, "showgrid": grid}, showticklabels=labels,
                    ticklabelstandoff=round(-2.2*PT), **ticks_of(rng, r - l, 3), **kw)

    def y_axis(name, grid=True, **kw):
        l, tp, r, b = boxes[name]
        rng = ranges[y_of[name]]
        return dict(domain=[Y(b), Y(tp)], range=rng, **{**axis_style, "showgrid": grid}, **ticks_of(rng, b - tp, 2),
                    **kw)

    hidden = lambda name, **kw: dict(domain=[Y(boxes[name][3]), Y(boxes[name][1])], range=[0, 1], fixedrange=True,
                                     visible=False, **kw)
    l, tp, r, b = boxes["w"]
    cb_l, cb_r = r + 0.01*(r - l), r + 0.025*(r - l)
    layout = dict(
        xaxis=x_axis("sig", labels=False, anchor="y"),  # A: signal and prediction (y: spans, y2: data)
        yaxis=hidden("sig", anchor="x"),
        yaxis2=y_axis("sig", overlaying="y", anchor="x", side="left"),
        xaxis2=x_axis("err", labels=False, anchor="y3", matches="x"),  # B: error (y4: spans, y5: +-sigma)
        yaxis3=y_axis("err", anchor="x2"),
        yaxis4=dict(overlaying="y3", anchor="x2", range=[0, 1], fixedrange=True, visible=False),
        yaxis5=dict(overlaying="y3", anchor="x2", matches="y3", visible=False),
        xaxis3=x_axis("w", labels=False, grid=False, anchor="y6", matches="x"),  # Huber weights
        yaxis6=dict(domain=[Y(b), Y(tp)], anchor="x3", range=[0, 1], fixedrange=True, showline=True, mirror=True,
                    linecolor="black", linewidth=pt(0.8), ticks="", showticklabels=False, showgrid=False,
                    zeroline=False),
        xaxis4=x_axis("z", anchor="y7", matches="x"),  # D: impulsiveness (y7: spans, y8: data)
        yaxis7=hidden("z", anchor="x4"),
        yaxis8=y_axis("z", overlaying="y7", anchor="x4", side="left"),
        xaxis5=x_axis("psd", anchor="y9"),  # C: PSD
        yaxis9=y_axis("psd", anchor="x5"),
        xaxis6=x_axis("bode", anchor="y10"),  # F: |H|
        yaxis10=y_axis("bode", anchor="x6"),
        xaxis7=x_axis("pz", grid=False, anchor="y11"),  # G: poles (aspect 1)
        yaxis11=y_axis("pz", grid=False, anchor="x7", scaleanchor="x7", scaleratio=1),
        xaxis8=dict(domain=[X(cb_l), X(cb_r)], anchor="y12", range=[0, 1], fixedrange=True, showline=True,
                    mirror=True, linecolor="black", linewidth=pt(0.8), showticklabels=False, ticks="",
                    showgrid=False, zeroline=False),  # Colorbar of the weights
        yaxis12=dict(domain=[Y(b), Y(tp)], anchor="x8", side="right", range=[0, 1], fixedrange=True,
                     **{**axis_style, "showgrid": False, "tickfont": dict(size=pt(7))}, tickvals=[0, 0.5, 1],
                     ticktext=cb_labels),
    )

    fig = go.Figure()
    hover_t = "%{x:.4f} s, %{y:.4g}"
    line = lambda color, lw, dash=None: dict(color=MPL_COLORS[color], width=pt(lw), dash=dash)
    dotted = lambda lw: f"{pt(lw):.2f}px,{pt(1.65*lw):.2f}px"  # ":" (lines.dotted_pattern 1, 1.65)
    dashed = lambda lw: f"{pt(3.7*lw):.1f}px,{pt(1.6*lw):.1f}px"  # "--"

    def add_spans(xa, ya, show):
        for k, (mask, color) in enumerate([(mask_clicks, "tab:red"), (mask_preselected, "tab:green")]):
            x, y = mask_polygons(mask, t)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=rgba(MPL_COLORS[color], 0.3),
                                     line=dict(width=0), hoverinfo="skip", xaxis=xa, yaxis=ya, showlegend=show,
                                     **item("spans", k)))

    # A: signal and prediction
    add_spans("x", "y", True)
    fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=signal, mode="lines", line=line("gray", 1), xaxis="x", yaxis="y2",
                             hovertemplate=hover_t, **item("sig", 0)))
    fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=pred, mode="lines", line=line("tab:blue", 1, dotted(1)), xaxis="x",
                             yaxis="y2", hovertemplate=hover_t, **item("sig", 1)))

    # B: error and +-sigma_loc (above the spans, as zorder=2)
    fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=err, mode="lines", line=line("tab:red", 1.5), xaxis="x2",
                             yaxis="y3", hovertemplate=hover_t, **item("err", 0)))
    add_spans("x2", "y4", False)
    xs, ys = [], []
    for px, py in finite_runs(t, -scale, scale):
        xs += px.tolist() + [None]
        ys += py.tolist() + [None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", fillcolor=rgba(MPL_COLORS["tab:cyan"], 0.6),
                             line=dict(width=0), hoverinfo="skip", xaxis="x2", yaxis="y5", **item("err", 1)))

    # Huber weights (imshow, cmap "hot", nearest) and colorbar
    hot = mpl_colorscale("hot")
    fig.add_trace(go.Heatmap(x=np.linspace(t[0], t[-1], len(w_plot) + 1), y=[0, 1], z=w_plot[None, :], zmin=0,
                             zmax=1, colorscale=hot, showscale=False, xaxis="x3", yaxis="y6", name="Weight",
                             hovertemplate="%{x:.4f} s, weight %{z:.2f}<extra></extra>"))
    cb_edges = np.linspace(0, 1, 257)
    fig.add_trace(go.Heatmap(x=[0, 1], y=cb_edges, z=((cb_edges[1:] + cb_edges[:-1])/2)[:, None], zmin=0, zmax=1,
                             colorscale=hot, showscale=False, xaxis="x8", yaxis="y12", hoverinfo="skip"))

    # D: impulsiveness, z_max of the candidates and Z_CLICK
    add_spans("x4", "y7", False)
    fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=z, mode="lines", line=line("blue", 1.5), xaxis="x4", yaxis="y8",
                             hovertemplate=hover_t, **item("z", 0)))
    if z_line:
        fig.add_trace(go.Scatter(x=ranges["t"], y=[Z_CLICK]*2, mode="lines", line=line("tab:red", 1, dashed(1)),
                                 xaxis="x4", yaxis="y8", hoverinfo="skip", **item("z", 1)))
    for k, (is_click, color) in enumerate([(True, "tab:red"), (False, "tab:green")], 1 + z_line):
        group = [(n, i, c) for n, i, click, c in peaks if click == is_click]
        fig.add_trace(go.Scatter(
            x=[t[i] for _, i, _ in group] or [None], y=[z[i] for _, i, _ in group] or [None], mode="markers",
            marker=dict(color=MPL_COLORS[color], size=pt(6), line=dict(color="black", width=pt(0.6))),
            customdata=[[n, c["start"], c["end"]] for n, _, c in group] or None, xaxis="x4", yaxis="y8",
            hovertemplate="Candidate %{customdata[0]}: %{customdata[1]:.3f}-%{customdata[2]:.3f} s<br>"
                          "z<sub>max</sub> = %{y:.2f}<extra></extra>", **item("z", k)))

    # C: PSD
    for k, (p, color, lw, dash) in enumerate([(psd[0], "gray", 1.6, None), (psd[1], "tab:blue", 1, dotted(1)),
                                              (psd[2], "tab:red", 1, None)]):
        fig.add_trace(go.Scatter(x=f_psd, y=p, mode="lines", line=line(color, lw, dash), xaxis="x5", yaxis="y9",
                                 hovertemplate="%{x:.0f} Hz, %{y:.1f} dB/Hz", **item("psd", k)))

    # F: Bode diagram
    fig.add_trace(go.Scatter(x=f_h, y=H_db, mode="lines", line=line("tab:purple", 2), xaxis="x6", yaxis="y10",
                             hovertemplate="%{x:.0f} Hz, %{y:.1f} dB", **item("bode", 0)))
    fig.add_trace(go.Scatter(x=ranges["f_h"], y=[0, 0], mode="lines", line=line("gray", 1, dotted(1)), xaxis="x6",
                             yaxis="y10", hoverinfo="skip", showlegend=False))

    # G: poles, unit circle with the frequency marks (angle = 2 pi f / fs)
    th = np.linspace(0, 2*np.pi, 512)
    fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=line("gray", 1, dashed(1)), xaxis="x7",
                             yaxis="y11", hoverinfo="skip", **item("pz", 0)))
    fig.add_trace(go.Scatter(x=[-1.4, 1.4, None, 0, 0], y=[0, 0, None, -1.4, 1.4], mode="lines",
                             line=dict(color="black", width=pt(0.1)), xaxis="x7", yaxis="y11", hoverinfo="skip",
                             showlegend=False))
    ang = 2*np.pi*f_marks/fs
    fig.add_trace(go.Scatter(x=np.c_[0.95*np.cos(ang), 1.05*np.cos(ang), np.full_like(ang, np.nan)].ravel(),
                             y=np.c_[0.95*np.sin(ang), 1.05*np.sin(ang), np.full_like(ang, np.nan)].ravel(),
                             mode="lines", line=line("gray", 0.8), xaxis="x7", yaxis="y11", hoverinfo="skip",
                             showlegend=False, legend=legend_of[("pz", 0)], legendgroup="pz0"))
    fig.add_trace(go.Scatter(x=1.2*np.cos(ang), y=1.2*np.sin(ang), mode="text", text=[f"{v:g}" for v in f_marks],
                             textposition="middle center", textfont=dict(size=pt(7), color="black"), xaxis="x7",
                             yaxis="y11", hoverinfo="skip", showlegend=False, legend=legend_of[("pz", 0)],
                             legendgroup="pz0"))
    f_poles = np.abs(np.angle(poles))*fs/(2*np.pi)
    fig.add_trace(go.Scatter(x=poles.real, y=poles.imag, mode="markers", customdata=np.c_[np.abs(poles), f_poles],
                             marker=dict(symbol="x-thin", size=pt(np.sqrt(40)), color=MPL_COLORS["tab:purple"],
                                         line=dict(color=MPL_COLORS["tab:purple"], width=pt(1.5))),
                             xaxis="x7", yaxis="y11", hovertemplate="p = %{x:.3f} %{y:+.3f}j<br>|p| = "
                             "%{customdata[0]:.3f}<br>f = %{customdata[1]:.0f} Hz<extra></extra>", **item("pz", 1)))
    if any_outside:
        fig.add_trace(go.Scatter(x=poles.real[outside], y=poles.imag[outside], mode="markers", hoverinfo="skip",
                                 marker=dict(symbol="circle-open", size=pt(np.sqrt(140)),
                                             line=dict(color=MPL_COLORS["tab:red"], width=pt(1.2))),
                                 xaxis="x7", yaxis="y11", **item("pz", 2)))

    # --- Titles, labels, legend frames ---
    ann = lambda s, x, y, size, xanchor="center", **kw: dict(text=s, x=X(x), y=Y(y), xref="paper", yref="paper",
                                                             xanchor=xanchor, yanchor="middle", showarrow=False,
                                                             font=dict(size=pt(size)), **kw)
    annotations = [ann(suptitle, W_fig/2 - crop_x, h_pad - crop_y + text_h(suptitle, 16, va="top")/2 - 0.1*pt(16),
                       16)]
    for name, (l, tp, r, b) in boxes.items():
        if name in titles:
            baseline = tp - pt(titles[name][1])
            annotations.append(ann(f"<b>{titles[name][0]}</b>", (l + r)/2,
                                   baseline - (title_ext[name][1] - title_ext[name][2])/2 - 0.1*pt(14), 14))
        if name in ylabels:
            x = l - tick_space - ytick_labels(name, b - tp)[0] - pt(4) - text_h(ylabels[name], 12)/2
            annotations.append(ann(ylabels[name], x, (tp + b)/2, 12, textangle=-90))
        if name in xlabels:
            annotations.append(ann(xlabels[name], (l + r)/2,
                                   b + xlabel_stack(name, r - l) - text_h(xlabels[name], 12, va="top")/2, 12))
    l, tp, r, b = boxes["w"]
    annotations.append(ann("Weight", l - pt(4), (tp + b)/2, 10, xanchor="right"))
    shapes = []
    for key, (fl, ft) in frames.items():
        leg_w, leg_h, _ = leg_layout[key]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above",
                           fillcolor="rgba(255, 255, 255, 0.8)", line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(fl), X(fl + leg_w), Y(ft + leg_h), Y(ft), 0.2*em, width, height)))

    fig.update_layout(
        **layout, **legends, width=width, height=height, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )

    return fig


def plot_isolation(signal: np.ndarray, fs: float, details: dict, params: dict) -> go.Figure:
    """
    Interactive (plotly) version of plot_isolation: same layout (3 x 1 subplots, figsize 14 x 9 in,
    layout="constrained"), sizes, fonts, colors and positions. The figure is cropped as the inline backend does
    (bbox_inches="tight"), so its size matches the matplotlib image.
    - Top: original signal and full-band envelope, with the candidate spans
    - Middle: isolation windows (ISO_WIN s at each side of each span) and their medians (thick: the one used as
      denominator, dashed: the other side), with the envelope peak of each span
    - Bottom: isolation of each span (envelope peak / larger median) as stems from 1, and the thresholds that fall
      inside the axes
    - Legends: one plotly legend per item, at the place of the matplotlib item, inside a common frame (loc="best"
      computed as matplotlib does). Clicking an item toggles that series; Clicks / Non-Clicks toggle the spans in
      the 3 subplots. The time axes are linked. Hover: peak, medians and isolation of each span
    - WIDTH: width of the figure (px). Only the axes area gets narrower/wider (the labels, the fonts and the
      heights do not change). None -> as matplotlib
    - DPI: px per inch (matplotlib inline: 100). Every size (fonts, lines, pads) scales with it
    - params: algorithm constants (ISO_WIN, ISOLATION, ISOLATION_IMPULSIVE, ISOLATION_ALIGNED; see summary_plots)
    Limit: plotly draws the grid below the spans and the isolation windows
    """
    width, dpi = WIDTH, DPI
    ISO_WIN, ISOLATION = params["ISO_WIN"], params["ISOLATION"]
    ISOLATION_IMPULSIVE, ISOLATION_ALIGNED = params["ISOLATION_IMPULSIVE"], params["ISOLATION_ALIGNED"]
    PT = dpi/72  # px per point
    pt = lambda v: v*PT  # points -> px
    em = pt(10)  # Font size of the ticks and the legends (px)
    text = lambda s, size, **kw: mpl_text_extent(s, size, dpi, **kw)  # width, above and below the anchor (va)
    text_h = lambda s, size, **kw: sum(text(s, size, **kw)[1:])
    tick_space = pt(3.5) + pt(3.5)  # Tick length + pad
    pad = 0.1*dpi  # bbox_inches="tight" pad
    w_pad = h_pad = 3/72*dpi  # constrained_layout pads (rcParams figure.constrained_layout.w_pad/h_pad)

    # --- Data (as the matplotlib version) ---
    t = np.arange(0, len(signal))/fs
    env = details["signal_envelope"]
    iso_win = round(ISO_WIN*fs)
    N = len(env)
    mask_clicks = details["click_span"].astype(bool)
    mask_preselected = details["click_span_preselected"].astype(bool) & ~mask_clicks
    mask = mask_clicks | mask_preselected
    iso_min_thr = min(ISOLATION_ALIGNED, ISOLATION_IMPULSIVE, ISOLATION)

    # Spans: valid sides (same criterion as find_click_events), medians, envelope peak and isolation
    lab, _ = ndi.label(mask)
    mask_iso = mask.copy()
    spans = []
    for a, b in [(s.start, s.stop) for (s,) in ndi.find_objects(lab)]:
        sides = []
        for lo, hi in ((max(0, a - iso_win), a), (b, min(N, b + iso_win))):
            if hi - lo >= iso_win // 4:
                sides.append((lo, hi, np.median(env[lo:hi])))
                mask_iso[lo:hi] = True
        i = a + np.argmax(env[a:b])
        k_bg = int(np.argmax([med for _, _, med in sides])) if sides else None  # Side of the denominator
        iso = env[i] / sides[k_bg][2] if sides else 0
        spans.append(dict(a=a, b=b, sides=sides, k_bg=k_bg, i=i, iso=iso))
    env_iso = np.where(mask_iso, env, np.nan)
    medians = [s[2] for sp in spans for s in sp["sides"]]
    isos = [sp["iso"] for sp in spans if sp["iso"] > 0]

    # Ranges (matplotlib autoscale: + 5 %; axhline(1) and the stems from 1 count in the isolation axes)
    ranges = dict(t=mpl_range(t), sig=mpl_range(signal, env), env=mpl_range(env_iso, medians or [np.nan]),
                  iso=mpl_range([1], isos))
    thresholds = [(f"{label} (iso = {thr:g})", thr, ls) for label, thr, ls in
                  [("Aligned threshold", ISOLATION_ALIGNED, ":"), ("Impulsive threshold", ISOLATION_IMPULSIVE, "-."),
                   ("Transient threshold", ISOLATION, "-")] if ranges["iso"][0] <= thr <= ranges["iso"][1]]
    has_windows = any(sp["sides"] for sp in spans)

    # Legends: matplotlib labels (= plotly names) and ncol
    legend_specs = {
        "sig": (["Original", "Envelope"], 1),
        "spans": (["Clicks", "Non-Clicks"], 2),
        "env": (["Isolation window"]*has_windows
                + ["Median (denominator)", "Median (other side)", "Envelope (in the windows of interest)"], 1),
        "iso": ([label for label, _, _ in thresholds] + [f"Isolation ≥ {iso_min_thr:g}", f"Isolation < {iso_min_thr:g}"],
                1),
    }
    leg_layout = {k: mpl_legend_layout(tuple(labels), ncol, 10, dpi) for k, (labels, ncol) in legend_specs.items()}

    names = ["sig", "env", "iso"]
    titles = {"sig": ("Original signal", 25), "env": ("Isolation windows", 6), "iso": ("Isolation per span", 6)}
    title_ext = {name: text(s, 14, weight="bold") for name, (s, _) in titles.items()}  # Baseline at pad pt
    ylabels = {"sig": "Amplitude (A.U.)", "env": "Amplitude (A.U.)", "iso": "Isolation"}
    suptitle = "Isolation calculation"
    x_ticks = lambda length: mpl_ticks(ranges["t"], length/PT, 3)
    xtick_h = lambda length: max(text_h(s, 10, va="top") for s in mpl_tick_labels(x_ticks(length)))

    def ytick_labels(name, length):
        """Width of the y tick labels and how much they stick out of the axes (top, bottom)"""
        r = ranges[name]
        ticks = mpl_ticks(r, length/PT, 2)
        pos = (r[1] - ticks)/(r[1] - r[0])*length  # px from the top. Labels: va="center_baseline"
        ext = np.array([text(s, 10, va="center_baseline") for s in mpl_tick_labels(ticks)])
        return ext[:, 0].max(), max(0, -(pos - ext[:, 1]).min()), max(0, (pos + ext[:, 2]).max() - length)

    def xtick_overhang(length):
        """How much the x tick labels stick out of the axes (left, right)"""
        ticks = x_ticks(length)
        r = ranges["t"]
        pos = (ticks - r[0])/(r[1] - r[0])*length
        half = np.array([text(s, 10)[0] for s in mpl_tick_labels(ticks)])/2
        return max(0, -(pos - half).min()), max(0, (pos + half).max() - length)

    def xlabel_stack(name, length):
        """From the axes bottom to the bottom of the x tick labels (and of the x label in the last axes)"""
        return tick_space + xtick_h(length) + (pt(4) + text_h("Time (s)", 12, va="top"))*(name == "iso")

    def decorations(name, w, h):
        """Space (px) of the decorations of an axes of size w x h outside its box: left, right, top, bottom"""
        w_lab, top, bottom = ytick_labels(name, h)
        o_l, o_r = xtick_overhang(w)
        top = max(top, pt(titles[name][1]) + title_ext[name][1])
        if name == "sig":  # Spans legend centered at (0.5, 1.07)
            top = max(top, 0.07*h + leg_layout["spans"][1]/2)
        return (max(tick_space + w_lab + pt(4) + text_h(ylabels[name], 12), o_l), o_r, top,
                max(bottom, xlabel_stack(name, w)))

    # --- constrained_layout (3 x 1, same heights) ---
    sup_margin = text_h(suptitle, 16, va="top") + 2*h_pad

    def layout_for(W):
        H = 9*dpi
        w, h = 0.8*W, 0.25*H
        for _ in range(3):  # The decorations depend on the size of the axes (number of ticks)
            deco = [decorations(name, w, h) for name in names]
            L, R = max(d[0] for d in deco), max(d[1] for d in deco)
            T, B = np.array([d[2] for d in deco]), np.array([d[3] for d in deco])
            w = W - L - R - 2*w_pad
            h = (H - sup_margin - (T + B).sum() - 2*3*h_pad)/3
        tops = sup_margin + np.cumsum(np.r_[0, (T + h + B + 2*h_pad)[:-1]]) + h_pad + T
        boxes = {name: (w_pad + L, tp, w_pad + L + w, tp + h) for name, tp in zip(names, tops)}
        # Bounding box of every element (bbox_inches="tight"): x0, x1, top, bottom
        sup_w = text(suptitle, 16)[0]
        ext = [(W/2 - sup_w/2, W/2 + sup_w/2, h_pad, h_pad + text_h(suptitle, 16, va="top"))]
        for name, (l, tp, r, b) in boxes.items():
            tw = title_ext[name][0]
            ext.append(((l + r - tw)/2, (l + r + tw)/2, tp - pt(titles[name][1]) - title_ext[name][1], tp))
            ext.append((l - (tick_space + ytick_labels(name, b - tp)[0] + pt(4) + text_h(ylabels[name], 12)), r, tp, b))
            o_l, o_r = xtick_overhang(r - l)
            ext.append((l - o_l, r + o_r, b, b + xlabel_stack(name, r - l)))
        x0, x1 = min(e[0] for e in ext) - pad, max(e[1] for e in ext) + pad
        y0, y1 = min(e[2] for e in ext) - pad, max(e[3] for e in ext) + pad
        return boxes, x0, y0, int(x1 - x0), int(y1 - y0)

    W = 14*dpi
    boxes, crop_x, crop_y, img_w, img_h = layout_for(W)
    for _ in range(4 if width is not None else 0):  # Only the axes area changes with width
        W += width - img_w
        boxes, crop_x, crop_y, img_w, img_h = layout_for(W)
    width, height = img_w, img_h
    boxes = {k: (l - crop_x, tp - crop_y, r - crop_x, b - crop_y) for k, (l, tp, r, b) in boxes.items()}
    X = lambda v: v/width  # px -> paper
    Y = lambda v: 1 - v/height

    # --- loc="best": what matplotlib counts (lines, markers, fill_between polygons, axvspan rectangles) ---
    def to_px(name, x, y=None, y_axes=None):
        """Data (or y in axes fraction) -> px"""
        l, tp, r, b = boxes[name]
        rx, ry = ranges["t"], ranges[name]
        px = l + (np.asarray(x, float) - rx[0])/(rx[1] - rx[0])*(r - l)
        py = (b - np.asarray(y_axes, float)*(b - tp) if y is None
              else b - (np.asarray(y, float) - ry[0])/(ry[1] - ry[0])*(b - tp))
        return px, py

    def span_paths(name):
        """Rectangles of the spans (fill_between with the axes transform), one path per run"""
        paths = []
        for m in (mask_clicks, mask_preselected):
            d = np.diff(np.r_[0, m.astype(np.int8), 0])
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
                paths.append(to_px(name, [t[s], t[e], t[e], t[s], t[s]], y_axes=[0, 0, 1, 1, 0]))
        return paths

    axes_line = lambda name, y: (np.array([boxes[name][0], boxes[name][2]]), to_px(name, [0, 0], [y, y])[1])
    window_rects = []
    for sp in spans:
        for lo, hi, _ in sp["sides"]:
            x0, x1 = to_px("env", [t[lo], t[hi - 1]], [0, 0])[0]
            window_rects.append((x0, boxes["env"][1], x1, boxes["env"][3]))
    paths = {
        "sig": [to_px("sig", t, signal), to_px("sig", t, env)] + span_paths("sig"),
        "env": [to_px("env", t, env_iso)] + [to_px("env", [t[sp["i"]]], [env[sp["i"]]]) for sp in spans]
               + span_paths("env"),
        "iso": ([axes_line("iso", 1)] + [axes_line("iso", thr) for _, thr, _ in thresholds]
                + [to_px("iso", [t[sp["i"]]], [sp["iso"]]) if sp["iso"] > 0 else to_px("iso", [t[sp["i"]]], y_axes=[0.04])
                   for sp in spans] + span_paths("iso")),
    }

    # Legend frames (left, top) and plotly legends (one per item)
    frames, legend_of, legends = {}, {}, {}
    for key, (labels, ncol) in legend_specs.items():
        leg_w, leg_h, item_boxes = leg_layout[key]
        if key == "spans":  # loc="center", bbox_to_anchor=(0.5, 1.07)
            l, tp, r, b = boxes["sig"]
            frames[key] = ((l + r - leg_w)/2, tp - 0.07*(b - tp) - leg_h/2)
        else:
            frames[key] = mpl_best_legend(boxes[key], (leg_w, leg_h), paths[key], em,
                                          rects=window_rects if key == "env" else ())
        fl, ft = frames[key]
        for k, (x0, _, top, bottom) in enumerate(item_boxes):
            lid = "legend" if not legends else f"legend{len(legends) + 1}"
            legend_of[(key, k)] = lid
            # Text at the matplotlib position (plotly: symbol at [5, 35] px, text at 40 px from the legend x).
            # Raised 0.1 font sizes: Arial (plotly) sits lower than DejaVu Sans
            legends[lid] = dict(x=X(fl + x0 - 40), y=Y(ft + (top + bottom)/2 - 0.1*em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)

    def item(key, label, show=True):
        """Legend arguments of the item with this label of the legend key"""
        k = legend_specs[key][0].index(label)
        return dict(name=label, legend=legend_of[(key, k)], legendgroup=f"{key}{k}", showlegend=show)

    # --- Axes ---
    axis_style = dict(showline=True, mirror=True, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none", tickformatstops=MPL_TICKFORMAT)
    layout = {}
    for n, name in enumerate(names):
        l, tp, r, b = boxes[name]
        xa, ya_hidden, ya = f"xaxis{n + 1 if n else ''}", f"yaxis{2*n + 1 if n else ''}", f"yaxis{2*n + 2}"
        x_ref, y_hidden_ref = f"x{n + 1 if n else ''}", f"y{2*n + 1 if n else ''}"
        layout[xa] = dict(domain=[X(l), X(r)], anchor=y_hidden_ref, range=ranges["t"], **axis_style,
                          nticks=mpl_nticks(ranges["t"], (r - l)/PT, 3), ticklabelstandoff=round(-2.2*PT),
                          **({"matches": "x"} if n else {}))
        # Hidden axis [0, 1] (spans, windows: axes transform) below the data axis
        layout[ya_hidden] = dict(domain=[Y(b), Y(tp)], anchor=x_ref, range=[0, 1], fixedrange=True, visible=False)
        layout[ya] = dict(overlaying=y_hidden_ref, anchor=x_ref, side="left", range=ranges[name], **axis_style,
                          nticks=mpl_nticks(ranges[name], (b - tp)/PT, 2))
    axes_of = {name: (f"x{n + 1 if n else ''}", f"y{2*n + 1 if n else ''}", f"y{2*n + 2}") for n, name in enumerate(names)}

    fig = go.Figure()
    line = lambda color, lw, ls="-": dict(color=MPL_COLORS[color], width=pt(lw),
                                          dash=None if ls == "-" else mpl_dashes(DASH_PATTERNS[ls], lw, PT))
    marker = lambda color, size=6: dict(color=MPL_COLORS[color], size=pt(size), line=dict(color="black", width=pt(0.6)))

    def add_spans(name):
        xa, ya, _ = axes_of[name]
        for m, label, color in [(mask_clicks, "Clicks", "tab:red"), (mask_preselected, "Non-Clicks", "tab:green")]:
            x, y = mask_polygons(m, t)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=rgba(MPL_COLORS[color], 0.3),
                                     line=dict(width=0), hoverinfo="skip", xaxis=xa, yaxis=ya,
                                     **item("spans", label, name == "sig")))

    # Top: signal and envelope
    xa, yh, yd = axes_of["sig"]
    add_spans("sig")
    for y, label, color, lw in [(signal, "Original", "gray", 1), (env, "Envelope", "blue", 1.5)]:
        fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=y, mode="lines", line=line(color, lw), xaxis=xa, yaxis=yd,
                                 hovertemplate="%{x:.4f} s, %{y:.4g}", **item("sig", label)))

    # Middle: isolation windows (axvspan, below the spans), envelope, medians and peaks
    xa, yh, yd = axes_of["env"]
    windows = [(lo, hi) for sp in spans for lo, hi, _ in sp["sides"]]
    for k, (lo, hi) in enumerate(windows):  # One trace per window: where they overlap the color accumulates
        fig.add_trace(go.Scatter(x=[t[lo], t[hi - 1], t[hi - 1], t[lo], t[lo]], y=[0, 0, 1, 1, 0], mode="lines",
                                 fill="toself", line=dict(width=0), hoverinfo="skip",
                                 fillcolor=rgba(MPL_COLORS["tab:purple"], 0.15), xaxis=xa, yaxis=yh,
                                 **item("env", "Isolation window", k == 0)))
    add_spans("env")
    fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=env_iso, mode="lines", line=line("blue", 1.5), xaxis=xa, yaxis=yd,
                             hovertemplate="%{x:.4f} s, %{y:.4g}", **item("env", "Envelope (in the windows of interest)")))
    for label, is_bg, lw, ls in [("Median (denominator)", True, 3, "-"), ("Median (other side)", False, 1.5, "--")]:
        xs, ys, hover = [], [], []
        for n, sp in enumerate(spans, 1):
            for k, (lo, hi, med) in enumerate(sp["sides"]):
                if (k == sp["k_bg"]) == is_bg:
                    xs += [t[lo], t[hi - 1], None]
                    ys += [med, med, None]
                    hover += [f"Span {n}: median {med:.4g} ({'denominator' if is_bg else 'other side'})"]*2 + [""]
        fig.add_trace(go.Scatter(x=xs or [None], y=ys or [None], mode="lines", line=line("tab:purple", lw, ls),
                                 hovertext=hover or None, hoverinfo="text", xaxis=xa, yaxis=yd, **item("env", label)))
    for is_click, color in [(True, "tab:red"), (False, "tab:green")]:
        group = [(n, sp) for n, sp in enumerate(spans, 1) if (sp["iso"] >= iso_min_thr) == is_click]
        fig.add_trace(go.Scatter(
            x=[t[sp["i"]] for _, sp in group] or [None], y=[env[sp["i"]] for _, sp in group] or [None],
            mode="markers", marker=marker(color), customdata=[[n, sp["iso"]] for n, sp in group] or None,
            xaxis=xa, yaxis=yd, showlegend=False, name="Peak",
            hovertemplate="Span %{customdata[0]}: peak %{y:.4g} at %{x:.4f} s<br>isolation %{customdata[1]:.2f}"
                          "<extra></extra>"))

    # Bottom: isolation per span (stems from 1), thresholds
    xa, yh, yd = axes_of["iso"]
    add_spans("iso")
    x_range = ranges["t"]
    fig.add_trace(go.Scatter(x=x_range, y=[1, 1], mode="lines", line=line("gray", 0.8), xaxis=xa, yaxis=yd,
                             hoverinfo="skip", showlegend=False))  # Base: peak = background
    for label, thr, ls in thresholds:
        fig.add_trace(go.Scatter(x=x_range, y=[thr, thr], mode="lines", line=line("tab:red", 2, ls), xaxis=xa,
                                 yaxis=yd, hoverinfo="skip", **item("iso", label)))
    for label, is_click, color in [(f"Isolation ≥ {iso_min_thr:g}", True, "tab:red"),
                                   (f"Isolation < {iso_min_thr:g}", False, "tab:green")]:
        group = [(n, sp) for n, sp in enumerate(spans, 1) if sp["iso"] > 0 and (sp["iso"] >= iso_min_thr) == is_click]
        fig.add_trace(go.Scatter(x=[v for _, sp in group for v in (t[sp["i"]], t[sp["i"]], None)] or [None],
                                 y=[v for _, sp in group for v in (1, sp["iso"], None)] or [None], mode="lines",
                                 line=line("orange", 1.5), xaxis=xa, yaxis=yd, hoverinfo="skip",
                                 **item("iso", label, False)))
        fig.add_trace(go.Scatter(
            x=[t[sp["i"]] for _, sp in group] or [None], y=[sp["iso"] for _, sp in group] or [None], mode="markers",
            marker=marker(color), customdata=[[n] for n, _ in group] or None, xaxis=xa, yaxis=yd,
            hovertemplate="Span %{customdata[0]}: isolation %{y:.2f}<extra></extra>", **item("iso", label)))
    no_sides = [(n, sp) for n, sp in enumerate(spans, 1) if sp["iso"] <= 0]
    if no_sides:  # Without valid sides: isolation 0 (x at 4 % of the height)
        fig.add_trace(go.Scatter(
            x=[t[sp["i"]] for _, sp in no_sides], y=[0.04]*len(no_sides), mode="markers",
            marker=dict(symbol="x-thin", size=pt(7), line=dict(color="black", width=pt(1))), xaxis=xa, yaxis=yh,
            customdata=[[n] for n, _ in no_sides], hovertemplate="Span %{customdata[0]}: no valid sides<extra></extra>",
            **item("iso", f"Isolation < {iso_min_thr:g}", False)))

    # --- Titles, labels, legend frames ---
    ann = lambda s, x, y, size, **kw: dict(text=s, x=X(x), y=Y(y), xref="paper", yref="paper", xanchor="center",
                                           yanchor="middle", showarrow=False, font=dict(size=pt(size)), **kw)
    annotations = [ann(suptitle, W/2 - crop_x, h_pad - crop_y + text_h(suptitle, 16, va="top")/2 - 0.1*pt(16), 16)]
    for name, (l, tp, r, b) in boxes.items():
        baseline = tp - pt(titles[name][1])
        annotations.append(ann(f"<b>{titles[name][0]}</b>", (l + r)/2,
                               baseline - (title_ext[name][1] - title_ext[name][2])/2 - 0.1*pt(14), 14))
        x = l - tick_space - ytick_labels(name, b - tp)[0] - pt(4) - text_h(ylabels[name], 12)/2
        annotations.append(ann(ylabels[name], x, (tp + b)/2, 12, textangle=-90))
    l, tp, r, b = boxes["iso"]
    annotations.append(ann("Time (s)", (l + r)/2, b + xlabel_stack("iso", r - l) - text_h("Time (s)", 12, va="top")/2,
                           12))
    shapes = []
    for key, (fl, ft) in frames.items():
        leg_w, leg_h, _ = leg_layout[key]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above",
                           fillcolor="rgba(255, 255, 255, 0.8)", line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(fl), X(fl + leg_w), Y(ft + leg_h), Y(ft), 0.2*em, width, height)))

    fig.update_layout(
        **layout, **legends, width=width, height=height, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )

    return fig


def plot_misalign(signal: np.ndarray, fs: float, details: dict, params: dict) -> go.Figure:
    """
    Interactive (plotly) version of plot_misalign: same layout (4 x 1 subplots, figsize 14 x (8 + 1.5 h_dev) in,
    layout="constrained"), sizes, fonts, colors and positions. The figure is cropped as the inline backend does
    (bbox_inches="tight"), so its size matches the matplotlib image.
    - Suptitle "Misalign calculation", centered in the figure
    - A: original signal and full-band envelope, with the candidate spans
    - B: envelope of each band (normalized, stacked), its peak in each span and the reference time t_ref of each
      span (median of the peaks; red: misalign <= MAX_MISALIGN, green: above)
    - C: misalign of each span (stems at t_ref) and the threshold if it falls inside the axes
    - D: deviation of the peak of each band from t_ref (ms) with the tolerance +-MAX_MISALIGN x FWHM of the band,
      one row per span (background: misalign under / over the threshold)
    - Legends: one plotly legend per item, at the place of the matplotlib item, inside a common frame (loc="best"
      computed as matplotlib does). Clicking a band (D1...D7) toggles its envelope and peaks (B) and its points
      (D); Clicks / Non-Clicks toggle the spans in A, B and C; Misalign <= / > toggle the stems (C) and the rows
      (D). The time axes (A, B, C) are linked. Hover: band, span, peak time, misalign, deviation and tolerance
    - WIDTH: width of the figure (px). Only the axes area gets narrower/wider (the labels, the fonts and the
      heights do not change). None -> as matplotlib
    - DPI: px per inch (matplotlib inline: 100). Every size (fonts, lines, pads) scales with it
    - params: algorithm constants (MAX_MISALIGN; see summary_plots). The time resolution (FWHM) of each band is
      computed with remove_noise.band_resolution, for the decomposition of details["bands"] (number of levels and
      split levels) and remove_noise.ENV_SMOOTH
    Limits: plotly draws the grid below the spans and the row backgrounds; the caps of the errorbar in the
    legend of D are drawn as shapes (they do not hide with the item)
    """
    width, dpi = WIDTH, DPI
    MAX_MISALIGN = params["MAX_MISALIGN"]
    PT = dpi/72  # px per point
    pt = lambda v: v*PT  # points -> px
    em = pt(10)  # Font size of the ticks and the legends (px)
    text = lambda s, size, **kw: mpl_text_extent(s, size, dpi, **kw)  # width, above and below the anchor (va)
    text_h = lambda s, size, **kw: sum(text(s, size, **kw)[1:])
    tick_space = pt(3.5) + pt(3.5)  # Tick length + pad
    pad = 0.1*dpi  # bbox_inches="tight" pad
    w_pad = h_pad = 3/72*dpi  # constrained_layout pads (rcParams figure.constrained_layout.w_pad/h_pad)

    # --- Data (as the matplotlib version) ---
    t = np.arange(len(signal)) / fs
    env = details["signal_envelope"]
    envelope = details["envelope"]  # bands x samples
    mask_prot = details["mask_protected"]
    n_bands = details["n_bands"]
    candidates = details["candidates"]
    n_spans = len(candidates)
    h_dev = max(1.2, 0.35*n_spans)  # Panel D grows with the number of spans
    mask_clicks = details["click_span"].astype(bool)
    mask_preselected = details["click_span_preselected"].astype(bool) & ~mask_clicks
    band_colors = [to_hex(c) for c in colormaps["turbo"](np.linspace(0, 0.9, n_bands))]
    suffix = {"full": "", "upper": "a", "lower": "b"}
    band_names = [f'D{b["level"]}{suffix[b["part"]]}' for b in details["bands"]]
    band_labels = [f'{name} ({b["f_low"]:g}-{b["f_high"]:g} Hz)' for name, b in zip(band_names, details["bands"])]

    GAP = 0.1  # Free fraction of each strip
    base = np.arange(n_bands)[::-1]  # D1 on top
    env_max = envelope.max(axis=1) + EPS
    env_norm = base[:, None] + (1 - GAP)*envelope / env_max[:, None]
    # Time resolution (FWHM, s) of each band, for the same decomposition as the algorithm (see band_resolution)
    n_levels = details["bands"][-1]["level"]
    split_levels = tuple(sorted({b["level"] for b in details["bands"] if b["part"] != "full"})) or None
    fwhm = rn.band_resolution(fs, n_levels, split_levels, round(rn.ENV_SMOOTH*fs))

    # Peak of each detected band in each span (as find_click_events), t_ref and deviations
    spans = []
    for n, c in enumerate(candidates, 1):
        a, b = round(c["start"]*fs), round(c["end"]*fs)
        bands = [k for k in range(n_bands) if mask_prot[k, a:b].any()]
        peaks = np.array([a + np.argmax(envelope[k, a:b]) for k in bands], dtype=int)
        mis = c["features"]["misalign"]
        sp = dict(n=n, c=c, bands=bands, peaks=peaks, mis=mis, ok=mis <= MAX_MISALIGN)
        if bands:
            sp["t_ref"] = np.median(peaks) / fs
            sp["off_ms"] = (peaks - np.median(peaks)) / fs * 1e3
            sp["tol_ms"] = MAX_MISALIGN * fwhm[bands] * 1e3
            sp["ys"] = n - 1 + np.linspace(-0.3, 0.3, len(bands)) if len(bands) > 1 else np.array([n - 1.0])
        spans.append(sp)
    stems = [sp for sp in spans if sp["bands"]]
    dev_lo = [sp["off_ms"] - sp["tol_ms"] for sp in stems]
    dev_hi = [sp["off_ms"] + sp["tol_ms"] for sp in stems]

    # Ranges (matplotlib autoscale: + 5 %; the stems start at 0)
    ranges = dict(t=mpl_range(t), sig=mpl_range(signal, env), env=mpl_range(env_norm),
                  mis=mpl_range([0], [sp["mis"] for sp in stems] or [np.nan]),
                  dev=mpl_range(*dev_lo, *dev_hi) if stems else [-0.055, 0.055], span=[n_spans - 0.5, -0.5])
    thr_line = bool(ranges["mis"][0] <= MAX_MISALIGN <= ranges["mis"][1])

    # Legends: matplotlib labels, plotly names (HTML) and ncol
    thr = f"{MAX_MISALIGN:g}"
    legend_specs = {
        "sig": (["Original", "Envelope"], ["Original", "Envelope"], 1),
        "spans": (["Clicks", "Non-Clicks"], ["Clicks", "Non-Clicks"], 2),
        "env": (["Envelope", "Max envelope for each span", "Level"] + band_labels,
                ["Envelope", "Max envelope for each span", "<b>Level</b>"] + band_labels, 1),
        "mis": ([f"threshold = {thr}"]*thr_line + [f"Misalign ≤ {thr}", f"Misalign > {thr}"],
                [f"threshold = {thr}"]*thr_line + [f"Misalign ≤ {thr}", f"Misalign > {thr}"], 1),
        "dev": ([f"Misalign ≤ {thr}", f"Misalign > {thr}", r"$t_{peak}-t_{ref}\ \pm\ $" + fr"{MAX_MISALIGN}$\cdot$FWHM"],
                [f"Misalign ≤ {thr}", f"Misalign > {thr}",
                 f"<i>t</i><sub><i>peak</i></sub>−<i>t</i><sub><i>ref</i></sub>±{MAX_MISALIGN}·FWHM"], 1),
    }
    leg_layout = {k: mpl_legend_layout(tuple(mpl), ncol, 10, dpi) for k, (mpl, _, ncol) in legend_specs.items()}

    names = ["sig", "env", "mis", "dev"]
    titles = {"sig": ("Original signal", 30), "env": ("Band envelopes", 6), "mis": ("Misalign per span", 6),
              "dev": ("Peak deviation per band", 6)}
    title_ext = {name: text(s, 14, weight="bold") for name, (s, _) in titles.items()}  # Baseline at pad pt
    ylabels = {"sig": "Amplitude (A.U.)", "env": "Amplitude (norm.)", "mis": "Misalign", "dev": "Span"}
    xlabels = {"mis": ("Time (s)", "Time (s)"),
               "dev": (r"$t_{peak} - t_{ref}$ (ms)", "<i>t</i><sub><i>peak</i></sub> − <i>t</i><sub><i>ref</i></sub> (ms)")}
    x_of = dict(sig="t", env="t", mis="t", dev="dev")
    suptitle = "Misalign calculation"
    sup_margin = text_h(suptitle, 16, va="top") + 2*h_pad  # constrained_layout: space of the suptitle
    span_labels = [f"#{i}" for i in range(1, n_spans + 1)]

    def xlabel_stack(name, length):
        """From the axes bottom to the bottom of the x tick labels (and of the x label, if any)"""
        labels = mpl_tick_labels(mpl_ticks(ranges[x_of[name]], length/PT, 3))
        h = tick_space + max(text_h(s, 10, va="top") for s in labels)
        return h + (pt(4) + text_h(xlabels[name][0], 12, va="top") if name in xlabels else 0)

    def ytick_labels(name, length):
        """Width of the y tick labels and how much they stick out of the axes (top, bottom)"""
        if name == "env":  # No y ticks
            return 0, 0, 0
        r = ranges["span" if name == "dev" else name]
        if name == "dev":
            ticks, labels = np.arange(n_spans), span_labels
        else:
            ticks = mpl_ticks(r, length/PT, 2)
            labels = mpl_tick_labels(ticks)
        pos = (r[1] - ticks)/(r[1] - r[0])*length  # px from the top. Labels: va="center_baseline"
        ext = np.array([text(s, 10, va="center_baseline") for s in labels])
        return ext[:, 0].max(), max(0, -(pos - ext[:, 1]).min()), max(0, (pos + ext[:, 2]).max() - length)

    def xtick_overhang(name, length):
        """How much the x tick labels stick out of the axes (left, right)"""
        r = ranges[x_of[name]]
        ticks = mpl_ticks(r, length/PT, 3)
        pos = (ticks - r[0])/(r[1] - r[0])*length
        half = np.array([text(s, 10)[0] for s in mpl_tick_labels(ticks)])/2
        return max(0, -(pos - half).min()), max(0, (pos + half).max() - length)

    def ylabel_space(name, h):
        """From the axes left to the left of the y label"""
        w_lab = ytick_labels(name, h)[0]
        return (tick_space + w_lab if name != "env" else 0) + pt(4) + text_h(ylabels[name], 12)

    def decorations(name, w, h):
        """Space (px) of the decorations of an axes of size w x h outside its box: left, right, top, bottom"""
        _, top, bottom = ytick_labels(name, h)
        o_l, o_r = xtick_overhang(name, w)
        top = max(top, pt(titles[name][1]) + title_ext[name][1])
        bottom = max(bottom, xlabel_stack(name, w))
        if name == "sig":  # Spans legend centered at (0.5, 1.15)
            top = max(top, 0.15*h + leg_layout["spans"][1]/2)
        if name == "env":  # Legend outside: upper left at (1.01, 1)
            o_r = max(o_r, 0.01*w + leg_layout["env"][0])
            bottom = max(bottom, leg_layout["env"][1] - h)
        return max(ylabel_space(name, h), o_l), o_r, top, bottom

    # --- constrained_layout (4 x 1, height_ratios [1, 2, 1, h_dev]) ---
    height_ratios = np.array([1, 2, 1, h_dev])
    H = (8 + 1.5*h_dev)*dpi

    def layout_for(W):
        h_cb = max(h_pad, 0.02/2/4*H)  # Padding (inside the grid: hspace)
        pads = 2*h_pad + 6*h_cb  # Above and below each row
        w, row_h = 0.8*W, height_ratios*0.15*H
        for _ in range(4):  # The decorations depend on the size of the axes (number of ticks)
            deco = [decorations(name, w, h) for name, h in zip(names, row_h)]
            L, R = max(d[0] for d in deco), max(d[1] for d in deco)
            T, B = np.array([d[2] for d in deco]), np.array([d[3] for d in deco])
            w = W - L - R - 2*w_pad
            row_h = height_ratios*(H - sup_margin - (T + B).sum() - pads)/height_ratios.sum()
        tops = sup_margin + np.cumsum(np.r_[0, (T + row_h + B + 2*h_cb)[:-1]]) + h_pad + T
        boxes = {name: (w_pad + L, tp, w_pad + L + w, tp + h) for name, tp, h in zip(names, tops, row_h)}
        # Bounding box of every element (bbox_inches="tight"): x0, x1, top, bottom
        sup_w = text(suptitle, 16)[0]
        ext = [(W/2 - sup_w/2, W/2 + sup_w/2, h_pad, h_pad + text_h(suptitle, 16, va="top"))]
        for name, (l, tp, r, b) in boxes.items():
            tw = title_ext[name][0]
            ext.append(((l + r - tw)/2, (l + r + tw)/2, tp - pt(titles[name][1]) - title_ext[name][1], tp))
            ext.append((l - ylabel_space(name, b - tp), r, tp, b))
            o_l, o_r = xtick_overhang(name, r - l)
            ext.append((l - o_l, r + o_r, b, b + xlabel_stack(name, r - l)))
        l, tp, r, b = boxes["env"]
        ext.append((l, r + 0.01*(r - l) + leg_layout["env"][0], tp, tp + leg_layout["env"][1]))
        x0, x1 = min(e[0] for e in ext) - pad, max(e[1] for e in ext) + pad
        y0, y1 = min(e[2] for e in ext) - pad, max(e[3] for e in ext) + pad
        return boxes, x0, y0, int(x1 - x0), int(y1 - y0)

    W = 14*dpi
    boxes, crop_x, crop_y, img_w, img_h = layout_for(W)
    for _ in range(4 if width is not None else 0):  # Only the axes area changes with width
        W += width - img_w
        boxes, crop_x, crop_y, img_w, img_h = layout_for(W)
    width, height = img_w, img_h
    boxes = {k: (l - crop_x, tp - crop_y, r - crop_x, b - crop_y) for k, (l, tp, r, b) in boxes.items()}
    X = lambda v: v/width  # px -> paper
    Y = lambda v: 1 - v/height

    # --- loc="best": what matplotlib counts (lines, markers, fill_between polygons, axhspan rectangles) ---
    def to_px(name, x, y=None, y_axes=None):
        """Data (or y in axes fraction) -> px"""
        l, tp, r, b = boxes[name]
        rx, ry = ranges[x_of[name]], ranges["span" if name == "dev" else name]
        px = l + (np.asarray(x, float) - rx[0])/(rx[1] - rx[0])*(r - l)
        py = (b - np.asarray(y_axes, float)*(b - tp) if y is None
              else b - (np.asarray(y, float) - ry[0])/(ry[1] - ry[0])*(b - tp))
        return px, py

    def span_paths(name):
        """Rectangles of the spans (fill_between with the axes transform), one path per run"""
        paths = []
        for m in (mask_clicks, mask_preselected):
            d = np.diff(np.r_[0, m.astype(np.int8), 0])
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
                paths.append(to_px(name, [t[s], t[e], t[e], t[s], t[s]], y_axes=[0, 0, 1, 1, 0]))
        return paths

    dev_points, dev_rects = [], []  # Errorbars: marker and caps (one Line2D each); rows: axhspan rectangles
    for sp in stems:
        for off, tol, y in zip(sp["off_ms"], sp["tol_ms"], sp["ys"]):
            dev_points += [to_px("dev", [v], [y]) for v in (off, off - tol, off + tol)]
        x0, x1 = boxes["dev"][0], boxes["dev"][2]
        _, (y0, y1) = to_px("dev", [0, 0], [sp["n"] - 1.5, sp["n"] - 0.5])
        dev_rects.append((x0, y1, x1, y0))
    paths = {
        "sig": [to_px("sig", t, signal), to_px("sig", t, env)] + span_paths("sig"),
        "mis": ([to_px("mis", [sp["t_ref"]], [sp["mis"]]) for sp in stems] + span_paths("mis")
                + ([(np.array(boxes["mis"][0:3:2]), to_px("mis", [0, 0], [MAX_MISALIGN]*2)[1])] if thr_line else [])),
        "dev": dev_points + [to_px("dev", [0, 0], y_axes=[0, 1])],
    }

    # Legend frames (left, top) and plotly legends (one per item; "Level" is a subtitle: annotation)
    frames, legend_of, legends, text_boxes = {}, {}, {}, {}
    for key, (mpl_labels, _, ncol) in legend_specs.items():
        leg_w, leg_h, item_boxes = leg_layout[key]
        l, tp, r, b = boxes["sig" if key == "spans" else key]
        if key == "spans":  # loc="center", bbox_to_anchor=(0.5, 1.15)
            frames[key] = ((l + r - leg_w)/2, tp - 0.15*(b - tp) - leg_h/2)
        elif key == "env":  # loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0
            frames[key] = (r + 0.01*(r - l), tp)
        else:
            frames[key] = mpl_best_legend(boxes[key], (leg_w, leg_h), paths[key], em,
                                          rects=dev_rects if key == "dev" else ())
        fl, ft = frames[key]
        for k, (x0, _, top, bottom) in enumerate(item_boxes):
            text_boxes[(key, k)] = (fl + x0, ft + (top + bottom)/2)
            if key == "env" and k == 2:
                continue
            lid = "legend" if not legends else f"legend{len(legends) + 1}"
            legend_of[(key, k)] = lid
            # Text at the matplotlib position (plotly: symbol at [5, 35] px, text at 40 px from the legend x).
            # Raised 0.1 font sizes: Arial (plotly) sits lower than DejaVu Sans
            legends[lid] = dict(x=X(fl + x0 - 40), y=Y(ft + (top + bottom)/2 - 0.1*em), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=em), groupclick="togglegroup", itemwidth=30)
            if key == "env" and k < 2 or key == "dev" and k == 2:  # Descriptive items: not clickable
                legends[lid].update(itemclick=False, itemdoubleclick=False)

    def item(key, k, show=True):
        """Legend arguments of the item k of the legend key"""
        return dict(name=legend_specs[key][1][k], legend=legend_of[(key, k)], legendgroup=f"{key}{k}",
                    showlegend=show)

    band_item = lambda k, show=False: item("env", 3 + k, show)
    k_ok, k_bad = int(thr_line), int(thr_line) + 1  # Items Misalign <= / > of C

    # --- Axes ---
    axis_style = dict(showline=True, mirror=True, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none", tickformatstops=MPL_TICKFORMAT)

    def ticks_of(r, length, label_factor):
        """matplotlib ticks: nticks (plotly keeps its automatic ticks when zooming) or, for steps of 2.5 x 10^k
        (plotly only uses 1, 2, 5), fixed step with the decimals of matplotlib"""
        ticks = mpl_ticks(r, length/PT, label_factor)
        step = ticks[1] - ticks[0]
        if np.isclose(step/10**np.floor(np.log10(step)), 2.5):
            dec = len(mpl_tick_labels(ticks)[0].split(".")[1]) if "." in mpl_tick_labels(ticks)[0] else 0
            return dict(tickmode="linear", tick0=ticks[0], dtick=step, tickformat=f".{dec}f", tickformatstops=[])
        return dict(nticks=mpl_nticks(r, length/PT, label_factor))

    layout = {}
    axes_of = {}  # x, hidden y [0, 1] (spans), data y
    for n, name in enumerate(["sig", "env", "mis"]):
        l, tp, r, b = boxes[name]
        xa, x_ref = f"xaxis{n + 1 if n else ''}", f"x{n + 1 if n else ''}"
        yh, yh_ref, yd, yd_ref = f"yaxis{2*n + 1 if n else ''}", f"y{2*n + 1 if n else ''}", f"yaxis{2*n + 2}", f"y{2*n + 2}"
        axes_of[name] = (x_ref, yh_ref, yd_ref)
        layout[xa] = dict(domain=[X(l), X(r)], anchor=yh_ref, range=ranges["t"],
                          **{**axis_style, **ticks_of(ranges["t"], r - l, 3)}, ticklabelstandoff=round(-2.2*PT),
                          **({"matches": "x"} if n else {}))
        layout[yh] = dict(domain=[Y(b), Y(tp)], anchor=x_ref, range=[0, 1], fixedrange=True, visible=False)
        layout[yd] = dict(overlaying=yh_ref, anchor=x_ref, side="left", range=ranges[name],
                          **{**axis_style, **ticks_of(ranges[name], b - tp, 2)})
    layout["yaxis4"].update(showgrid=False, ticks="", showticklabels=False)  # B: no y ticks, grid only in x
    l, tp, r, b = boxes["dev"]
    axes_of["dev"] = ("x4", None, "y7")
    layout["xaxis4"] = dict(domain=[X(l), X(r)], anchor="y7", range=ranges["dev"],
                            **{**axis_style, "gridcolor": rgba("#b0b0b0", 0.3), **ticks_of(ranges["dev"], r - l, 3)},
                            ticklabelstandoff=round(-2.2*PT))
    layout["yaxis7"] = dict(domain=[Y(b), Y(tp)], anchor="x4", range=ranges["span"],
                            **{**axis_style, "showgrid": False}, tickmode="array", tickvals=list(range(n_spans)),
                            ticktext=span_labels)

    fig = go.Figure()
    line = lambda color, lw, dash=None: dict(color=MPL_COLORS.get(color, color), width=pt(lw), dash=dash)
    dashed = lambda lw: mpl_dashes(DASH_PATTERNS["--"], lw, PT)
    marker = lambda color, size=6, mew=0.6: dict(color=MPL_COLORS.get(color, color), size=pt(size),
                                                 line=dict(color="black", width=pt(mew)))

    def add_spans(name):
        xa, ya, _ = axes_of[name]
        for k, (m, color) in enumerate([(mask_clicks, "tab:red"), (mask_preselected, "tab:green")]):
            x, y = mask_polygons(m, t)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=rgba(MPL_COLORS[color], 0.3),
                                     line=dict(width=0), hoverinfo="skip", xaxis=xa, yaxis=ya,
                                     **item("spans", k, name == "sig")))

    # A: signal and envelope
    xa, _, yd = axes_of["sig"]
    add_spans("sig")
    for k, (y, color, lw) in enumerate([(signal, "gray", 1), (env, "blue", 1.5)]):
        fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=y, mode="lines", line=line(color, lw), xaxis=xa, yaxis=yd,
                                 hovertemplate="%{x:.4f} s, %{y:.4g}", **item("sig", k)))

    # B: band envelopes, t_ref of each span, peaks of each band
    xa, _, yd = axes_of["env"]
    add_spans("env")
    for k in range(n_bands):
        fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=env_norm[k], mode="lines", line=line(band_colors[k], 1.5),
                                 customdata=envelope[k], xaxis=xa, yaxis=yd, **band_item(k, True),
                                 hovertemplate=f"{band_names[k]}: %{{x:.4f}} s, %{{customdata:.4g}}<extra></extra>"))
    for ok, color in [(True, "red"), (False, "green")]:
        group = [sp for sp in stems if sp["ok"] == ok]
        fig.add_trace(go.Scatter(x=[v for sp in group for v in (sp["t_ref"],)*2 + (None,)] or [None],
                                 y=[v for _ in group for v in ranges["env"] + [None]] or [None], mode="lines",
                                 line=line(color, 1.5, dashed(1.5)), xaxis=xa, yaxis=yd, hoverinfo="skip",
                                 **item("mis", k_ok if ok else k_bad, False)))
    for k in range(n_bands):
        pts = [(sp, p) for sp in stems for kk, p in zip(sp["bands"], sp["peaks"]) if kk == k]
        if pts:
            fig.add_trace(go.Scatter(
                x=[t[p] for _, p in pts], y=[env_norm[k, p] for _, p in pts], mode="markers",
                marker=marker(band_colors[k], 5, 1), customdata=[[sp["n"], envelope[k, p]] for sp, p in pts],
                xaxis=xa, yaxis=yd, **band_item(k),
                hovertemplate=f"Span #%{{customdata[0]}}, {band_names[k]}: peak at %{{x:.4f}} s "
                              f"(%{{customdata[1]:.4g}})<extra></extra>"))
    # Descriptive items of the legend (Envelope, Max envelope for each span)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=line("black", 1.5), xaxis=xa, yaxis=yd,
                             hoverinfo="skip", **item("env", 0)))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=marker("black", 5, 1), xaxis=xa, yaxis=yd,
                             hoverinfo="skip", **item("env", 1)))

    # C: misalign per span (stems from 0 at t_ref), threshold
    xa, _, yd = axes_of["mis"]
    add_spans("mis")
    if thr_line:
        fig.add_trace(go.Scatter(x=ranges["t"], y=[MAX_MISALIGN]*2, mode="lines", line=line("tab:red", 2, dashed(2)),
                                 xaxis=xa, yaxis=yd, hoverinfo="skip", **item("mis", 0)))
    for ok, color in [(True, "tab:red"), (False, "tab:green")]:
        group = [sp for sp in stems if sp["ok"] == ok]
        k = k_ok if ok else k_bad
        fig.add_trace(go.Scatter(x=[v for sp in group for v in (sp["t_ref"],)*2 + (None,)] or [None],
                                 y=[v for sp in group for v in (0, sp["mis"], None)] or [None], mode="lines",
                                 line=line("sienna", 1.5), xaxis=xa, yaxis=yd, hoverinfo="skip",
                                 **item("mis", k, False)))
        fig.add_trace(go.Scatter(
            x=[sp["t_ref"] for sp in group] or [None], y=[sp["mis"] for sp in group] or [None], mode="markers",
            marker=marker(color), customdata=[[sp["n"], len(sp["bands"])] for sp in group] or None, xaxis=xa,
            yaxis=yd, **item("mis", k),
            hovertemplate="Span #%{customdata[0]} (%{customdata[1]} bands): t<sub>ref</sub> = %{x:.4f} s<br>"
                          "misalign = %{y:.3f}<extra></extra>"))

    # D: row backgrounds (axhspan, edge lw 1), t = t_ref, deviation of the peak of each band +- tolerance
    xa, _, yd = axes_of["dev"]
    x_rng = ranges["dev"]
    for k, (ok, color) in enumerate([(True, "tab:blue"), (False, "tab:gray")]):
        xs, ys = [], []
        for sp in stems:
            if sp["ok"] == ok:
                y0, y1 = sp["n"] - 1.5, sp["n"] - 0.5
                xs += [x_rng[0], x_rng[1], x_rng[1], x_rng[0], x_rng[0], None]
                ys += [y0, y0, y1, y1, y0, None]
        fig.add_trace(go.Scatter(x=xs or [None], y=ys or [None], mode="lines", fill="toself",
                                 fillcolor=rgba(MPL_COLORS[color], 0.12), line=dict(color=rgba(MPL_COLORS[color], 0.12),
                                 width=pt(1)), hoverinfo="skip", xaxis=xa, yaxis=yd, **item("dev", k, False)))
        # Legend item (patch alpha 0.3, as fill_between([], []))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", fill="toself",
                                 fillcolor=rgba(MPL_COLORS[color], 0.3), line=dict(width=0), xaxis=xa, yaxis=yd,
                                 hoverinfo="skip", **item("dev", k)))
    fig.add_trace(go.Scatter(x=[0, 0], y=ranges["span"], mode="lines", line=line("black", 1), xaxis=xa, yaxis=yd,
                             hoverinfo="skip", showlegend=False))
    for k in range(n_bands):
        pts = [(sp, off, tol, y) for sp in stems
               for kk, off, tol, y in zip(sp["bands"], sp["off_ms"], sp["tol_ms"], sp["ys"]) if kk == k]
        if pts:
            fig.add_trace(go.Scatter(
                x=[off for _, off, _, _ in pts], y=[y for *_, y in pts], mode="markers",
                marker=marker(band_colors[k], 5), xaxis=xa, yaxis=yd, **band_item(k),
                error_x=dict(type="data", array=[tol for _, _, tol, _ in pts], color=band_colors[k],
                             thickness=pt(1.5), width=pt(2)),
                customdata=[[sp["n"], tol] for sp, _, tol, _ in pts],
                hovertemplate=f"Span #%{{customdata[0]}}, {band_names[k]}: t<sub>peak</sub> - t<sub>ref</sub> = "
                              f"%{{x:.2f}} ± %{{customdata[1]:.2f}} ms<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines+markers", line=line("black", 1.5),
                             marker=marker("black", 5), xaxis=xa, yaxis=yd, hoverinfo="skip", **item("dev", 2)))

    # --- Titles, labels, legend frames ---
    ann = lambda s, x, y, size, xanchor="center", **kw: dict(text=s, x=X(x), y=Y(y), xref="paper", yref="paper",
                                                             xanchor=xanchor, yanchor="middle", showarrow=False,
                                                             font=dict(size=pt(size)), **kw)
    annotations = [ann(suptitle, W/2 - crop_x, h_pad - crop_y + text_h(suptitle, 16, va="top")/2 - 0.1*pt(16), 16)]
    for name, (l, tp, r, b) in boxes.items():
        baseline = tp - pt(titles[name][1])
        annotations.append(ann(f"<b>{titles[name][0]}</b>", (l + r)/2,
                               baseline - (title_ext[name][1] - title_ext[name][2])/2 - 0.1*pt(14), 14))
        x = l - ylabel_space(name, b - tp) + text_h(ylabels[name], 12)/2
        annotations.append(ann(ylabels[name], x, (tp + b)/2, 12, textangle=-90))
        if name in xlabels:
            mpl_label, html = xlabels[name]
            annotations.append(ann(html, (l + r)/2, b + xlabel_stack(name, r - l) - text_h(mpl_label, 12, va="top")/2,
                                   12))
    x, y = text_boxes[("env", 2)]
    annotations.append(ann("<b>Level</b>", x, y - 0.1*em, 10, xanchor="left"))
    shapes = []
    for key, (fl, ft) in frames.items():
        leg_w, leg_h, _ = leg_layout[key]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above",
                           fillcolor="rgba(255, 255, 255, 0.8)", line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(fl), X(fl + leg_w), Y(ft + leg_h), Y(ft), 0.2*em, width, height)))
    x, y = text_boxes[("dev", 2)]  # Caps of the errorbar of the legend (ends of the plotly symbol)
    for xc in (x - 35, x - 5):
        shapes.append(dict(type="line", xref="paper", yref="paper", x0=X(xc), x1=X(xc), y0=Y(y - 0.1*em - pt(2)),
                           y1=Y(y - 0.1*em + pt(2)), line=dict(color="black", width=pt(1)), layer="above"))

    fig.update_layout(
        **layout, **legends, width=width, height=height, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )

    return fig


def plot_click_criteria(fs: float, details: dict) -> go.Figure:
    """
    Classification of the click candidates (time spans with a detection in some band after the
    long-event protection, see find_click_events):
    - Signal with the candidate spans shaded (red: click -> attenuated, blue: not a click -> kept) and
      numbered. Hovering over the number shows the conditions of each rule
    - Table with, for each span and rule, every condition (✓/✗, value and threshold). Green: rule met.
      A span is a click if it meets some rule (broadband, impulsive, transient or aligned) and, if it touches
      a border of the fragment, also the border condition (z_max > Z_EDGE)
    """
    original = details["original"]
    clean = original - details["removed_per_level"].sum(axis=0)
    candidates = details["candidates"]
    t = np.arange(len(original)) / fs
    rules = {"broadband": "1. Broadband click", "impulsive": "2. Impulsive click",
             "transient": "3. Isolated transient", "aligned": "4. Aligned transient", "border": "At border"}

    # Text of each rule (one condition per line)
    def rule_text(c: dict, rule: str, sep: str) -> str:
        if not c["conditions"][rule]:
            return "not at border"
        return sep.join(format_condition(*cond) for cond in c["conditions"][rule])

    # Heights (px): signal + table (one row per candidate, up to 5 lines each). The table gets exactly the
    # height it needs (otherwise plotly adds a scroll bar): fixed margins, spacing relative to the plot area
    sig_h, row_h, head_h, gap = 330, 95, 45, 70
    margin_t, margin_b = 110, 30
    table_h = head_h + row_h*max(len(candidates), 1)
    plot_h = sig_h + gap + table_h
    height = plot_h + margin_t + margin_b
    fig = make_subplots(rows=2, cols=1, specs=[[{"type": "xy"}], [{"type": "table"}]],
                        row_heights=[sig_h, table_h], vertical_spacing=gap/plot_h)

    fig.add_trace(go.Scatter(x=t, y=original, name="Original (>15 Hz)", line=dict(color=GRAY, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=clean, name="Filtered", line=dict(color=GREEN, width=1)), row=1, col=1)

    # Candidate spans: shaded and numbered (hover -> conditions)
    y_label = 1.1*np.abs(original).max()
    for is_click, name, color, fill in [(True, "Click (attenuated)", RED, CLICK_FILL),
                                        (False, "Not a click (kept)", "rgb(0, 90, 200)", KEPT_FILL)]:
        group = [(k, c) for k, c in enumerate(candidates, 1) if c["is_click"] == is_click]
        for _, c in group:
            fig.add_vrect(x0=c["start"], x1=c["end"], fillcolor=fill, line_width=0, layer="below", row=1, col=1)
        hover = [f"<b>#{k}</b> {c['start']:.3f}-{c['end']:.3f} s: {'CLICK' if c['is_click'] else 'not a click'}<br>"
                 + "<br>".join(f"<b>{rules[r]}</b> {'✓' if c['passed'][r] else '✗'}<br>  " + rule_text(c, r, "<br>  ")
                               for r in rules)
                 for k, c in group]
        fig.add_trace(go.Scatter(
            x=[(c["start"] + c["end"]) / 2 for _, c in group], y=[y_label]*len(group),
            mode="markers+text", text=[str(k) for k, _ in group], textposition="top center",
            marker=dict(color=color, size=8, symbol="triangle-down"), name=name,
            hovertext=hover, hoverinfo="text", hoverlabel=dict(align="left")
        ), row=1, col=1)

    # Table: one row per candidate, one column per rule
    header = ["#", "Span (s)", "Result"] + list(rules.values())
    if candidates:
        cols = [[str(k) for k in range(1, len(candidates) + 1)],
                [f"{c['start']:.3f}-{c['end']:.3f}" for c in candidates],
                ["<b>CLICK</b>" if c["is_click"] else "kept" for c in candidates]]
        cols += [[rule_text(c, r, "<br>") for c in candidates] for r in rules]
        fills = [[FAIL_FILL]*len(candidates)]*2 + [[PASS_FILL if c["is_click"] else FAIL_FILL for c in candidates]]
        fills += [[PASS_FILL if c["passed"][r] else FAIL_FILL for c in candidates] for r in rules]
    else:
        cols = [["-"], ["No candidates"], ["-"]] + [["-"]]*len(rules)
        fills = [[FAIL_FILL]]*len(header)
    # Column widths proportional to the longest line of each column (fixed widths cut the text that does not
    # fit): ~6.2 px per character (font 11) + cell padding
    def text_px(texts: list, px_char: float) -> float:
        lines = [line.replace("<b>", "").replace("</b>", "") for s in texts for line in s.split("<br>")]
        return px_char*max(len(line) for line in lines) + 12

    col_w = [max(text_px([h], 7.5), text_px(col, 6.8)) for h, col in zip(header, cols)]
    col_w[-1] += 25  # "At border": its short lines wrap earlier than the estimate
    margin_l, margin_r = 70, 30
    fig.add_trace(go.Table(
        columnwidth=col_w,
        header=dict(values=[f"<b>{h}</b>" for h in header], fill_color="rgb(230, 230, 230)", align="left",
                    font=dict(size=12)),
        cells=dict(values=cols, fill_color=fills, align="left", font=dict(size=11), height=22)
    ), row=2, col=1)

    n_clicks = sum(c["is_click"] for c in candidates)
    fig.update_yaxes(title_text="Amplitude", range=[-1.25*y_label/1.1, 1.3*y_label], row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", range=[t[0], t[-1]], row=1, col=1)
    fig.update_layout(
        height=height, width=1200, margin=dict(t=margin_t, b=margin_b, l=margin_l, r=margin_r),
        title=dict(text=f"Click classification: {n_clicks} of {len(candidates)} candidates are clicks"
                        "<br><sup>A span is a click if it meets all the conditions of some rule (green) and, "
                        "if it touches a border, z_max > Z_EDGE. Widths in ms</sup>"),
        legend=dict(orientation="h", x=1, y=1, xanchor="right", yanchor="bottom"),
        template="plotly_white"
    )

    return fig


def plot_noise_removal(fs: float, details: dict) -> go.Figure:
    """
    Summary figure in 3 columns: original (> 15 Hz), filtered signal and residual (removed component).
    Each column: signal on top and spectrogram below. The spectrograms of the original and the filtered
    signal are normalized to their own maximum; the one of the residual, to the maximum of the original
    (so its colors show how much was removed relative to the original).

    Parameters
    ----------
    fs: float
        Sample frequency
    details: dict
        Intermediate results of remove_noise: "original" (> 15 Hz) and "clean" (filtered signal)

    The color scale of the spectrograms is SPEC_SCALE: "linear" (normalized power, 0-1) or "log" (dB
    relative to the maximum, -SPEC_DB_RANGE-0 dB)

    Returns
    -------
    fig: go.Figure
        Figure
    """
    original, clean = details["original"], details["clean"]
    scale, db_range = SPEC_SCALE, SPEC_DB_RANGE
    if scale not in ("linear", "log"):
        raise ValueError(f"scale must be 'linear' or 'log', not {scale!r}")
    t = np.arange(len(original)) / fs
    residual = original - clean

    stft = ShortTimeFFT(hann(400), 8, fs, mfft=2048, scale_to="psd")
    t_s = stft.t(len(original))
    f_mask = (stft.f >= FMIN) & (stft.f <= FMAX)
    f = stft.f[f_mask]
    S0 = stft.spectrogram(original)[f_mask]
    S1 = stft.spectrogram(clean)[f_mask]
    S2 = stft.spectrogram(residual)[f_mask]
    P = [S0/S0.max(), S1/S1.max(), S2/S0.max()]  # Linear power, normalized (residual: to the original max)
    if scale == "log":
        P = [10*np.log10(np.maximum(p, 10**(-db_range/10))) for p in P]  # dB re max, floor at -db_range
        cmin, cmax, cbar_title = -db_range, 0, "PSD (dB, norm.)"
    else:
        cmin, cmax, cbar_title = 0, 1, "PSD (linear, norm.)"

    # Sizes (px). The width is fixed; the height is computed so that the spectrograms are square
    width, h_spacing = 1450, 0.05
    margin_l, margin_r, margin_t, margin_b = 80, 150, 120, 70
    col_w = (width - margin_l - margin_r) * (1 - 2*h_spacing) / 3  # Width of each column
    spec_h, sig_h, gap = col_w, 0.55*col_w, 75  # gap: x ticks of the signals + titles of the spectrograms
    plot_h = sig_h + gap + spec_h

    columns = [("Original (>15 Hz)", original, GRAY), ("Filtered", clean, GREEN), ("Residual", residual, RED)]
    fig = make_subplots(
        rows=2, cols=3, row_heights=[sig_h, spec_h], vertical_spacing=gap/plot_h, horizontal_spacing=h_spacing,
        subplot_titles=[f"<b>{name}</b>" for name, _, _ in columns]
                       + [f"<b>Spectrogram ({name.split(' ')[0].lower()})</b>" for name, _, _ in columns]
    )
    fig.update_annotations(font=dict(size=15))  # Subplot titles (before adding other annotations)
    for col, ((name, x, color), p) in enumerate(zip(columns, P), 1):
        fig.add_trace(go.Scatter(x=t, y=x, name=name, line=dict(color=color, width=1)), row=1, col=col)
        fig.add_trace(go.Heatmap(x=t_s, y=f, z=p, coloraxis="coloraxis", name=name), row=2, col=col)
        # All the time axes linked (zoom in one -> all); same amplitude range in the 3 signals
        fig.update_xaxes(range=[t[0], t[-1]], matches="x", row=1, col=col)
        fig.update_xaxes(matches="x", row=2, col=col)
        fig.update_yaxes(matches="y", row=1, col=col)
        fig.update_yaxes(range=[FMIN, FMAX], matches="y4", row=2, col=col)

    # Amplitude range fitted to the 3 signals (shared), with a small margin
    y_min = min(original.min(), clean.min(), residual.min())
    y_max = max(original.max(), clean.max(), residual.max())
    pad = 0.05*(y_max - y_min)
    fig.update_yaxes(title_text="Amplitude (A.U.)", range=[y_min - pad, y_max + pad], row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_font=dict(size=14))
    fig.update_yaxes(title_font=dict(size=14))
    # Single time label for the whole figure (as matplotlib supxlabel)
    fig.add_annotation(text="Time (s)", x=0.5, y=0, xref="paper", yref="paper", xanchor="center", yanchor="top",
                       yshift=-35, showarrow=False, font=dict(size=15))
    fig.update_layout(
        height=round(plot_h + margin_t + margin_b), width=width,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        title=dict(text=f"<b>Noise removal</b> - spectrograms in {'dB' if scale == 'log' else 'linear'} scale",
                   x=0.5, xanchor="center", font=dict(size=22)),
        coloraxis=dict(
            colorscale=SPEC_COLORSCALE,
            cmin=cmin, cmax=cmax,
            colorbar=dict(  # Next to the spectrograms only, same height
                title=dict(text=cbar_title, side="right", font=dict(size=14)),
                y=0, yanchor="bottom", len=spec_h/plot_h, thickness=20, ticklabelposition="outside",
                ticks="outside", ticklen=5, xpad=15, ypad=0
            )
        ),
        showlegend=False,
        template="plotly_white"
    )

    return fig


def plot_frequency_response(fs: float, details: dict) -> go.Figure:
    """
    Interactive (plotly) version of plot_frequency_response: same layout (gridspec 4 x 2, figsize 14 x 11 in,
    layout="constrained"), sizes, fonts, colors and positions. The figure is cropped as the inline backend does
    (bbox_inches="tight"), so its size matches the matplotlib image.
    - A: original and filtered signals with the candidate spans; B: residual (original - filtered), same y axis
    - C: PSD (Welch, nperseg 512) of the three signals; D: attenuation (filtered / original, dB)
    - E: removed energy per band (% of the energy of each band)
    - Layout: the margins of constrained_layout are computed from the decorations of each axes (tick labels,
      labels, titles, legends, bar labels), with the same pads and rules (A and B span the 2 columns, E spans the
      2 bottom rows)
    - Legends (A, B, C, D and Clicks / Non-Clicks above A): one plotly legend per item, at the place of the
      matplotlib item (loc="best" computed as matplotlib does). Clicking an item toggles that series; Clicks /
      Non-Clicks toggle the spans in A and B. The time axes
      (A, B) and the frequency axes (C, D) are linked. Hover: values of the curves and of the bars
    - WIDTH: width of the figure (px). Only the axes area gets narrower/wider (the labels, the fonts and the
      heights do not change). None -> as matplotlib
    - DPI: px per inch (matplotlib inline: 100). Every size (fonts, lines, pads) scales with it
    Limits: plotly draws the grid below the spans; the x ticks of C and D (step 250, which plotly does not use)
    are fixed (they do not change when zooming)
    """
    width, dpi = WIDTH, DPI
    PT = dpi/72  # px per point
    pt = lambda v: v*PT  # points -> px
    em = pt(10)  # Font size of the ticks and the legends (px)
    text = lambda s, size, **kw: mpl_text_extent(s, size, dpi, **kw)  # width, above and below the anchor (va)
    text_h = lambda s, size, **kw: sum(text(s, size, **kw)[1:])
    tick_space = pt(3.5) + pt(3.5)  # Tick length + pad
    pad = 0.1*dpi  # bbox_inches="tight" pad
    w_pad = h_pad = 3/72*dpi  # constrained_layout pads (rcParams figure.constrained_layout.w_pad/h_pad)

    # --- Data (as the matplotlib version) ---
    original = details["original"]  # Reconstruction without the approximation (< 15.6 Hz): fair reference
    clean = details["clean"]
    removed = original - clean
    t = np.arange(len(original)) / fs
    mask_clicks = details["click_span"].astype(bool)
    mask_preselected = details["click_span_preselected"].astype(bool) & ~mask_clicks

    kw = dict(fs=fs, nperseg=512, window="hann")  # Same PSD parameters for the three signals
    f, P_orig = welch(original, **kw)
    _, P_clean = welch(clean, **kw)
    _, P_rem = welch(removed, **kw)
    psd = [10*np.log10(P + EPS) for P in (P_orig, P_clean, P_rem)]
    att_db = 10*np.log10((P_clean + EPS) / (P_orig + EPS))

    band_energy = (details["detail"]**2).sum(axis=1)  # Removed energy per band (% of the energy of each band)
    removed_energy = (details["removed_per_level"]**2).sum(axis=1)
    pct = 100*removed_energy / (band_energy + EPS)
    n_b = len(pct)
    suffix = {"full": "", "upper": "a", "lower": "b"}
    band_names = [f'D{b["level"]}{suffix[b["part"]]}' for b in details["bands"]]
    bar_ticks = [f'{n}:\n ({b["f_low"]:g}-{b["f_high"]:g} Hz)' for n, b in zip(band_names, details["bands"])]
    bar_labels = [f"{v:.1f} %" for v in pct]

    # Ranges (matplotlib autoscale: + 5 %; C, D: xlim (0, fs/2); E: xlim (0, max(1, 1.3 max)), inverted y)
    ranges = dict(t=mpl_range(t), amp=mpl_range(original, clean, removed), f=[0, fs/2], psd=mpl_range(*psd),
                  att=mpl_range(att_db, [0]), pct=[0, max(1, 1.3*pct.max())], band=mpl_range([-0.4], [n_b - 0.6])[::-1])

    # Legends: labels (= plotly names), ncol, font size
    legend_specs = {
        "sig": (["Original", "Filtered"], 1, 10),
        "spans": (["Clicks", "Non-Clicks"], 2, 10),
        "rem": (["Residual"], 1, 10),
        "psd": (["Original", "Filtered", "Residual"], 1, 10),
        "att": (["Attenuation"], 1, 10),
    }
    leg_layout = {k: mpl_legend_layout(tuple(labels), ncol, size, dpi) for k, (labels, ncol, size) in legend_specs.items()}

    names = ["sig", "rem", "psd", "att", "bar"]
    titles = {"sig": ("Original vs Filtered signal", 24), "rem": ("Residual", 6), "psd": ("Power spectral density", 6),
              "att": ("Attenuation", 6), "bar": ("Removed energy per band", 6)}
    title_ext = {name: text(s, 14, weight="bold") for name, (s, _) in titles.items()}  # Baseline at pad pt
    ylabels = {"sig": "Amplitude (A.U.)", "rem": "Amplitude (A.U.)", "psd": "PSD (dB/Hz)", "att": "Clean / Original (dB)"}
    xlabels = {"rem": "Time (s)", "att": "Frequency (Hz)", "bar": "% of the band energy"}
    x_of = dict(sig="t", rem="t", psd="f", att="f", bar="pct")
    y_of = dict(sig="amp", rem="amp", psd="psd", att="att", bar="band")
    suptitle = "Effect in frequency"
    sup_margin = text_h(suptitle, 16, va="top") + 2*h_pad  # constrained_layout: space of the suptitle

    def xlabel_stack(name, length):
        """From the axes bottom to the bottom of the x tick labels (and of the x label, if any)"""
        labels = mpl_tick_labels(mpl_ticks(ranges[x_of[name]], length/PT, 3))
        h = tick_space + max(text_h(s, 10, va="top") for s in labels)
        return h + (pt(4) + text_h(xlabels[name], 12, va="top") if name in xlabels else 0)

    def ytick_labels(name, length):
        """Width of the y tick labels and how much they stick out of the axes (top, bottom). A and B share the
        y locator (that of A, so the ticks of B are computed with the height of A)"""
        r = ranges[y_of[name]]
        if name == "bar":
            ticks, labels = np.arange(n_b), bar_ticks
        else:
            ticks = mpl_ticks(r, (length if name != "rem" else sizes["sig"][1])/PT, 2)
            labels = mpl_tick_labels(ticks)
        pos = (r[1] - ticks)/(r[1] - r[0])*length  # px from the top. Labels: va="center_baseline"
        ext = np.array([text(s, 10, va="center_baseline") for s in labels])
        return ext[:, 0].max(), max(0, -(pos - ext[:, 1]).min()), max(0, (pos + ext[:, 2]).max() - length)

    def xtick_overhang(name, length):
        """How much the x tick labels stick out of the axes (left, right)"""
        r = ranges[x_of[name]]
        ticks = mpl_ticks(r, length/PT, 3)
        pos = (ticks - r[0])/(r[1] - r[0])*length
        half = np.array([text(s, 10)[0] for s in mpl_tick_labels(ticks)])/2
        return max(0, -(pos - half).min()), max(0, (pos + half).max() - length)

    def ylabel_space(name, h):
        """From the axes left to the left of the y label (E: of the tick labels)"""
        return tick_space + ytick_labels(name, h)[0] + (pt(4) + text_h(ylabels[name], 12) if name in ylabels else 0)

    def decorations(name, w, h):
        """Space (px) of the decorations of an axes of size w x h outside its box: left, right, top, bottom"""
        _, top, bottom = ytick_labels(name, h)
        o_l, o_r = xtick_overhang(name, w)
        top = max(top, pt(titles[name][1]) + title_ext[name][1])
        bottom = max(bottom, xlabel_stack(name, w))
        if name == "sig":  # Spans legend: loc="lower center" at (0.5, 1), borderaxespad 0.2
            top = max(top, 0.2*em + leg_layout["spans"][1])
        if name == "bar":  # Bar labels (padding 3 pt)
            r = ranges["pct"]
            ends = pct/(r[1] - r[0])*w + pt(3) + np.array([text(s, 10)[0] for s in bar_labels])
            o_r = max(o_r, ends.max() - w)
        return max(ylabel_space(name, h), o_l), o_r, top, bottom

    # --- constrained_layout (rows: sig, rem, [psd, bar], [att, bar]; columns: [sig, rem, psd, att], [sig, rem, bar]) ---
    width_ratios, height_ratios = np.array([2.2, 1]), np.array([1, 0.7, 1.4, 0.8])
    H = 11*dpi
    sizes = {}

    def solve(W, col_w, row_h):
        """Axes boxes (left, top, right, bottom; px from the top-left of the figure) for the sizes of the previous
        iteration (the decorations depend on the size of the axes)"""
        sizes.update(sig=(col_w.sum() + gap[0], row_h[0]), rem=(col_w.sum() + gap[0], row_h[1]),
                     psd=(col_w[0], row_h[2]), att=(col_w[0], row_h[3]), bar=(col_w[1], row_h[2:].sum() + gap[1]))
        deco = {name: decorations(name, *sizes[name]) for name in names}
        L = np.array([max(deco[n][0] for n in ("sig", "rem", "psd", "att")), deco["bar"][0]])
        R = np.array([max(deco["psd"][1], deco["att"][1]), max(deco["sig"][1], deco["rem"][1], deco["bar"][1])])
        T = np.array([deco["sig"][2], deco["rem"][2], max(deco["psd"][2], deco["bar"][2]), deco["att"][2]])
        B = np.array([deco["sig"][3], deco["rem"][3], deco["psd"][3], max(deco["att"][3], deco["bar"][3])])
        w_cb, h_cb = max(w_pad, 0.02/2/2*W), max(h_pad, 0.02/2/4*H)  # Padding (inside the grid: w/hspace)
        Lcb, Rcb = np.array([w_pad, w_cb]), np.array([w_cb, w_pad])
        Tcb, Bcb = np.array([h_pad] + [h_cb]*3), np.array([h_cb]*3 + [h_pad])
        col_w = width_ratios*(W - (L + R + Lcb + Rcb).sum())/width_ratios.sum()
        row_h = height_ratios*(H - sup_margin - (T + B + Tcb + Bcb).sum())/height_ratios.sum()
        lefts = np.cumsum(np.r_[0, (Lcb + L + col_w + R + Rcb)[:-1]]) + Lcb + L
        tops = sup_margin + np.cumsum(np.r_[0, (Tcb + T + row_h + B + Bcb)[:-1]]) + Tcb + T
        gap[:] = [lefts[1] - lefts[0] - col_w[0], tops[3] - tops[2] - row_h[2]]  # Inside the spanning axes
        span_r = lefts[1] + col_w[1]
        boxes = {"sig": (lefts[0], tops[0], span_r, tops[0] + row_h[0]),
                 "rem": (lefts[0], tops[1], span_r, tops[1] + row_h[1]),
                 "psd": (lefts[0], tops[2], lefts[0] + col_w[0], tops[2] + row_h[2]),
                 "att": (lefts[0], tops[3], lefts[0] + col_w[0], tops[3] + row_h[3]),
                 "bar": (lefts[1], tops[2], span_r, tops[3] + row_h[3])}
        return boxes, col_w, row_h

    gap = np.zeros(2)

    def layout_for(W):
        col_w, row_h = width_ratios*0.4*W, height_ratios*0.15*H
        gap[:] = [0.1*W, 0.05*H]
        for _ in range(5):  # Fixed point (matplotlib: 2 passes)
            boxes, col_w, row_h = solve(W, col_w, row_h)
        for name, (l, tp, r, b) in boxes.items():  # Sizes of the final boxes (ticks of A for B)
            sizes[name] = (r - l, b - tp)
        # Bounding box of every element (bbox_inches="tight"): x0, x1, top, bottom
        sup_w = text(suptitle, 16)[0]
        ext = [(W/2 - sup_w/2, W/2 + sup_w/2, h_pad, h_pad + text_h(suptitle, 16, va="top"))]
        for name, (l, tp, r, b) in boxes.items():
            tw = title_ext[name][0]
            ext.append(((l + r - tw)/2, (l + r + tw)/2, tp - pt(titles[name][1]) - title_ext[name][1], tp))
            dl, dr, _, _ = decorations(name, r - l, b - tp)
            ext.append((l - dl, r + dr, tp, b))
            ext.append((l, r, b, b + xlabel_stack(name, r - l)))
        x0, x1 = min(e[0] for e in ext) - pad, max(e[1] for e in ext) + pad
        y0, y1 = min(e[2] for e in ext) - pad, max(e[3] for e in ext) + pad
        return boxes, x0, y0, int(x1 - x0), int(y1 - y0)

    W = 14*dpi
    boxes, crop_x, crop_y, img_w, img_h = layout_for(W)
    for _ in range(4 if width is not None else 0):  # Only the axes area changes with width
        W += width - img_w
        boxes, crop_x, crop_y, img_w, img_h = layout_for(W)
    width, height = img_w, img_h
    boxes = {k: (l - crop_x, tp - crop_y, r - crop_x, b - crop_y) for k, (l, tp, r, b) in boxes.items()}
    X = lambda v: v/width  # px -> paper
    Y = lambda v: 1 - v/height

    # --- loc="best": what matplotlib counts (lines, fill_between polygons) ---
    def to_px(name, x, y=None, y_axes=None):
        """Data (or y in axes fraction) -> px"""
        l, tp, r, b = boxes[name]
        rx, ry = ranges[x_of[name]], ranges[y_of[name]]
        px = l + (np.asarray(x, float) - rx[0])/(rx[1] - rx[0])*(r - l)
        py = (b - np.asarray(y_axes, float)*(b - tp) if y is None
              else b - (np.asarray(y, float) - ry[0])/(ry[1] - ry[0])*(b - tp))
        return px, py

    def span_paths(name):
        """Rectangles of the spans (fill_between with the axes transform), one path per run"""
        paths = []
        for m in (mask_clicks, mask_preselected):
            d = np.diff(np.r_[0, m.astype(np.int8), 0])
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1):
                paths.append(to_px(name, [t[s], t[e], t[e], t[s], t[s]], y_axes=[0, 0, 1, 1, 0]))
        return paths

    paths = {"sig": [to_px("sig", t, original), to_px("sig", t, clean)] + span_paths("sig"),
             "rem": [to_px("rem", t, removed)] + span_paths("rem"),
             "psd": [to_px("psd", f, p) for p in psd],
             "att": [to_px("att", f, att_db), to_px("att", ranges["f"], [0, 0])]}  # Curve and axhline

    # Legend frames (left, top) and plotly legends (one per item)
    frames, legend_of, legends = {}, {}, {}
    for key, (labels, ncol, size) in legend_specs.items():
        leg_w, leg_h, item_boxes = leg_layout[key]
        if key == "spans":  # loc="lower center", bbox_to_anchor=(0.5, 1), borderaxespad 0.2
            l, tp, r, b = boxes["sig"]
            frames[key] = ((l + r - leg_w)/2, tp - 0.2*em - leg_h)
        else:
            frames[key] = mpl_best_legend(boxes[key], (leg_w, leg_h), paths[key], em)
        fl, ft = frames[key]
        for k, (x0, _, top, bottom) in enumerate(item_boxes):
            lid = "legend" if not legends else f"legend{len(legends) + 1}"
            legend_of[(key, k)] = lid
            # Text at the matplotlib position (plotly: symbol at [5, 35] px, text at 40 px from the legend x).
            # Raised 0.1 font sizes: Arial (plotly) sits lower than DejaVu Sans
            legends[lid] = dict(x=X(fl + x0 - 40), y=Y(ft + (top + bottom)/2 - 0.1*pt(size)), xanchor="left",
                                yanchor="middle", orientation="h", bgcolor="rgba(0, 0, 0, 0)", borderwidth=0,
                                font=dict(size=pt(size)), groupclick="togglegroup", itemwidth=30)

    def item(key, k, show=True):
        """Legend arguments of the item k of the legend key"""
        return dict(name=legend_specs[key][0][k], legend=legend_of[(key, k)], legendgroup=f"{key}{k}",
                    showlegend=show)

    # --- Axes ---
    axis_style = dict(showline=True, mirror=True, linecolor="black", linewidth=pt(0.8), ticks="outside",
                      ticklen=pt(3.5), tickwidth=pt(0.8), tickcolor="black", tickfont=dict(size=em),
                      showgrid=True, gridcolor="#b0b0b0", gridwidth=pt(0.8), zeroline=False, automargin=False,
                      exponentformat="none", tickformatstops=MPL_TICKFORMAT)
    light_grid = rgba("#b0b0b0", 0.3)  # grid(alpha=0.3)

    def ticks_of(r, length, label_factor):
        """matplotlib ticks: nticks (plotly keeps its automatic ticks when zooming) or, for steps of 2.5 x 10^k
        (plotly only uses 1, 2, 5), fixed step with the decimals of matplotlib"""
        ticks = mpl_ticks(r, length/PT, label_factor)
        step = ticks[1] - ticks[0]
        if np.isclose(step/10**np.floor(np.log10(step)), 2.5):
            label = mpl_tick_labels(ticks)[0]
            dec = len(label.split(".")[1]) if "." in label else 0
            return dict(tickmode="linear", tick0=ticks[0], dtick=step, tickformat=f".{dec}f", tickformatstops=[])
        return dict(nticks=mpl_nticks(r, length/PT, label_factor))

    def x_axis(name, anchor, **kw):
        l, tp, r, b = boxes[name]
        rng = ranges[x_of[name]]
        return dict(domain=[X(l), X(r)], anchor=anchor, range=rng, **{**axis_style, **ticks_of(rng, r - l, 3), **kw},
                    ticklabelstandoff=round(-2.2*PT))

    def y_axis(name, anchor, **kw):
        l, tp, r, b = boxes[name]
        rng = ranges[y_of[name]]
        length = boxes["sig"][3] - boxes["sig"][1] if name == "rem" else b - tp  # Shared locator of A
        ticks = {} if name == "bar" else ticks_of(rng, length, 2)  # E: one tick per band
        return dict(domain=[Y(b), Y(tp)], anchor=anchor, range=rng, **{**axis_style, **ticks, **kw})

    hidden = lambda name, anchor: dict(domain=[Y(boxes[name][3]), Y(boxes[name][1])], anchor=anchor, range=[0, 1],
                                       fixedrange=True, visible=False)
    layout = dict(
        xaxis=x_axis("sig", "y"), yaxis=hidden("sig", "x"),  # A (y: spans, y2: data)
        yaxis2={**y_axis("sig", "x"), "overlaying": "y", "side": "left"},
        xaxis2=x_axis("rem", "y3", matches="x"), yaxis3=hidden("rem", "x2"),  # B (y3: spans, y4: data)
        yaxis4={**y_axis("rem", "x2", matches="y2"), "overlaying": "y3", "side": "left"},
        xaxis3=x_axis("psd", "y5"), yaxis5=y_axis("psd", "x3"),  # C
        xaxis4=x_axis("att", "y6", matches="x3", gridcolor=light_grid),  # D
        yaxis6=y_axis("att", "x4", gridcolor=light_grid),
        xaxis5=x_axis("bar", "y7", gridcolor=light_grid),  # E
        yaxis7=y_axis("bar", "x5", showgrid=False, tickmode="array", tickvals=list(range(n_b)),
                      ticktext=[s.replace("\n", "<br>") for s in bar_ticks]),
    )

    fig = go.Figure()
    line = lambda color, lw, dash=None: dict(color=MPL_COLORS[color], width=pt(lw), dash=dash)

    def add_spans(xa, ya, show):
        for k, (m, color) in enumerate([(mask_clicks, "tab:red"), (mask_preselected, "tab:green")]):
            x, y = mask_polygons(m, t)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=rgba(MPL_COLORS[color], 0.3),
                                     line=dict(width=0), hoverinfo="skip", xaxis=xa, yaxis=ya, **item("spans", k, show)))

    # A: original and filtered (spans below)
    add_spans("x", "y", True)
    for k, (y, color) in enumerate([(original, "gray"), (clean, "tab:green")]):
        fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=y, mode="lines", line=line(color, 1), xaxis="x", yaxis="y2",
                                 hovertemplate="%{x:.4f} s, %{y:.4g}", **item("sig", k)))

    # B: residual
    add_spans("x2", "y3", False)
    fig.add_trace(go.Scatter(x0=t[0], dx=1/fs, y=removed, mode="lines", line=line("tab:red", 1), xaxis="x2",
                             yaxis="y4", hovertemplate="%{x:.4f} s, %{y:.4g}", **item("rem", 0)))

    # C: PSD
    for k, (p, color) in enumerate(zip(psd, ["gray", "tab:green", "tab:red"])):
        fig.add_trace(go.Scatter(x=f, y=p, mode="lines", line=line(color, 1.5), xaxis="x3", yaxis="y5",
                                 hovertemplate="%{x:.0f} Hz, %{y:.1f} dB/Hz", **item("psd", k)))

    # D: attenuation and 0 dB
    fig.add_trace(go.Scatter(x=ranges["f"], y=[0, 0], mode="lines", line=line("gray", 1.5, mpl_dashes(DASH_PATTERNS["--"], 1.5, PT)),
                             xaxis="x4", yaxis="y6", hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=f, y=att_db, mode="lines", line=line("brown", 1.5), xaxis="x4", yaxis="y6",
                             hovertemplate="%{x:.0f} Hz, %{y:.2f} dB<extra></extra>", **item("att", 0)))

    # E: removed energy per band (barh, height 0.8) and bar labels
    fig.add_trace(go.Bar(x=pct, y=np.arange(n_b), orientation="h", width=0.8, marker=dict(color=rgba(MPL_COLORS["tab:red"], 0.7),
                         line=dict(width=0)), xaxis="x5", yaxis="y7", showlegend=False,
                         customdata=[s.replace("\n", "") for s in bar_ticks],
                         hovertemplate="%{customdata}: %{x:.1f} % of the band energy<extra></extra>"))

    # --- Titles, labels, bar labels ---
    ann = lambda s, x, y, size, xanchor="center", **kw: dict(text=s, x=X(x), y=Y(y), xref="paper", yref="paper",
                                                             xanchor=xanchor, yanchor="middle", showarrow=False,
                                                             font=dict(size=pt(size)), **kw)
    annotations = [ann(suptitle, W/2 - crop_x, h_pad - crop_y + text_h(suptitle, 16, va="top")/2 - 0.1*pt(16), 16)]
    for name, (l, tp, r, b) in boxes.items():
        baseline = tp - pt(titles[name][1])
        annotations.append(ann(f"<b>{titles[name][0]}</b>", (l + r)/2,
                               baseline - (title_ext[name][1] - title_ext[name][2])/2 - 0.1*pt(14), 14))
        if name in ylabels:
            x = l - ylabel_space(name, b - tp) + text_h(ylabels[name], 12)/2
            annotations.append(ann(ylabels[name], x, (tp + b)/2, 12, textangle=-90))
        if name in xlabels:
            annotations.append(ann(xlabels[name], (l + r)/2,
                                   b + xlabel_stack(name, r - l) - text_h(xlabels[name], 12, va="top")/2, 12))
    for k, (v, s) in enumerate(zip(pct, bar_labels)):  # Follow the zoom (data coordinates)
        annotations.append(dict(text=s, x=v, y=k, xref="x5", yref="y7", xanchor="left", yanchor="middle",
                                xshift=pt(3), yshift=0.1*em, showarrow=False, font=dict(size=em)))
    shapes = []
    for key in legend_specs:
        fl, ft = frames[key]
        leg_w, leg_h, _ = leg_layout[key]
        shapes.append(dict(type="path", xref="paper", yref="paper", layer="above",
                           fillcolor="rgba(255, 255, 255, 0.8)", line=dict(color="#cccccc", width=pt(1)),
                           path=rounded_rect(X(fl), X(fl + leg_w), Y(ft + leg_h), Y(ft), 0.2*em, width, height)))

    fig.update_layout(
        **layout, **legends, width=width, height=height, margin=dict(l=0, r=0, t=0, b=0, autoexpand=False),
        annotations=annotations, shapes=shapes, font=dict(family=FONT, size=em, color="black"),
        paper_bgcolor="white", plot_bgcolor="white", hovermode="closest", template="none"
    )

    return fig

# ====================================================================================================================
# 5. Summary
# ====================================================================================================================

# File names (without extension) of the figures saved by summary_plots, in the order they are shown
FIGURE_NAMES = ["decomposition_summary", "detection_map", "impulsiveness", "isolation", "misalign",
                "click_criteria", "noise_removal", "frequency_response"]

def summary_plots(signal: np.ndarray, fs: float, details: dict, params: dict, save_dir: str | None = None,
                  scale: float = 5):
    """
    Shows all the summary figures of the noise removal, one after the other: SWT decomposition, detection map,
    impulsiveness, isolation, misalign, click classification, spectrograms and effect in frequency.

    Parameters
    ----------
    signal: np.ndarray
        The respiratory sound (first channel, as processed by remove_noise)
    fs: float
        Sample frequency
    details: dict
        Intermediate results of remove_noise (return_details=True)
    params: dict
        Algorithm constants used by the figures:
        - G_FAC: attenuation level (decomposition_summary_plot)
        - K: detection threshold (plot_detection_map)
        - Z_CLICK, Z_TOLERANCE, AR_ORDER: impulsiveness (plot_impulsiveness)
        - ISO_WIN, ISOLATION, ISOLATION_IMPULSIVE, ISOLATION_ALIGNED: isolation (plot_isolation)
        - MAX_MISALIGN: misalign threshold (plot_misalign)
    save_dir: str or None
        Folder where each figure is also saved as <save_dir>/<name>.png (created if needed; names: see
        FIGURE_NAMES). None -> the figures are only shown
    scale: float
        Scale factor of the saved images (size in px = scale x size of the figure)

    Examples
    --------
    >>> params = dict(G_FAC=G_FAC, K=K, Z_CLICK=Z_CLICK, Z_TOLERANCE=Z_TOLERANCE, AR_ORDER=AR_ORDER,
    ...               ISO_WIN=ISO_WIN, ISOLATION=ISOLATION, ISOLATION_IMPULSIVE=ISOLATION_IMPULSIVE,
    ...               ISOLATION_ALIGNED=ISOLATION_ALIGNED, MAX_MISALIGN=MAX_MISALIGN)
    >>> summary_plots(signal, fs, details, params)
    >>> summary_plots(signal, fs, details, params, save_dir="assets", scale=5)  # Also saves the .png images
    """
    figures = [
        lambda: decomposition_summary_plot(signal, fs, details, params),
        lambda: plot_detection_map(signal, fs, details, params),
        lambda: plot_impulsiveness(signal, fs, details, params),
        lambda: plot_isolation(signal, fs, details, params),
        lambda: plot_misalign(signal, fs, details, params),
        lambda: plot_click_criteria(fs, details),
        lambda: plot_noise_removal(fs, details),
        lambda: plot_frequency_response(fs, details),
    ]
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name, make_figure in zip(FIGURE_NAMES, figures):  # Each figure is built just before it is shown
        fig = make_figure()
        if save_dir is not None:
            fig.write_image(Path(save_dir) / f"{name}.png", scale=scale)
        fig.show()
