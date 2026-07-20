#!/usr/bin/env python3
"""
Interactive Bokeh server app for exploring how quantify_cells.py's calibrated
segmentation parameters affect accept/reject decisions, FOV by FOV.

Every slider defaults to the actual calibrated value used in production
(imported from quantify_cells.py, not retyped), and the math is the same
math -- this app imports disk/otsu_threshold/fit_ellipse/UM_PER_PX from
quantify_cells.py directly rather than reimplementing them, so exploring
here can't silently drift from what the real pipeline does.

Parameters split into two groups for responsiveness:
  - Structural (sigma, Otsu multiplier, closing/opening radius): changing
    these requires re-running the Sobel-gradient segmentation from scratch,
    so they're bound to value_throttled (fires once on release, not per
    drag frame).
  - Filter (min area, length/width/aspect ranges, min solidity): these only
    change which of the ALREADY-segmented components pass or fail, which is
    cheap to recompute, so they're bound to value (live update while
    dragging). Per-component geometry (area, ellipse axes, solidity, border
    contact) is computed once per structural-parameter change and cached;
    moving a filter slider just re-checks that cached geometry against the
    new thresholds.

Panels: a=DIC (raw), b=binary mask after Sobel+Otsu+morphology (before any
shape filtering), c=DIC with every component colored green (passes all
filters) or red (fails at least one), with full stats on hover -- reject
reasons are not simplified/summarized, hover shows the actual numbers
against the actual current thresholds.

A guard against pathological parameter combinations (e.g. sigma near 0,
which can produce tens of thousands of noise components -- see the
Gaussian-smoothing discussion in this project's README): if a structural
change produces more than MAX_COMPONENTS components, only the largest
MAX_COMPONENTS by area get per-component geometry computed, and a warning
is shown.

Run:
    bokeh serve --show segmentation_params_explorer_app.py --args <input_dir>

Defaults to ./renamed_composites if omitted.
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
from scipy.spatial import ConvexHull, QhullError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantify_cells import (
    group_fovs, disk, otsu_threshold, fit_ellipse, UM_PER_PX,
    GAUSSIAN_SIGMA, THRESHOLD_MULT, CLOSE_RADIUS_PX, OPEN_RADIUS_PX, MIN_COMPONENT_AREA_PX,
    LENGTH_UM_RANGE, WIDTH_UM_RANGE, ASPECT_RATIO_RANGE, MIN_SOLIDITY,
)

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Button, Select, Div, Slider, RangeSlider, HoverTool,
    LinearColorMapper, Range1d,
)
from bokeh.plotting import figure

INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "renamed_composites"

WHITE_STYLE = {"background-color": "white"}
FULL_WIDTH_TEXT_STYLE = {"background-color": "white", "margin": "0"}

GEOMETRY_FLOOR_PX = 5      # below this, not even worth computing a ConvexHull
MAX_COMPONENTS = 500       # safety cap for pathological (e.g. near-zero sigma) settings

OK_FILL, OK_LINE = "#5DCAA5", "#0F6E56"
REJECT_FILL, REJECT_LINE = "#E24B4A", "#A32D2D"

fov_items = sorted(group_fovs(INPUT_DIR).items())
fov_items = [(key, paths) for key, paths in fov_items if "DIC" in paths]
if not fov_items:
    raise SystemExit(f"No DIC files found in {INPUT_DIR}")

SAMPLE_HEIGHT, SAMPLE_WIDTH = tifffile.imread(fov_items[0][1]["DIC"]).shape
ASPECT = SAMPLE_HEIGHT / SAMPLE_WIDTH
PANEL_W = 450
PANEL_H = int(PANEL_W * ASPECT)

_dic_cache = {}
_geometry_cache = {}  # (idx, sigma, threshold_mult, close_r, open_r) -> dict


def load_dic(idx):
    if idx not in _dic_cache:
        (prefix, fov_num), paths = fov_items[idx]
        _dic_cache[idx] = dict(prefix=prefix, fov_num=fov_num, dic=tifffile.imread(paths["DIC"]))
    return _dic_cache[idx]


def height_to_bokeh_y(row_coords, height):
    return height - row_coords


def compute_geometry(idx, sigma, threshold_mult, close_r, open_r):
    """Run the actual segment_dic algorithm (same math, parametrized) and compute
    per-component geometry once. Independent of the filter-slider values."""
    key = (idx, sigma, threshold_mult, close_r, open_r)
    if key in _geometry_cache:
        return _geometry_cache[key]

    dic = load_dic(idx)["dic"].astype(np.float64)
    smooth = ndi.gaussian_filter(dic, sigma=sigma)
    grad_mag = np.hypot(ndi.sobel(smooth, axis=1), ndi.sobel(smooth, axis=0))
    thresh = otsu_threshold(grad_mag.ravel())
    mask = grad_mag > thresh * threshold_mult
    if close_r > 0:
        mask = ndi.binary_closing(mask, structure=disk(close_r))
    mask = ndi.binary_fill_holes(mask)
    if open_r > 0:
        mask = ndi.binary_opening(mask, structure=disk(open_r))
    labeled, n = ndi.label(mask)

    height, width = dic.shape
    sizes = ndi.sum(mask, labeled, index=np.arange(1, n + 1)) if n > 0 else np.array([])
    candidate_ids = [i + 1 for i, s in enumerate(sizes) if s >= GEOMETRY_FLOOR_PX]
    truncated = len(candidate_ids) > MAX_COMPONENTS
    if truncated:
        candidate_ids = sorted(candidate_ids, key=lambda i: -sizes[i - 1])[:MAX_COMPONENTS]

    components = []
    for label_id in candidate_ids:
        ys, xs = np.where(labeled == label_id)
        area = len(ys)
        major_px, minor_px = fit_ellipse(ys, xs)
        if minor_px <= 0:
            continue
        if area >= 3:
            try:
                hull_area = ConvexHull(np.column_stack([xs, ys])).volume
            except QhullError:
                # degenerate (collinear / near-1D) component -- e.g. a thin noise
                # line under extreme parameter settings. Not a valid solid shape,
                # so it should fail the solidity filter, not crash the app.
                hull_area = 0
        else:
            hull_area = 0
        solidity = area / hull_area if hull_area > 0 else 0
        touches_border = ys.min() == 0 or xs.min() == 0 or ys.max() == height - 1 or xs.max() == width - 1
        contour_ys = height_to_bokeh_y(ys, height)
        components.append(dict(
            ys=ys, xs=xs, area=area,
            length_um=major_px * UM_PER_PX, width_um=minor_px * UM_PER_PX,
            aspect_ratio=major_px / minor_px, solidity=solidity, touches_border=touches_border,
            cy=float(contour_ys.mean()), cx=float(xs.mean()),
        ))

    result = dict(mask=mask, n_raw_components=n, components=components, truncated=truncated)
    _geometry_cache.clear()  # only ever need the current structural setting
    _geometry_cache[key] = result
    return result


def build_outline(ys, xs, height):
    """Boundary polygon for one component, via find_contours on a tight local
    crop (not the full image) -- matters when there are hundreds of components."""
    from skimage.measure import find_contours
    y0, x0 = int(ys.min()), int(xs.min())
    local = np.zeros((int(ys.max()) - y0 + 3, int(xs.max()) - x0 + 3), dtype=np.float64)
    local[ys - y0 + 1, xs - x0 + 1] = 1
    contours = find_contours(local, 0.5)
    contour = max(contours, key=len)
    abs_ys = contour[:, 0] + y0 - 1
    abs_xs = contour[:, 1] + x0 - 1
    return abs_xs.tolist(), height_to_bokeh_y(abs_ys, height).tolist()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

shared_x_range = Range1d(0, SAMPLE_WIDTH)
shared_y_range = Range1d(0, SAMPLE_HEIGHT)


def make_panel(title, width=PANEL_W):
    fig = figure(
        title=title, width=width, height=int(width * ASPECT),
        sizing_mode="scale_width",
        x_range=shared_x_range, y_range=shared_y_range,
        tools="pan,wheel_zoom,reset", match_aspect=True,
        background_fill_color="white", border_fill_color="white",
    )
    fig.axis.visible = False
    fig.grid.visible = False
    fig.toolbar.logo = None
    return fig


dic_fig = make_panel("a: DIC (raw)")
mask_fig = make_panel("b: binary mask (post Sobel+Otsu+morphology, pre shape-filter)")
result_fig = make_panel("c: accept (green) / reject (red) -- hover for stats", width=PANEL_W)

dic_src = ColumnDataSource(data=dict(image=[]))
mask_src = ColumnDataSource(data=dict(image=[]))
dic_bg_src = ColumnDataSource(data=dict(image=[]))

dic_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)
mask_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)

dic_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=dic_src, color_mapper=dic_mapper)
mask_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=mask_src, color_mapper=mask_mapper)
result_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=dic_bg_src, color_mapper=dic_mapper)

components_src = ColumnDataSource(data=dict(
    xs=[], ys=[], fill_color=[], line_color=[],
    area=[], length_um=[], width_um=[], aspect_ratio=[], solidity=[], status=[],
))
patches = result_fig.patches(
    xs="xs", ys="ys", source=components_src,
    fill_color="fill_color", fill_alpha=0.45, line_color="line_color", line_width=2,
)
result_fig.add_tools(HoverTool(renderers=[patches], tooltips=[
    ("status", "@status"),
    ("area (px)", "@area"),
    ("length (um)", "@length_um{0.0}"),
    ("width (um)", "@width_um{0.0}"),
    ("aspect ratio", "@aspect_ratio{0.00}"),
    ("solidity", "@solidity{0.000}"),
]))

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

fov_options = [
    (str(i), f"{prefix}FOV{fov_num} ({i + 1}/{len(fov_items)})")
    for i, ((prefix, fov_num), _paths) in enumerate(fov_items)
]
fov_select = Select(title="Field of view", options=fov_options, value="0", width=300)
prev_button = Button(label="< Previous", width=100)
next_button = Button(label="Next >", width=100)
reset_button = Button(label="Reset to calibrated defaults", button_type="primary", width=220)
status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)
warning_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

# structural (expensive -- value_throttled, fires on release only)
sigma_slider = Slider(title="Gaussian smoothing sigma (px)", start=0.0, end=5.0,
                       value=GAUSSIAN_SIGMA, step=0.1, width=340)
threshold_mult_slider = Slider(title="Otsu threshold multiplier", start=0.05, end=1.5,
                                value=THRESHOLD_MULT, step=0.05, width=340)
close_radius_slider = Slider(title="Morphological closing radius (px)", start=0, end=30,
                              value=CLOSE_RADIUS_PX, step=1, width=340)
open_radius_slider = Slider(title="Morphological opening radius (px)", start=0, end=15,
                             value=OPEN_RADIUS_PX, step=1, width=340)

# filter (cheap -- live value, updates while dragging)
min_area_slider = Slider(title="Minimum component area (px^2)", start=GEOMETRY_FLOOR_PX, end=2000,
                          value=MIN_COMPONENT_AREA_PX, step=10, width=340)
length_range_slider = RangeSlider(title="Length range (um)", start=0, end=80,
                                   value=LENGTH_UM_RANGE, step=1, width=340)
width_range_slider = RangeSlider(title="Width range (um)", start=0, end=25,
                                  value=WIDTH_UM_RANGE, step=0.5, width=340)
aspect_range_slider = RangeSlider(title="Aspect ratio range", start=1, end=30,
                                   value=ASPECT_RATIO_RANGE, step=0.5, width=340)
min_solidity_slider = Slider(title="Minimum solidity", start=0.0, end=1.0,
                              value=MIN_SOLIDITY, step=0.01, width=340)

current_idx = [0]


def apply_filters_and_render():
    idx = current_idx[0]
    sigma, threshold_mult = sigma_slider.value, threshold_mult_slider.value
    close_r, open_r = int(close_radius_slider.value), int(open_radius_slider.value)
    geom = compute_geometry(idx, sigma, threshold_mult, close_r, open_r)

    min_area = min_area_slider.value
    length_lo, length_hi = length_range_slider.value
    width_lo, width_hi = width_range_slider.value
    aspect_lo, aspect_hi = aspect_range_slider.value
    min_solidity = min_solidity_slider.value

    height = load_dic(idx)["dic"].shape[0]
    xs_list, ys_list, fill_color, line_color = [], [], [], []
    area_list, length_list, width_list, aspect_list, solidity_list, status_list = [], [], [], [], [], []
    n_accepted = 0

    for comp in geom["components"]:
        reasons = []
        if comp["area"] < min_area:
            reasons.append("area")
        if not (length_lo <= comp["length_um"] <= length_hi):
            reasons.append("length")
        if not (width_lo <= comp["width_um"] <= width_hi):
            reasons.append("width")
        if not (aspect_lo <= comp["aspect_ratio"] <= aspect_hi):
            reasons.append("aspect")
        if comp["solidity"] < min_solidity:
            reasons.append("solidity")
        if comp["touches_border"]:
            reasons.append("border")

        accepted = not reasons
        n_accepted += accepted
        outline_xs, outline_ys = build_outline(comp["ys"], comp["xs"], height)
        xs_list.append(outline_xs)
        ys_list.append(outline_ys)
        fill_color.append(OK_FILL if accepted else REJECT_FILL)
        line_color.append(OK_LINE if accepted else REJECT_LINE)
        area_list.append(comp["area"])
        length_list.append(round(comp["length_um"], 2))
        width_list.append(round(comp["width_um"], 2))
        aspect_list.append(round(comp["aspect_ratio"], 3))
        solidity_list.append(round(comp["solidity"], 4))
        status_list.append("accepted" if accepted else "rejected: " + ",".join(reasons))

    components_src.data = dict(
        xs=xs_list, ys=ys_list, fill_color=fill_color, line_color=line_color,
        area=area_list, length_um=length_list, width_um=width_list,
        aspect_ratio=aspect_list, solidity=solidity_list, status=status_list,
    )

    data = load_dic(idx)
    warning_div.text = (
        f"<b>Warning:</b> {geom['n_raw_components']} raw components exceeded the "
        f"{MAX_COMPONENTS} cap -- only the largest {MAX_COMPONENTS} by area were evaluated."
        if geom["truncated"] else ""
    )
    status_div.text = (
        f"<b>{data['prefix']}FOV{data['fov_num']}</b> &mdash; FOV {idx + 1} of {len(fov_items)} "
        f"&mdash; {geom['n_raw_components']} raw component(s) &mdash; "
        f"<b>{n_accepted} accepted</b> of {len(geom['components'])} evaluated"
    )


def show_fov(idx):
    idx = max(0, min(len(fov_items) - 1, idx))
    current_idx[0] = idx
    fov_select.value = str(idx)

    dic = load_dic(idx)["dic"]
    dic_norm = (dic.astype(np.float64) - dic.min()) / (dic.max() - dic.min())
    dic_flipped = np.flipud(dic_norm)
    dic_src.data = dict(image=[dic_flipped])
    dic_bg_src.data = dict(image=[dic_flipped])

    apply_filters_and_render()
    render_mask()


def render_mask():
    idx = current_idx[0]
    sigma, threshold_mult = sigma_slider.value, threshold_mult_slider.value
    close_r, open_r = int(close_radius_slider.value), int(open_radius_slider.value)
    geom = compute_geometry(idx, sigma, threshold_mult, close_r, open_r)
    mask_src.data = dict(image=[np.flipud(geom["mask"]).astype(np.float64)])


def on_structural_change(attr, old, new):
    render_mask()
    apply_filters_and_render()


def on_filter_change(attr, old, new):
    apply_filters_and_render()


def on_prev():
    show_fov(current_idx[0] - 1)


def on_next():
    show_fov(current_idx[0] + 1)


def on_select(attr, old, new):
    show_fov(int(new))


def on_reset():
    sigma_slider.value = GAUSSIAN_SIGMA
    threshold_mult_slider.value = THRESHOLD_MULT
    close_radius_slider.value = CLOSE_RADIUS_PX
    open_radius_slider.value = OPEN_RADIUS_PX
    min_area_slider.value = MIN_COMPONENT_AREA_PX
    length_range_slider.value = LENGTH_UM_RANGE
    width_range_slider.value = WIDTH_UM_RANGE
    aspect_range_slider.value = ASPECT_RATIO_RANGE
    min_solidity_slider.value = MIN_SOLIDITY
    render_mask()
    apply_filters_and_render()


prev_button.on_click(on_prev)
next_button.on_click(on_next)
fov_select.on_change("value", on_select)
reset_button.on_click(on_reset)

for slider in (sigma_slider, threshold_mult_slider, close_radius_slider, open_radius_slider):
    slider.on_change("value_throttled", on_structural_change)

for slider in (min_area_slider, length_range_slider, width_range_slider,
               aspect_range_slider, min_solidity_slider):
    slider.on_change("value", on_filter_change)

instructions = Div(text=(
    "<p>Sliders default to quantify_cells.py's actual calibrated values. "
    "<b>Structural</b> parameters (sigma, Otsu multiplier, closing/opening radius) "
    "re-run the full segmentation on release; <b>filter</b> parameters (area, "
    "length/width/aspect ranges, solidity) just re-check the already-segmented "
    "components, live while dragging. Panel c colors every evaluated component "
    "green (passes all filters) or red (fails at least one) -- hover any shape "
    "for its exact numbers against the current thresholds. Panels a-c share "
    "pan/zoom.</p>"
), sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

structural_col = column(
    Div(text="<b>Structural (expensive -- updates on release)</b>", styles=WHITE_STYLE),
    sigma_slider, threshold_mult_slider, close_radius_slider, open_radius_slider,
    styles=WHITE_STYLE,
)
filter_col = column(
    Div(text="<b>Filter (cheap -- updates live)</b>", styles=WHITE_STYLE),
    min_area_slider, length_range_slider, width_range_slider, aspect_range_slider, min_solidity_slider,
    styles=WHITE_STYLE,
)

layout = column(
    instructions,
    row(prev_button, next_button, fov_select, reset_button),
    status_div,
    warning_div,
    row(structural_col, filter_col, sizing_mode="stretch_width"),
    row(dic_fig, mask_fig, result_fig, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
    styles=dict(WHITE_STYLE, padding="12px"),
)

show_fov(0)

curdoc().add_root(layout)
curdoc().title = "Segmentation parameter explorer"
