#!/usr/bin/env python3
"""
Interactive Bokeh server app for exploring Difference-of-Gaussians (DoG)
preprocessing of the DIC channel, as an alternative to quantify_cells.py's
current single light Gaussian blur (sigma=1.0) before the Sobel gradient step.

Panels: a=original DIC, b=heavily-blurred DIC at the slider's sigma (the
"background estimate" DoG would subtract), c and d=the same DoG (light-smoothed
sigma=1.0, same as the production pipeline, minus panel b) shown at two
different display scales.

Panel c uses an "honest" display scale (+/- 3 std of the current DoG image,
recomputed every time sigma changes). Panel d uses Bokeh/matplotlib's default
auto-contrast (stretched to the image's own actual min/max), which made a
near-zero, low-amplitude DoG image look deceptively identical to the raw DIC
panel in earlier exploration -- panel d is kept here specifically so that
misleading default is visible side by side with the honest version, not
hidden.

This tool was built to test a specific hypothesis: does subtracting a
heavily-smoothed background from the DIC image, before the existing
Sobel-gradient segmentation, improve on quantify_cells.py's current single
light Gaussian? On this dataset, the answer was no -- Sobel is already a
local derivative and therefore already largely insensitive to slow
background trends, so DoG changes results only for cells already sitting
within a hair of a calibrated threshold (length/width/aspect/solidity),
nudging a small number across the line in either direction, not
systematically improving segmentation. This app is kept as a general
exploration tool in case that conclusion needs revisiting on different data
(e.g. data with a much stronger illumination gradient).

Run:
    bokeh serve --show dog_sigma_explorer_app.py --args <input_dir>

Defaults to ./renamed_composites if omitted.
"""

import os
import sys

import numpy as np
import scipy.ndimage as ndi
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantify_cells import group_fovs, GAUSSIAN_SIGMA

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Button, Select, Div, Slider, LinearColorMapper, Range1d
from bokeh.plotting import figure

INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "renamed_composites"

SIGMA_LARGE_DEFAULT = 20
SIGMA_LARGE_MAX = 80

WHITE_STYLE = {"background-color": "white"}
FULL_WIDTH_TEXT_STYLE = {"background-color": "white", "margin": "0"}

fov_items = sorted(group_fovs(INPUT_DIR).items())
fov_items = [(key, paths) for key, paths in fov_items if "DIC" in paths]
if not fov_items:
    raise SystemExit(f"No DIC files found in {INPUT_DIR}")

SAMPLE_HEIGHT, SAMPLE_WIDTH = tifffile.imread(fov_items[0][1]["DIC"]).shape
ASPECT = SAMPLE_HEIGHT / SAMPLE_WIDTH
PANEL_W = 420
PANEL_H = int(PANEL_W * ASPECT)

_dic_cache = {}


def load_dic(idx):
    if idx not in _dic_cache:
        (prefix, fov_num), paths = fov_items[idx]
        dic = tifffile.imread(paths["DIC"]).astype(np.float64)
        light_smoothed = ndi.gaussian_filter(dic, sigma=GAUSSIAN_SIGMA)
        _dic_cache[idx] = dict(prefix=prefix, fov_num=fov_num, dic=dic, light_smoothed=light_smoothed)
    return _dic_cache[idx]


shared_x_range = Range1d(0, SAMPLE_WIDTH)
shared_y_range = Range1d(0, SAMPLE_HEIGHT)


def make_panel(title):
    fig = figure(
        title=title, width=PANEL_W, height=PANEL_H,
        sizing_mode="scale_width",
        x_range=shared_x_range, y_range=shared_y_range,
        tools="pan,wheel_zoom,reset", match_aspect=True,
        background_fill_color="white", border_fill_color="white",
    )
    fig.axis.visible = False
    fig.grid.visible = False
    fig.toolbar.logo = None
    return fig


dic_fig = make_panel("a: original DIC")
blur_fig = make_panel("b: heavily-blurred (background estimate)")
dog_fig = make_panel("c: DoG, honest scale (+/- 3 std)")
dog_unclipped_fig = make_panel("d: DoG, unclipped (matplotlib-style auto-contrast)")

dic_src = ColumnDataSource(data=dict(image=[]))
blur_src = ColumnDataSource(data=dict(image=[]))
dog_src = ColumnDataSource(data=dict(image=[]))

dic_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)
blur_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)
dog_mapper = LinearColorMapper(palette="Greys256", low=-1, high=1)
# same underlying DoG data as panel c's dog_mapper, but stretched to the image's own
# actual min/max -- this is what made a near-zero, low-amplitude DoG image look
# deceptively like the original DIC in earlier exploration.
dog_unclipped_mapper = LinearColorMapper(palette="Greys256", low=-1, high=1)

dic_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=dic_src, color_mapper=dic_mapper)
blur_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=blur_src, color_mapper=blur_mapper)
dog_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=dog_src, color_mapper=dog_mapper)
dog_unclipped_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=dog_src, color_mapper=dog_unclipped_mapper)

fov_options = [
    (str(i), f"{prefix}FOV{fov_num} ({i + 1}/{len(fov_items)})")
    for i, ((prefix, fov_num), _paths) in enumerate(fov_items)
]
fov_select = Select(title="Field of view", options=fov_options, value="0", width=320)
prev_button = Button(label="< Previous", width=100)
next_button = Button(label="Next >", width=100)
sigma_slider = Slider(
    title="Heavy blur sigma (px) -- the background estimate DoG subtracts",
    start=2, end=SIGMA_LARGE_MAX, value=SIGMA_LARGE_DEFAULT, step=1, width=700,
)
status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

current_idx = [0]


def update(idx=None, sigma=None):
    idx = current_idx[0] if idx is None else idx
    sigma = sigma_slider.value if sigma is None else sigma
    idx = max(0, min(len(fov_items) - 1, idx))
    current_idx[0] = idx
    fov_select.value = str(idx)

    data = load_dic(idx)
    dic, light_smoothed = data["dic"], data["light_smoothed"]
    blurred = ndi.gaussian_filter(dic, sigma=sigma)
    dog = light_smoothed - blurred

    dic_src.data = dict(image=[np.flipud(dic)])
    blur_src.data = dict(image=[np.flipud(blurred)])
    dog_src.data = dict(image=[np.flipud(dog)])

    dic_mapper.low, dic_mapper.high = float(dic.min()), float(dic.max())
    # panel b shares panel a's color scale on purpose -- makes the contrast
    # reduction from blurring visible, rather than each panel auto-stretched
    # to look equally "full contrast" regardless of how flat it really is.
    blur_mapper.low, blur_mapper.high = dic_mapper.low, dic_mapper.high

    clip = max(3 * float(dog.std()), 1e-6)
    dog_mapper.low, dog_mapper.high = -clip, clip
    dog_unclipped_mapper.low, dog_unclipped_mapper.high = float(dog.min()), float(dog.max())

    status_div.text = (
        f"<b>{data['prefix']}FOV{data['fov_num']}</b> &mdash; FOV {idx + 1} of {len(fov_items)} "
        f"&mdash; sigma={sigma:.0f}px &mdash; DoG range [{dog.min():.0f}, {dog.max():.0f}], "
        f"std={dog.std():.1f}, honest display clipped to +/-{clip:.0f} (3 std), "
        f"unclipped panel stretched to the full [{dog.min():.0f}, {dog.max():.0f}] range"
    )


def on_prev():
    update(idx=current_idx[0] - 1)


def on_next():
    update(idx=current_idx[0] + 1)


def on_select(attr, old, new):
    update(idx=int(new))


def on_sigma_change(attr, old, new):
    update(sigma=new)


prev_button.on_click(on_prev)
next_button.on_click(on_next)
fov_select.on_change("value", on_select)
sigma_slider.on_change("value", on_sigma_change)

instructions = Div(text=(
    "<p>Drag the <b>sigma</b> slider to change the heavy-blur background estimate (panel b) "
    "used to compute the Difference-of-Gaussians (panels c and d show the same DoG data). "
    "Panel c's display scale is recomputed every time (+/- 3 std of the current DoG image), "
    "so it honestly reflects how flat or structured the result really is at that sigma. "
    "Panel d shows the identical data stretched to its own min/max instead -- the "
    "matplotlib/Bokeh auto-contrast default -- which makes a near-zero, low-amplitude DoG "
    "image look deceptively like a normal-contrast image; compare c and d directly to see "
    "how much that display choice alone changes the apparent result. Panels a-d share "
    "pan/zoom.</p>"
), sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

layout = column(
    instructions,
    row(prev_button, next_button, fov_select),
    sigma_slider,
    status_div,
    row(dic_fig, blur_fig, dog_fig, dog_unclipped_fig, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
    styles=dict(WHITE_STYLE, padding="12px"),
)

update(idx=0, sigma=SIGMA_LARGE_DEFAULT)

curdoc().add_root(layout)
curdoc().title = "DoG sigma explorer"
