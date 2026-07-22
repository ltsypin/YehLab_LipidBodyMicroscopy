#!/usr/bin/env python3
"""
Interactive Bokeh server app for exploring quantify_cells.py's bright-blob
counting parameters, cell by cell, on either the BODIPY channel (lipid
bodies) or the Chlorophyll channel (plastids -- P. tricornutum normally
carries a single plastid that duplicates before the cell divides, so >1
detected plastid is a candidate marker for a dividing cell). Both channels
are counted by the same underlying math (quantify_cells.count_bright_blobs),
imported directly, not reimplemented -- switching the channel toggle just
picks which image is fed into that same function, so this app can't silently
drift from what the production pipeline does on either channel.

The DIC segmentation that decides which cells exist is treated as a FIXED
upstream input here -- for each FOV it's computed on demand (and cached)
using the current calibrated production pipeline (group_fovs/segment_dic/
accepted_cells/compute_dic_background/correct_dic_background, imported
directly) -- explore that with segmentation_params_explorer_app.py instead.
This app is only about what happens next, inside each already-accepted
cell's own fluorescence signal:

  - A per-cell Otsu threshold from the *smoothed* channel intensities within
    that cell's own mask only (not the whole FOV) -- matches production,
    where each cell gets its own bright/dim cutoff rather than one global
    threshold.
  - Either connected components on the resulting bright mask (plain
    threshold), or a watershed split at local intensity peaks first
    (production's actual method, as of the watershed integration -- see
    project conversation). Plain intensity thresholding can't separate two
    bright regions that are touching if the valley between their peaks never
    dips below the threshold; watershed finds one seed per local peak and
    floods outward from each, splitting at the ridge between them. A
    single-peak blob watersheds right back to itself unchanged.
  - Any bright blob at least `min_size_px` counts; smaller specks are noise.
  - Production always uses 8-connectivity for the plain-threshold path (and
    the watershed-equivalent connectivity=2); a 4-connected option is
    exposed too, since diagonal-only touches are a plausible source of
    over/under-splitting production doesn't currently explore.

BODIPY/lipid-body defaults (LIPID_SMOOTH_SIGMA, LIPID_MIN_SIZE_PX,
LIPID_WATERSHED_MIN_DISTANCE_PX) are calibrated-ish -- min peak distance 3px
was confirmed as a reasonable starting default (see project conversation).
Chlorophyll/plastid defaults (PLASTID_*) are NOT calibrated at all yet -- a
first production run with min_distance=3 (mirroring the BODIPY default) gave
implausible results (>1 plastid in 190 of 209 cells, i.e. it reads as "91% of
cells are dividing," which is not biologically credible), almost certainly
because plastids are larger and more internally textured than lipid
droplets, so a small min-distance finds spurious peaks within one plastid's
own texture. Switching the channel toggle resets sigma/threshold
multiplier/min-distance/min-size to that channel's own defaults, so
exploring one channel never leaves stale, wrong-channel parameter values in
place for the other. Use this app to find a Chlorophyll-appropriate
min-distance before trusting n_plastids for anything.

A "registration correction" toggle controls whether the fluorescence channel
is translated by the calibrated DIC-alignment offset
(quantify_cells.correct_fluorescence_registration) before anything else runs.
Watershed peak-seeding is restricted to the DIC-derived cell mask, so if the
fluorescence channel isn't registered to DIC first, a real peak sitting just
past the mask edge -- most likely near a cell's tapered tip, where the mask
is narrowest -- gets clipped and reported at the wrong location. Confirmed on
this dataset (see project conversation): in Arginine_Day3_rep2FOV2 cell 2,
the true peak of one plastid sat 11px outside the mask, at exactly the point
where the mask tapered toward the cell's tip; with the correction on, that
same peak lands in a part of the mask over 100px wide instead. This is now
on by default, matching quantify_cells_shifted.py.

Detected watershed peaks are drawn as x marks on panels d and e.

Navigation is two-level: pick a field of view, then step through that FOV's
accepted cells (Prev/Next roll over into the neighboring FOV once you run off
either end, skipping any FOV with zero accepted cells). Each FOV's cell list
is computed lazily on first visit and cached -- there's no whole-dataset
pre-scan at startup, so the app loads as fast as the other exploration apps
in this project rather than re-running a ~50-FOV segmentation pass on every
new browser session (Bokeh re-executes this whole script per session).

Each figure's own pixel width/height (not just its data range) is recomputed
to match the current cell's crop aspect ratio on every render, rather than
relying on Bokeh's match_aspect to recompute it dynamically -- match_aspect
did not reliably resize on every cell switch in testing (a cell's image could
render stretched to the previous cell's crop shape); setting width/height
directly is deterministic and doesn't depend on that client-side recompute.

Purely for legibility, a crop wider than it is tall gets rotated 90 degrees
(rotate_crop_for_display/rotate_point_for_display, matching np.rot90) before
display, on every panel and on every outline/peak drawn on top of them --
otherwise a horizontally-oriented cell renders thin and small once squeezed
into a page-width-constrained row of 5 panels. This is cosmetic only: it
doesn't touch any of the underlying math, just which way the crop is spun
before rendering.

Parameters split as in segmentation_params_explorer_app.py:
  - Structural (channel, smoothing sigma, threshold multiplier, connectivity,
    method, watershed peak distance): changing these re-runs thresholding
    and labeling from scratch, so they're bound to value_throttled (fires
    once on release).
  - Filter (minimum blob size): only changes which already-labeled blobs
    count, so it's bound to value (live update while dragging).

Panels: a=DIC + this cell's mask outline (context, for orientation), b=the
selected channel (raw), c=that channel smoothed (shares panel b's color
scale, so over-smoothing's peak-flattening is visible rather than hidden by
independent auto-contrast), d=bright mask before size filtering, e=labeled
blobs colored green (counted) or red (below the size cutoff), with size on
hover.

A guard against pathological parameter combinations (e.g. sigma near 0, which
can fragment a bright blob into many 1px noise specks): if a structural
change produces more than MAX_BLOBS blobs, only the largest MAX_BLOBS by size
get rendered, and a warning is shown.

Run:
    bokeh serve --show blob_counting_params_explorer_app.py --args <input_dir>

Defaults to ./renamed_composites if omitted.
"""

import os
import sys

import numpy as np
import scipy.ndimage as ndi
import tifffile
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import h_maxima

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantify_cells import (
    group_fovs, segment_dic, accepted_cells, otsu_threshold, UM_PER_PX,
    compute_dic_background, correct_dic_background, correct_fluorescence_registration,
    LIPID_SMOOTH_SIGMA, LIPID_MIN_SIZE_PX, LIPID_WATERSHED_MIN_DISTANCE_PX, LIPID_WATERSHED_MIN_PROMINENCE,
    PLASTID_SMOOTH_SIGMA, PLASTID_MIN_SIZE_PX, PLASTID_WATERSHED_MIN_DISTANCE_PX, PLASTID_WATERSHED_MIN_PROMINENCE,
)

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Button, Select, Div, Slider, HoverTool,
    LinearColorMapper, Range1d, RadioButtonGroup,
)
from bokeh.plotting import figure

INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "renamed_composites"

WHITE_STYLE = {"background-color": "white"}
FULL_WIDTH_TEXT_STYLE = {"background-color": "white", "margin": "0"}

CROP_PAD_PX = 15
BLOB_FLOOR_PX = 1     # a labeled blob is always >=1px by definition -- no floor needed below this
MAX_BLOBS = 500       # safety cap for pathological (e.g. near-zero sigma) settings
PANEL_W = 260

OK_FILL, OK_LINE = "#5DCAA5", "#0F6E56"
REJECT_FILL, REJECT_LINE = "#E24B4A", "#A32D2D"

CHANNEL_LABELS = ["BODIPY (lipid bodies)", "Chlorophyll (plastids)"]
CHANNEL_TIFF_KEY = {"BODIPY": "BODIPY", "Chlorophyll": "Chlorophyll"}
CHANNEL_DEFAULTS = {
    "BODIPY": dict(sigma=LIPID_SMOOTH_SIGMA, threshold_mult=1.0,
                   min_distance=LIPID_WATERSHED_MIN_DISTANCE_PX, min_prominence=LIPID_WATERSHED_MIN_PROMINENCE,
                   min_size=LIPID_MIN_SIZE_PX),
    "Chlorophyll": dict(sigma=PLASTID_SMOOTH_SIGMA, threshold_mult=1.0,
                         min_distance=PLASTID_WATERSHED_MIN_DISTANCE_PX, min_prominence=PLASTID_WATERSHED_MIN_PROMINENCE,
                         min_size=PLASTID_MIN_SIZE_PX),
}

all_fovs = group_fovs(INPUT_DIR)
fov_items = sorted(all_fovs.items())
fov_items = [(key, paths) for key, paths in fov_items if "DIC" in paths and "BODIPY" in paths and "Chlorophyll" in paths]
if not fov_items:
    raise SystemExit(f"No FOVs with DIC+BODIPY+Chlorophyll found in {INPUT_DIR}")

dic_paths = [paths["DIC"] for _key, paths in fov_items]
DIC_BACKGROUND = compute_dic_background(dic_paths)

_dic_cache = {}
_channel_cache = {}  # (fov_idx, channel) -> float64 array
_fov_cells_cache = {}  # fov_idx -> list of (ys, xs) accepted cells in that FOV


def load_dic_corrected(fov_idx):
    if fov_idx not in _dic_cache:
        _key, paths = fov_items[fov_idx]
        dic = tifffile.imread(paths["DIC"])
        _dic_cache[fov_idx] = correct_dic_background(dic, DIC_BACKGROUND)
    return _dic_cache[fov_idx]


def load_channel(fov_idx, channel):
    key = (fov_idx, channel)
    if key not in _channel_cache:
        _fov_key, paths = fov_items[fov_idx]
        _channel_cache[key] = tifffile.imread(paths[CHANNEL_TIFF_KEY[channel]]).astype(np.float64)
    return _channel_cache[key]


def get_cells_for_fov(fov_idx):
    """The production-accepted (ys, xs) cell masks for one FOV, computed on first
    visit and cached -- this is the whole-dataset scan's per-FOV cost, paid lazily
    instead of upfront for every session."""
    if fov_idx not in _fov_cells_cache:
        dic_corr = load_dic_corrected(fov_idx)
        labeled, n = segment_dic(dic_corr)
        cells = [(ys, xs) for ys, xs, _props in accepted_cells(labeled, n, dic_corr.shape)]
        _fov_cells_cache[fov_idx] = cells
    return _fov_cells_cache[fov_idx]


def height_to_bokeh_y(row_coords, height):
    return height - row_coords


def build_outline(ys, xs, height):
    """Boundary polygon for one blob/cell, via find_contours -- ys/xs and height
    must all be in the SAME local coordinate frame (e.g. crop-relative)."""
    from skimage.measure import find_contours
    y0, x0 = int(ys.min()), int(xs.min())
    local = np.zeros((int(ys.max()) - y0 + 3, int(xs.max()) - x0 + 3), dtype=np.float64)
    local[ys - y0 + 1, xs - x0 + 1] = 1
    contours = find_contours(local, 0.5)
    contour = max(contours, key=len)
    abs_ys = contour[:, 0] + y0 - 1
    abs_xs = contour[:, 1] + x0 - 1
    return abs_xs.tolist(), height_to_bokeh_y(abs_ys, height).tolist()


def crop_bounds(ys, xs, shape, pad=CROP_PAD_PX):
    height, width = shape
    r0 = max(0, int(ys.min()) - pad)
    r1 = min(height, int(ys.max()) + pad + 1)
    c0 = max(0, int(xs.min()) - pad)
    c1 = min(width, int(xs.max()) + pad + 1)
    return r0, r1, c0, c1


def rotate_crop_for_display(crop, do_rotate):
    """90deg CCW (np.rot90 default) if the crop is wider than tall, so a
    horizontally-oriented cell doesn't render tiny in a page-width-constrained
    panel row -- purely cosmetic, doesn't change any math."""
    return np.rot90(crop, k=1) if do_rotate else crop


def rotate_point_for_display(r, c, orig_crop_w, do_rotate):
    """Apply the SAME transform as rotate_crop_for_display to a crop-relative
    (row, col) point or array of points, so outlines/peaks drawn on top of a
    rotated image still land in the right place. Matches np.rot90(k=1): a point
    at (r, c) in a (crop_h, orig_crop_w)-shaped array lands at
    (orig_crop_w - 1 - c, r) after rotation."""
    if not do_rotate:
        return r, c
    return orig_crop_w - 1 - c, r


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

_geometry_cache = {}  # (fov_idx, cell_idx, channel, sigma, threshold_mult, connectivity, method, min_distance, min_prominence, registration) -> dict


def compute_blob_geometry(fov_idx, cell_idx, channel, sigma, threshold_mult, connectivity, method, min_distance,
                           min_prominence, registration):
    """Run the actual count_bright_blobs algorithm (same math, parametrized) and
    compute per-blob geometry once. Independent of the min-size filter value.
    method is "threshold" or "watershed" (production's current method -- see
    module docstring). registration=True applies correct_fluorescence_registration
    before anything else, matching quantify_cells_shifted.py."""
    key = (fov_idx, cell_idx, channel, sigma, threshold_mult, connectivity, method, min_distance, min_prominence,
           registration)
    if key in _geometry_cache:
        return _geometry_cache[key]

    ys, xs = get_cells_for_fov(fov_idx)[cell_idx]
    raw = load_channel(fov_idx, channel)
    if registration:
        raw = correct_fluorescence_registration(raw)
    height, width = raw.shape
    r0, r1, c0, c1 = crop_bounds(ys, xs, (height, width))

    smooth = ndi.gaussian_filter(raw, sigma=sigma)
    thresh = otsu_threshold(smooth[ys, xs])

    cell_mask = np.zeros_like(raw, dtype=bool)
    cell_mask[ys, xs] = True
    bright_mask = cell_mask & (smooth > thresh * threshold_mult)

    peak_ys, peak_xs = np.array([], dtype=int), np.array([], dtype=int)
    if method == "watershed":
        # Prominence (topographic: how deep is the valley to the nearest equal-or-
        # higher peak) is a non-spatial complement to min_distance -- see
        # quantify_cells.count_bright_blobs' docstring for the full explanation.
        search_mask = bright_mask
        if min_prominence > 0:
            search_mask = h_maxima(smooth, min_prominence).astype(bool) & bright_mask
        coords = peak_local_max(smooth, min_distance=int(min_distance), labels=search_mask.astype(int))
        if len(coords) == 0:
            structure = np.ones((3, 3)) if connectivity == 8 else None
            labeled, n = ndi.label(bright_mask, structure=structure)
        else:
            peak_ys, peak_xs = coords[:, 0], coords[:, 1]
            markers = np.zeros(smooth.shape, dtype=int)
            markers[peak_ys, peak_xs] = np.arange(1, len(coords) + 1)
            ws_connectivity = 2 if connectivity == 8 else 1
            labeled = watershed(-smooth, markers=markers, mask=bright_mask, connectivity=ws_connectivity)
            n = int(labeled.max())
    else:
        structure = np.ones((3, 3)) if connectivity == 8 else None
        labeled, n = ndi.label(bright_mask, structure=structure)

    sizes = ndi.sum(bright_mask, labeled, index=np.arange(1, n + 1)) if n > 0 else np.array([])

    truncated = n > MAX_BLOBS
    keep_ids = list(range(1, n + 1))
    if truncated:
        keep_ids = sorted(keep_ids, key=lambda i: -sizes[i - 1])[:MAX_BLOBS]

    blobs = []
    for label_id in keep_ids:
        bys, bxs = np.where(labeled == label_id)
        blobs.append(dict(ys=bys, xs=bxs, size_px=len(bys)))

    result = dict(
        raw_crop=raw[r0:r1, c0:c1],
        smooth_crop=smooth[r0:r1, c0:c1],
        bright_mask_crop=bright_mask[r0:r1, c0:c1].astype(np.float64),
        crop_bounds=(r0, r1, c0, c1),
        thresh=thresh, n_raw_blobs=n, blobs=blobs, truncated=truncated,
        cell_ys=ys, cell_xs=xs, peak_ys=peak_ys, peak_xs=peak_xs,
    )
    _geometry_cache.clear()  # only ever need the current structural setting
    _geometry_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

shared_x_range = Range1d(0, 1)
shared_y_range = Range1d(0, 1)


def make_panel(title):
    fig = figure(
        title=title, width=PANEL_W, height=PANEL_W,
        sizing_mode="scale_width",
        x_range=shared_x_range, y_range=shared_y_range,
        tools="pan,wheel_zoom,reset",
        background_fill_color="white", border_fill_color="white",
    )
    fig.axis.visible = False
    fig.grid.visible = False
    fig.toolbar.logo = None
    return fig


dic_fig = make_panel("a: DIC + outline")
raw_fig = make_panel("b: raw")
smooth_fig = make_panel("c: smoothed")
mask_fig = make_panel("d: bright mask")
result_fig = make_panel("e: counted/too small")
ALL_FIGS = (dic_fig, raw_fig, smooth_fig, mask_fig, result_fig)

dic_src = ColumnDataSource(data=dict(image=[np.zeros((1, 1))], dw=[1], dh=[1]))
raw_src = ColumnDataSource(data=dict(image=[np.zeros((1, 1))], dw=[1], dh=[1]))
smooth_src = ColumnDataSource(data=dict(image=[np.zeros((1, 1))], dw=[1], dh=[1]))
mask_src = ColumnDataSource(data=dict(image=[np.zeros((1, 1))], dw=[1], dh=[1]))
cell_outline_src = ColumnDataSource(data=dict(xs=[], ys=[]))
peaks_src = ColumnDataSource(data=dict(x=[], y=[]))

dic_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)
channel_mapper = LinearColorMapper(palette="Viridis256", low=0, high=1)
mask_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)

dic_fig.image(image="image", x=0, y=0, dw="dw", dh="dh", source=dic_src, color_mapper=dic_mapper)
dic_fig.line(x="xs", y="ys", source=cell_outline_src, color="#3366CC", line_width=2)
raw_fig.image(image="image", x=0, y=0, dw="dw", dh="dh", source=raw_src, color_mapper=channel_mapper)
smooth_fig.image(image="image", x=0, y=0, dw="dw", dh="dh", source=smooth_src, color_mapper=channel_mapper)
mask_fig.image(image="image", x=0, y=0, dw="dw", dh="dh", source=mask_src, color_mapper=mask_mapper)
mask_fig.scatter(x="x", y="y", source=peaks_src, marker="x", size=10, line_color="red", line_width=2)
result_fig.scatter(x="x", y="y", source=peaks_src, marker="x", size=10, line_color="black", line_width=2)

blobs_src = ColumnDataSource(data=dict(
    xs=[], ys=[], fill_color=[], line_color=[],
    size_px=[], size_um2=[], status=[],
))
patches = result_fig.patches(
    xs="xs", ys="ys", source=blobs_src,
    fill_color="fill_color", fill_alpha=0.5, line_color="line_color", line_width=2,
)
result_fig.add_tools(HoverTool(renderers=[patches], tooltips=[
    ("status", "@status"),
    ("size (px)", "@size_px"),
    ("size (um^2)", "@size_um2{0.00}"),
]))

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

fov_options = [
    (str(i), f"{prefix}FOV{fov_num} ({i + 1}/{len(fov_items)})")
    for i, ((prefix, fov_num), _paths) in enumerate(fov_items)
]
fov_select = Select(title="Field of view", options=fov_options, value="0", width=260)
cell_select = Select(title="Cell in this FOV", options=[("0", "-")], value="0", width=170)
prev_button = Button(label="< Previous cell", width=115)
next_button = Button(label="Next cell >", width=115)
reset_button = Button(label="Reset to calibrated defaults", button_type="primary", width=200)
status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)
warning_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

channel_toggle = RadioButtonGroup(labels=CHANNEL_LABELS, active=0, width=400)
registration_toggle = RadioButtonGroup(
    labels=["With registration correction (production default)", "Without correction (raw channel)"],
    active=0, width=400,
)
connectivity_toggle = RadioButtonGroup(
    labels=["8-connected (production default)", "4-connected"],
    active=0, width=400,
)
method_toggle = RadioButtonGroup(
    labels=["Simple threshold", "Watershed (production default)"],
    active=1, width=400,
)


def channel_value():
    return "BODIPY" if channel_toggle.active == 0 else "Chlorophyll"


def registration_value():
    return registration_toggle.active == 0


def connectivity_value():
    return 8 if connectivity_toggle.active == 0 else 4


def method_value():
    return "watershed" if method_toggle.active == 1 else "threshold"


# structural (expensive -- value_throttled, fires on release only)
sigma_slider = Slider(title="Smoothing sigma (px)", start=0.0, end=5.0,
                       value=LIPID_SMOOTH_SIGMA, step=0.1, width=340)
threshold_mult_slider = Slider(title="Otsu threshold multiplier", start=0.2, end=3.0,
                                value=1.0, step=0.05, width=340)
watershed_min_distance_slider = Slider(title="Watershed minimum peak distance (px)", start=1, end=30,
                                        value=LIPID_WATERSHED_MIN_DISTANCE_PX, step=1, width=340)
watershed_min_prominence_slider = Slider(title="Watershed minimum peak prominence (intensity units)", start=0, end=1500,
                                          value=LIPID_WATERSHED_MIN_PROMINENCE, step=10, width=340)

# filter (cheap -- live value, updates while dragging)
min_size_slider = Slider(title="Minimum blob size (px^2)", start=BLOB_FLOOR_PX, end=100,
                          value=LIPID_MIN_SIZE_PX, step=1, width=340)

current_fov_idx = [0]
current_cell_idx = [0]


def current_geometry():
    fov_idx, cell_idx = current_fov_idx[0], current_cell_idx[0]
    channel = channel_value()
    sigma, threshold_mult = sigma_slider.value, threshold_mult_slider.value
    connectivity, method = connectivity_value(), method_value()
    min_distance = watershed_min_distance_slider.value
    min_prominence = watershed_min_prominence_slider.value
    registration = registration_value()
    return compute_blob_geometry(fov_idx, cell_idx, channel, sigma, threshold_mult, connectivity, method,
                                  min_distance, min_prominence, registration)


def apply_filters_and_render():
    fov_idx, cell_idx = current_fov_idx[0], current_cell_idx[0]
    channel = channel_value()
    threshold_mult = threshold_mult_slider.value
    connectivity, method = connectivity_value(), method_value()
    geom = current_geometry()
    min_size = min_size_slider.value

    r0, r1, c0, c1 = geom["crop_bounds"]
    crop_h, crop_w = r1 - r0, c1 - c0
    do_rotate = crop_w > crop_h
    disp_h = crop_w if do_rotate else crop_h

    xs_list, ys_list, fill_color, line_color = [], [], [], []
    size_px_list, size_um2_list, status_list = [], [], []
    n_counted = 0

    for blob in geom["blobs"]:
        accepted = blob["size_px"] >= min_size
        n_counted += accepted
        blob_r, blob_c = rotate_point_for_display(blob["ys"] - r0, blob["xs"] - c0, crop_w, do_rotate)
        outline_xs, outline_ys = build_outline(blob_r, blob_c, disp_h)
        xs_list.append(outline_xs)
        ys_list.append(outline_ys)
        fill_color.append(OK_FILL if accepted else REJECT_FILL)
        line_color.append(OK_LINE if accepted else REJECT_LINE)
        size_px_list.append(blob["size_px"])
        size_um2_list.append(round(blob["size_px"] * UM_PER_PX ** 2, 4))
        status_list.append("counted" if accepted else "too small")

    blobs_src.data = dict(
        xs=xs_list, ys=ys_list, fill_color=fill_color, line_color=line_color,
        size_px=size_px_list, size_um2=size_um2_list, status=status_list,
    )

    (prefix, fov_num), _paths = fov_items[fov_idx]
    n_cells = len(get_cells_for_fov(fov_idx))
    blob_noun = "lipid bod" if channel == "BODIPY" else "plastid"
    blob_word = f"{blob_noun}y" if (blob_noun == "lipid bod" and n_counted == 1) else (
        f"{blob_noun}ies" if blob_noun == "lipid bod" else f"{blob_noun}{'s' if n_counted != 1 else ''}"
    )
    warning_div.text = (
        f"<b>Warning:</b> {geom['n_raw_blobs']} raw bright blob(s) exceeded the "
        f"{MAX_BLOBS} cap -- only the largest {MAX_BLOBS} by size were evaluated."
        if geom["truncated"] else ""
    )
    min_prominence = watershed_min_prominence_slider.value
    status_div.text = (
        f"<b>{prefix}FOV{fov_num}</b> &mdash; cell {cell_idx + 1} of {n_cells} in this FOV &mdash; "
        f"channel: <b>{channel}</b> &mdash; registration correction: <b>{'ON' if registration_value() else 'OFF'}</b> "
        f"&mdash; per-cell Otsu threshold (smoothed): {geom['thresh']:.1f} &times; {threshold_mult:.2f} "
        f"&mdash; {geom['n_raw_blobs']} raw bright blob(s) &mdash; "
        f"<b>{n_counted} {blob_word}</b> "
        f"(&ge; {min_size:.0f}px, {connectivity}-connected, method: {method}, "
        f"min prominence: {min_prominence:.0f})"
    )


def render_static_panels():
    channel = channel_value()
    geom = current_geometry()
    r0, r1, c0, c1 = geom["crop_bounds"]
    crop_h, crop_w = r1 - r0, c1 - c0
    do_rotate = crop_w > crop_h
    disp_h, disp_w = (crop_w, crop_h) if do_rotate else (crop_h, crop_w)

    # Set each figure's own pixel width/height to match this cell's DISPLAYED (post-
    # rotation) aspect ratio directly, rather than relying on Bokeh's match_aspect to
    # recompute it on every range change -- see module docstring for why.
    if disp_w >= disp_h:
        fig_w, fig_h = PANEL_W, max(1, round(PANEL_W * disp_h / disp_w))
    else:
        fig_w, fig_h = max(1, round(PANEL_W * disp_w / disp_h)), PANEL_W
    for fig in ALL_FIGS:
        fig.width, fig.height = fig_w, fig_h

    shared_x_range.start, shared_x_range.end = 0, disp_w
    shared_y_range.start, shared_y_range.end = 0, disp_h

    raw_fig.title.text = f"b: {channel} raw"
    smooth_fig.title.text = f"c: {channel} smoothed"

    dic_corr = load_dic_corrected(current_fov_idx[0])
    dic_crop = dic_corr[r0:r1, c0:c1]
    dic_crop_disp = rotate_crop_for_display(dic_crop, do_rotate)
    dic_src.data = dict(image=[np.flipud(dic_crop_disp)], dw=[disp_w], dh=[disp_h])
    dic_mapper.low, dic_mapper.high = float(dic_crop.min()), float(dic_crop.max())

    cell_r, cell_c = rotate_point_for_display(geom["cell_ys"] - r0, geom["cell_xs"] - c0, crop_w, do_rotate)
    outline_xs, outline_ys = build_outline(cell_r, cell_c, disp_h)
    cell_outline_src.data = dict(xs=outline_xs, ys=outline_ys)

    raw_crop, smooth_crop = geom["raw_crop"], geom["smooth_crop"]
    raw_src.data = dict(image=[np.flipud(rotate_crop_for_display(raw_crop, do_rotate))], dw=[disp_w], dh=[disp_h])
    smooth_src.data = dict(image=[np.flipud(rotate_crop_for_display(smooth_crop, do_rotate))], dw=[disp_w], dh=[disp_h])
    # b and c share panel b's (raw) color scale on purpose -- makes over-smoothing's
    # peak-flattening visible, rather than each panel auto-stretched independently.
    channel_mapper.low, channel_mapper.high = float(raw_crop.min()), float(raw_crop.max())

    mask_crop_disp = rotate_crop_for_display(geom["bright_mask_crop"], do_rotate)
    mask_src.data = dict(image=[np.flipud(mask_crop_disp)], dw=[disp_w], dh=[disp_h])

    if len(geom["peak_ys"]):
        peak_r, peak_c = rotate_point_for_display(geom["peak_ys"] - r0, geom["peak_xs"] - c0, crop_w, do_rotate)
        peaks_src.data = dict(x=peak_c.tolist(), y=height_to_bokeh_y(peak_r, disp_h).tolist())
    else:
        peaks_src.data = dict(x=[], y=[])


def show_cell(fov_idx, cell_idx):
    fov_idx = max(0, min(len(fov_items) - 1, fov_idx))
    n_cells = len(get_cells_for_fov(fov_idx))
    if n_cells == 0:
        return  # caller is responsible for landing only on FOVs with >=1 cell
    cell_idx = max(0, min(n_cells - 1, cell_idx))
    current_fov_idx[0], current_cell_idx[0] = fov_idx, cell_idx

    fov_select.value = str(fov_idx)
    cell_select.options = [(str(i), f"Cell {i + 1} of {n_cells}") for i in range(n_cells)]
    cell_select.value = str(cell_idx)

    render_static_panels()
    apply_filters_and_render()


def first_fov_with_cells(start, step):
    """Walk FOVs from `start` in direction `step` (+1/-1) until one has an
    accepted cell, or the dataset is exhausted (returns None)."""
    idx = start
    while 0 <= idx < len(fov_items):
        if len(get_cells_for_fov(idx)) > 0:
            return idx
        idx += step
    return None


def on_structural_change(attr, old, new):
    render_static_panels()
    apply_filters_and_render()


def on_channel_change(attr, old, new):
    defaults = CHANNEL_DEFAULTS[channel_value()]
    sigma_slider.value = defaults["sigma"]
    threshold_mult_slider.value = defaults["threshold_mult"]
    watershed_min_distance_slider.value = defaults["min_distance"]
    watershed_min_prominence_slider.value = defaults["min_prominence"]
    min_size_slider.value = defaults["min_size"]
    render_static_panels()
    apply_filters_and_render()


def on_filter_change(attr, old, new):
    apply_filters_and_render()


def on_prev():
    fov_idx, cell_idx = current_fov_idx[0], current_cell_idx[0]
    if cell_idx - 1 >= 0:
        show_cell(fov_idx, cell_idx - 1)
        return
    target = first_fov_with_cells(fov_idx - 1, -1)
    if target is not None:
        show_cell(target, len(get_cells_for_fov(target)) - 1)


def on_next():
    fov_idx, cell_idx = current_fov_idx[0], current_cell_idx[0]
    n_cells = len(get_cells_for_fov(fov_idx))
    if cell_idx + 1 < n_cells:
        show_cell(fov_idx, cell_idx + 1)
        return
    target = first_fov_with_cells(fov_idx + 1, 1)
    if target is not None:
        show_cell(target, 0)


def on_fov_select(attr, old, new):
    fov_idx = int(new)
    if len(get_cells_for_fov(fov_idx)) == 0:
        target = first_fov_with_cells(fov_idx, 1) or first_fov_with_cells(fov_idx, -1)
        if target is not None:
            show_cell(target, 0)
        return
    show_cell(fov_idx, 0)


def on_cell_select(attr, old, new):
    show_cell(current_fov_idx[0], int(new))


def on_reset():
    channel_toggle.active = 0
    registration_toggle.active = 0
    connectivity_toggle.active = 0
    method_toggle.active = 1
    defaults = CHANNEL_DEFAULTS["BODIPY"]
    sigma_slider.value = defaults["sigma"]
    threshold_mult_slider.value = defaults["threshold_mult"]
    watershed_min_distance_slider.value = defaults["min_distance"]
    watershed_min_prominence_slider.value = defaults["min_prominence"]
    min_size_slider.value = defaults["min_size"]
    render_static_panels()
    apply_filters_and_render()


prev_button.on_click(on_prev)
next_button.on_click(on_next)
fov_select.on_change("value", on_fov_select)
cell_select.on_change("value", on_cell_select)
reset_button.on_click(on_reset)
channel_toggle.on_change("active", on_channel_change)
registration_toggle.on_change("active", on_structural_change)
connectivity_toggle.on_change("active", on_structural_change)
method_toggle.on_change("active", on_structural_change)

for slider in (sigma_slider, threshold_mult_slider, watershed_min_distance_slider, watershed_min_prominence_slider):
    slider.on_change("value_throttled", on_structural_change)

min_size_slider.on_change("value", on_filter_change)

instructions = Div(text=(
    "<p><b>Channel</b> picks BODIPY (lipid bodies) or Chlorophyll (plastids); switching "
    "it resets the sliders below to that channel's own defaults, so stale parameters "
    "from the other channel never carry over. <b>Registration correction</b> translates "
    "the fluorescence channel to align with DIC before anything else runs (matches "
    "quantify_cells_shifted.py) -- watershed peak-seeding is restricted to the "
    "DIC-derived mask, so without this a real peak near a cell's tapered tip can fall "
    "just outside the mask and get reported at the wrong location; toggle it off to see "
    "the difference directly. Production's actual current method is <b>watershed</b> "
    "(the default here) -- <b>Simple threshold</b> is kept for comparison, matching the "
    "pre-watershed behavior. <b>Structural</b> parameters (channel, registration, sigma, "
    "threshold multiplier, connectivity, watershed peak distance/prominence) re-run "
    "thresholding and labeling on release; the <b>filter</b> parameter (minimum blob "
    "size) just re-checks the already-labeled blobs, live while dragging. Panel e colors "
    "every blob green (counted) or red (below the size cutoff) -- hover for its exact "
    "size. Panel a shows the DIC channel with this cell's own mask outline, for "
    "orientation only -- the DIC segmentation itself is fixed here; explore it with "
    "segmentation_params_explorer_app.py instead. Prev/Next cell rolls into the "
    "neighboring FOV once you run off either end. Panels a-e share pan/zoom. Watershed "
    "mode marks detected intensity peaks with x's on panels d/e -- a blob with 2+ "
    "surviving peaks gets split at the ridge between them; a single-peak blob is "
    "unaffected. <b>Minimum peak distance</b> is purely spatial (px) -- two peaks "
    "closer than this always collapse to one, however deep the valley between them. "
    "<b>Minimum peak prominence</b> is a different, complementary criterion (intensity "
    "units, disabled at 0): a peak only survives if there's no path to an equal-or-"
    "higher peak with an intensity drop smaller than this -- distinguishing two real "
    "neighboring blobs (deep valley, high prominence, stays split) from one blob with "
    "two lobes at different focus/brightness (shallow dip that never reaches "
    "background, low prominence, gets merged). Chlorophyll/plastid defaults are NOT "
    "yet calibrated (see module docstring) -- use this app to find sensible values "
    "before trusting n_plastids for anything.</p>"
), sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

structural_col = column(
    Div(text="<b>Structural (expensive -- updates on release)</b>", styles=WHITE_STYLE),
    channel_toggle,
    registration_toggle,
    connectivity_toggle,
    method_toggle,
    sigma_slider, threshold_mult_slider, watershed_min_distance_slider, watershed_min_prominence_slider,
    styles=WHITE_STYLE,
)
filter_col = column(
    Div(text="<b>Filter (cheap -- updates live)</b>", styles=WHITE_STYLE),
    min_size_slider,
    styles=WHITE_STYLE,
)

layout = column(
    instructions,
    row(prev_button, next_button, reset_button, sizing_mode="stretch_width"),
    row(fov_select, cell_select, sizing_mode="stretch_width"),
    status_div,
    warning_div,
    row(structural_col, filter_col, sizing_mode="stretch_width"),
    row(dic_fig, raw_fig, smooth_fig, mask_fig, result_fig, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
    styles=dict(WHITE_STYLE, padding="12px"),
)

_start_fov = first_fov_with_cells(0, 1)
if _start_fov is None:
    raise SystemExit("No accepted cells found in any FOV -- nothing to explore.")
show_cell(_start_fov, 0)

curdoc().add_root(layout)
curdoc().title = "Blob counting (lipid body / plastid) parameter explorer"
