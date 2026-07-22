#!/usr/bin/env python3
"""
Interactive Bokeh server app that merges qc_review_app.py's per-cell
note-taking/CSV-export workflow with segmentation_params_explorer_app.py's DIC
sliders and blob_counting_params_explorer_app.py's five-panel BODIPY/
Chlorophyll figure, so a human can walk the whole dataset cell by cell,
manually estimate segmentation/counting/focus parameters for each cell, and
export everything to one CSV -- for comparing manual parameter estimates
against an automated optimization, and for checking how consistent manual
estimates are across FOVs/cells.

Three tabs per cell:
  1. DIC segmentation -- the same 3-panel view as segmentation_params_explorer_app.py
     (a: raw DIC, b: binary mask, c: accept/reject with hover), at FULL-FOV
     scale, with every slider that script exposes. DIC segmentation is a
     property of the whole FOV, not of one cell, so its parameters are keyed
     and restored PER FOV (see "Parameter keying" below) -- panel c highlights
     the current cell's own outline in blue so you know which cell you're
     reviewing while looking at the whole field.
  2. Chlorophyll (plastids) -- the same 5-panel view as
     blob_counting_params_explorer_app.py (a: DIC+outline, b: raw, c:
     smoothed, d: bright mask, e: counted/too small), fixed to the
     Chlorophyll channel, with every slider that script exposes (including the
     skeleton-clustered prototype method), PLUS a focus-score section: a
     pre-Laplacian smoothing-sigma slider (0 = matches production's raw/
     unsmoothed compute_focus_score) and a focus-score threshold slider
     (defaults to PLASTID_MIN_FOCUS_SCORE), showing the live focus score and
     in_focus flag for the current cell.
  3. BODIPY (lipid bodies) -- identical to tab 2's 5-panel view, fixed to the
     BODIPY channel, no focus section (compute_focus_score is not used for
     BODIPY in production).

The underlying math is never reimplemented here -- segmentation/counting/focus
functions are imported from quantify_cells.py directly, exactly as the three
source apps already do; this app only adds the tabbed layout, per-cell
navigation, and the save/export workflow around them.

DIC segmentation itself is treated as a FIXED upstream input for navigation
and for tabs 2/3's cell list (compute_dic_background/correct_dic_background +
segment_dic/accepted_cells, the actual production pipeline) -- tab 1's own
sliders are an independent exploration surface layered on top and never
change which cells exist or how many there are; this avoids the cell list
shifting under you while you explore segmentation parameters.

Parameter keying: Chlorophyll/BODIPY parameters (and the note) are saved PER
CELL, since production already computes a per-cell Otsu threshold for blob
counting. DIC parameters are saved PER FOV, since DIC segmentation runs once
per FOV and it wouldn't make sense for the "same" segmentation to have two
different recorded parameter sets depending on which of its cells you
happened to be looking at when you tuned it. The exported CSV still has one
row per cell (matching qc_review_app.py's per-cell CSV convention); every
cell sharing a FOV gets that FOV's current DIC parameters copied into its row
at export time, so the CSV always reflects the latest DIC settings recorded
for that FOV regardless of which cell you were on when you set them.

Save behavior: navigating to another cell (Prev/Next/dropdowns) commits the
current sliders into an in-memory registry (DIC values keyed by FOV,
Chlorophyll/BODIPY values + note keyed by cell) -- nothing is written to disk
until you click "Export CSV". Revisiting an already-committed FOV/cell
restores its last-committed values into the sliders (rather than resetting
to calibrated defaults), so you can review or revise your own prior estimate.
The CSV filename is set in the app (default cell_parameter_review.csv,
joined with <output_dir>) and is loaded back into the in-memory registry both
at startup and whenever the filename is changed, mirroring qc_review_app.py.

Run:
    bokeh serve --show cell_params_review_app.py --args <input_dir> [<output_dir>]

Defaults to ./renamed_composites and ./quantification if omitted.
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
from scipy.spatial import ConvexHull, QhullError
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import h_maxima, skeletonize
from skimage.measure import find_contours

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantify_cells import (
    group_fovs, segment_dic, accepted_cells, otsu_threshold, disk, fit_ellipse, UM_PER_PX,
    compute_dic_background, correct_dic_background, correct_fluorescence_registration,
    has_body_branch, prune_skeleton, _geodesic_distances, compute_focus_score,
    SKELETON_PRUNE_ITER,
    GAUSSIAN_SIGMA, THRESHOLD_MULT, CLOSE_RADIUS_PX, OPEN_RADIUS_PX, MIN_COMPONENT_AREA_PX,
    LENGTH_UM_RANGE, WIDTH_UM_RANGE, ASPECT_RATIO_RANGE, MIN_SOLIDITY,
    LIPID_SMOOTH_SIGMA, LIPID_MIN_SIZE_PX, LIPID_WATERSHED_MIN_DISTANCE_PX, LIPID_WATERSHED_MIN_PROMINENCE,
    PLASTID_SMOOTH_SIGMA, PLASTID_MIN_SIZE_PX, PLASTID_WATERSHED_MIN_DISTANCE_PX, PLASTID_WATERSHED_MIN_PROMINENCE,
    PLASTID_MIN_FOCUS_SCORE,
)

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Button, Select, Div, Slider, RangeSlider, HoverTool, TextAreaInput, TextInput,
    LinearColorMapper, Range1d, RadioButtonGroup, TabPanel, Tabs,
)
from bokeh.plotting import figure

INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "renamed_composites"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "quantification"
DEFAULT_CSV_NAME = "cell_parameter_review.csv"

WHITE_STYLE = {"background-color": "white"}
FULL_WIDTH_TEXT_STYLE = {"background-color": "white", "margin": "0"}

OK_FILL, OK_LINE = "#5DCAA5", "#0F6E56"
REJECT_FILL, REJECT_LINE = "#E24B4A", "#A32D2D"
CURRENT_CELL_LINE = "#2255CC"

GEOMETRY_FLOOR_PX = 5
DIC_MAX_COMPONENTS = 500
BLOB_FLOOR_PX = 1
BLOB_MAX_COUNT = 500
CROP_PAD_PX = 15
CHANNEL_PANEL_W = 195  # 3/4 of the original 260 -- scales better across browser window widths
DEFAULT_CLUSTER_GAP_PX = 20  # prototype only, not calibrated -- see blob_counting_params_explorer_app.py

# ---------------------------------------------------------------------------
# Data loading: reuse quantify_cells.py's own functions for everything that
# decides which cells exist, so review == what's in the production CSVs.
# ---------------------------------------------------------------------------

all_fovs = group_fovs(INPUT_DIR)
fov_items = sorted(all_fovs.items())
fov_items = [(key, paths) for key, paths in fov_items if all(c in paths for c in ("DIC", "Chlorophyll", "BODIPY"))]
if not fov_items:
    raise SystemExit(f"No FOVs with DIC+BODIPY+Chlorophyll found in {INPUT_DIR}")

SAMPLE_HEIGHT, SAMPLE_WIDTH = tifffile.imread(fov_items[0][1]["DIC"]).shape
ASPECT = SAMPLE_HEIGHT / SAMPLE_WIDTH
DIC_PANEL_W = 315  # 3/4 of the original 420 -- scales better across browser window widths
DIC_PANEL_H = int(DIC_PANEL_W * ASPECT)

DIC_BACKGROUND = compute_dic_background([paths["DIC"] for _key, paths in fov_items])

_dic_raw_cache = {}
_dic_corrected_cache = {}
_channel_cache = {}
_fov_cells_cache = {}


def load_dic_raw(fov_idx):
    if fov_idx not in _dic_raw_cache:
        _key, paths = fov_items[fov_idx]
        _dic_raw_cache[fov_idx] = tifffile.imread(paths["DIC"]).astype(np.float64)
    return _dic_raw_cache[fov_idx]


def load_dic_corrected(fov_idx):
    if fov_idx not in _dic_corrected_cache:
        _dic_corrected_cache[fov_idx] = correct_dic_background(load_dic_raw(fov_idx), DIC_BACKGROUND)
    return _dic_corrected_cache[fov_idx]


def load_channel(fov_idx, channel):
    key = (fov_idx, channel)
    if key not in _channel_cache:
        _fov_key, paths = fov_items[fov_idx]
        _channel_cache[key] = tifffile.imread(paths[channel]).astype(np.float64)
    return _channel_cache[key]


def get_cells_for_fov(fov_idx):
    """The production-accepted (ys, xs) cell masks for one FOV -- FIXED upstream
    input for navigation and tabs 2/3, computed lazily and cached (see module
    docstring for why tab 1's own sliders never affect this list)."""
    if fov_idx not in _fov_cells_cache:
        dic_corr = load_dic_corrected(fov_idx)
        labeled, n = segment_dic(dic_corr)
        cells = [(ys, xs) for ys, xs, _props in accepted_cells(labeled, n, dic_corr.shape)]
        _fov_cells_cache[fov_idx] = cells
    return _fov_cells_cache[fov_idx]


def height_to_bokeh_y(row_coords, height):
    return height - row_coords


def build_outline(ys, xs, height):
    """Boundary polygon for one blob/component/cell, via find_contours -- ys/xs
    and height must all be in the SAME coordinate frame."""
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
    return np.rot90(crop, k=1) if do_rotate else crop


def rotate_point_for_display(r, c, orig_crop_w, do_rotate):
    if not do_rotate:
        return r, c
    return orig_crop_w - 1 - c, r


# ---------------------------------------------------------------------------
# Skeleton helpers (channel-independent -- built from the cell's own DIC
# mask), ported from blob_counting_params_explorer_app.py.
# ---------------------------------------------------------------------------

_skeleton_cache = {}


def find_tips_with_coords(skel):
    any_point = tuple(np.argwhere(skel)[0])
    dist_any = _geodesic_distances(skel, any_point)
    tip_a = tuple(np.unravel_index(np.argmax(np.where(skel, dist_any, -1)), skel.shape))
    dist_a = _geodesic_distances(skel, tip_a)
    tip_b = tuple(np.unravel_index(np.argmax(np.where(skel, dist_a, -1)), skel.shape))
    dist_b = _geodesic_distances(skel, tip_b)
    return tip_a, dist_a, tip_b, dist_b


def compute_cell_skeleton(fov_idx, cell_idx):
    key = (fov_idx, cell_idx)
    if key not in _skeleton_cache:
        ys, xs = get_cells_for_fov(fov_idx)[cell_idx]
        y0, x0 = int(ys.min()), int(xs.min())
        local_mask = np.zeros((int(ys.max()) - y0 + 3, int(xs.max()) - x0 + 3), dtype=bool)
        local_mask[ys - y0 + 1, xs - x0 + 1] = True
        skel = prune_skeleton(skeletonize(local_mask), SKELETON_PRUNE_ITER)
        tip_a, dist_a, tip_b, dist_b = find_tips_with_coords(skel)
        _skeleton_cache.clear()
        _skeleton_cache[key] = dict(skel=skel, tip_a=tip_a, tip_b=tip_b, dist_a=dist_a, dist_b=dist_b, y0=y0, x0=x0)
    return _skeleton_cache[key]


def project_peaks_onto_skeleton(peak_ys, peak_xs, skel_info):
    skel, dist_a, y0, x0 = skel_info["skel"], skel_info["dist_a"], skel_info["y0"], skel_info["x0"]
    skel_ys, skel_xs = np.where(skel)
    skel_arc = dist_a[skel_ys, skel_xs]
    arc_lens, sides = [], []
    for py, px in zip(peak_ys, peak_xs):
        lpy, lpx = py - y0 + 1, px - x0 + 1
        d2 = (skel_ys - lpy) ** 2 + (skel_xs - lpx) ** 2
        i = int(np.argmin(d2))
        nsy, nsx, arc_len = skel_ys[i], skel_xs[i], skel_arc[i]
        before = np.where(np.abs(skel_arc - (arc_len - 4)) < 1.5)[0]
        after = np.where(np.abs(skel_arc - (arc_len + 4)) < 1.5)[0]
        if len(before) and len(after):
            ta = (skel_ys[before[0]], skel_xs[before[0]])
            tb = (skel_ys[after[0]], skel_xs[after[0]])
            tangent = np.array([tb[0] - ta[0], tb[1] - ta[1]])
            radial = np.array([lpy - nsy, lpx - nsx])
            cross = tangent[0] * radial[1] - tangent[1] * radial[0]
            side = float(np.sign(cross))
        else:
            side = 0.0
        arc_lens.append(float(arc_len))
        sides.append(side)
    return np.array(arc_lens), np.array(sides)


def cluster_peaks_by_skeleton(peak_ys, peak_xs, skel_info, cluster_gap_px):
    if len(peak_ys) == 0:
        return np.array([], dtype=int), np.array([]), np.array([])
    arc_lens, sides = project_peaks_onto_skeleton(peak_ys, peak_xs, skel_info)
    order = np.lexsort((arc_lens, sides))
    cluster_ids = np.zeros(len(peak_ys), dtype=int)
    next_id = -1
    prev_side, prev_arc = None, None
    for idx in order:
        side, arc = sides[idx], arc_lens[idx]
        if prev_side is None or side != prev_side or abs(arc - prev_arc) > cluster_gap_px:
            next_id += 1
        cluster_ids[idx] = next_id
        prev_side, prev_arc = side, arc
    return cluster_ids, arc_lens, sides


# ---------------------------------------------------------------------------
# DIC-tab geometry (FOV scale), ported from segmentation_params_explorer_app.py.
# ---------------------------------------------------------------------------

_dic_geometry_cache = {}


def compute_dic_geometry(fov_idx, sigma, threshold_mult, close_r, open_r, corrected):
    key = (fov_idx, sigma, threshold_mult, close_r, open_r, corrected)
    if key in _dic_geometry_cache:
        return _dic_geometry_cache[key]

    dic = load_dic_raw(fov_idx)
    if corrected:
        dic = correct_dic_background(dic, DIC_BACKGROUND)
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
    truncated = len(candidate_ids) > DIC_MAX_COMPONENTS
    if truncated:
        candidate_ids = sorted(candidate_ids, key=lambda i: -sizes[i - 1])[:DIC_MAX_COMPONENTS]

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
                hull_area = 0
        else:
            hull_area = 0
        solidity = area / hull_area if hull_area > 0 else 0
        local_mask = np.zeros((ys.max() - ys.min() + 3, xs.max() - xs.min() + 3), dtype=bool)
        local_mask[ys - ys.min() + 1, xs - xs.min() + 1] = True
        body_branch = has_body_branch(local_mask)
        touches_border = ys.min() == 0 or xs.min() == 0 or ys.max() == height - 1 or xs.max() == width - 1
        components.append(dict(
            ys=ys, xs=xs, area=area,
            length_um=major_px * UM_PER_PX, width_um=minor_px * UM_PER_PX,
            aspect_ratio=major_px / minor_px, solidity=solidity, body_branch=body_branch,
            touches_border=touches_border,
        ))

    result = dict(mask=mask, n_raw_components=n, components=components, truncated=truncated)
    _dic_geometry_cache.clear()
    _dic_geometry_cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Channel-tab (Chlorophyll/BODIPY) blob geometry, ported from
# blob_counting_params_explorer_app.py.
# ---------------------------------------------------------------------------

_blob_geometry_cache = {}


def compute_blob_geometry(fov_idx, cell_idx, channel, sigma, threshold_mult, connectivity, method, min_distance,
                           min_prominence, cluster_gap, registration):
    key = (fov_idx, cell_idx, channel, sigma, threshold_mult, connectivity, method, min_distance, min_prominence,
           cluster_gap, registration)
    if key in _blob_geometry_cache:
        return _blob_geometry_cache[key]

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

    peak_ys, peak_xs, cluster_ids = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    arc_lens, sides = np.array([]), np.array([])
    if method in ("watershed", "skeleton"):
        search_mask = bright_mask
        if min_prominence > 0:
            search_mask = h_maxima(smooth, min_prominence).astype(bool) & bright_mask
        coords = peak_local_max(smooth, min_distance=int(min_distance), labels=search_mask.astype(int))
        if len(coords) == 0:
            structure = np.ones((3, 3)) if connectivity == 8 else None
            labeled, n = ndi.label(bright_mask, structure=structure)
        else:
            peak_ys, peak_xs = coords[:, 0], coords[:, 1]
            if method == "skeleton":
                skel_info = compute_cell_skeleton(fov_idx, cell_idx)
                cluster_ids, arc_lens, sides = cluster_peaks_by_skeleton(peak_ys, peak_xs, skel_info, cluster_gap)
                marker_ids = cluster_ids + 1
            else:
                marker_ids = np.arange(1, len(coords) + 1)
            markers = np.zeros(smooth.shape, dtype=int)
            markers[peak_ys, peak_xs] = marker_ids
            ws_connectivity = 2 if connectivity == 8 else 1
            labeled = watershed(-smooth, markers=markers, mask=bright_mask, connectivity=ws_connectivity)
            n = int(labeled.max())
    else:
        structure = np.ones((3, 3)) if connectivity == 8 else None
        labeled, n = ndi.label(bright_mask, structure=structure)

    sizes = ndi.sum(bright_mask, labeled, index=np.arange(1, n + 1)) if n > 0 else np.array([])

    truncated = n > BLOB_MAX_COUNT
    keep_ids = list(range(1, n + 1))
    if truncated:
        keep_ids = sorted(keep_ids, key=lambda i: -sizes[i - 1])[:BLOB_MAX_COUNT]

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
        cluster_ids=cluster_ids, arc_lens=arc_lens, sides=sides,
    )
    _blob_geometry_cache.clear()
    _blob_geometry_cache[key] = result
    return result


_focus_cache = {}


def compute_focus(fov_idx, cell_idx, registration, presmooth_sigma):
    key = (fov_idx, cell_idx, registration, presmooth_sigma)
    if key not in _focus_cache:
        ys, xs = get_cells_for_fov(fov_idx)[cell_idx]
        img = load_channel(fov_idx, "Chlorophyll")
        if registration:
            img = correct_fluorescence_registration(img)
        if presmooth_sigma > 0:
            img = ndi.gaussian_filter(img, sigma=presmooth_sigma)
        score = compute_focus_score(img, ys, xs)
        _focus_cache.clear()
        _focus_cache[key] = score
    return _focus_cache[key]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DIC_DEFAULTS = dict(
    correction=True, sigma=GAUSSIAN_SIGMA, threshold_mult=THRESHOLD_MULT,
    close_r=CLOSE_RADIUS_PX, open_r=OPEN_RADIUS_PX, min_area=MIN_COMPONENT_AREA_PX,
    length_range=LENGTH_UM_RANGE, width_range=WIDTH_UM_RANGE,
    aspect_range=ASPECT_RATIO_RANGE, min_solidity=MIN_SOLIDITY,
)

# Chlorophyll's own starting method/prominence (skeleton-clustered, min_prominence=600)
# and BODIPY's min_prominence=100 are this REVIEW TOOL's own defaults, set at the
# user's explicit request for manual-estimation sessions -- they deliberately differ
# from quantify_cells.py's own production constants (PLASTID_WATERSHED_MIN_PROMINENCE/
# LIPID_WATERSHED_MIN_PROMINENCE, both 0/disabled) and from blob_counting_params_
# explorer_app.py's defaults, which are untouched. Change here only, not there.
CHANNEL_DEFAULTS = {
    "Chlorophyll": dict(
        registration=True, connectivity=8, method="skeleton",
        sigma=PLASTID_SMOOTH_SIGMA, threshold_mult=1.0,
        min_distance=PLASTID_WATERSHED_MIN_DISTANCE_PX, min_prominence=600,
        cluster_gap=DEFAULT_CLUSTER_GAP_PX, min_size=PLASTID_MIN_SIZE_PX,
        focus_presmooth_sigma=0.0, focus_threshold=PLASTID_MIN_FOCUS_SCORE,
    ),
    "BODIPY": dict(
        registration=True, connectivity=8, method="watershed",
        sigma=LIPID_SMOOTH_SIGMA, threshold_mult=1.0,
        min_distance=LIPID_WATERSHED_MIN_DISTANCE_PX, min_prominence=100,
        cluster_gap=DEFAULT_CLUSTER_GAP_PX, min_size=LIPID_MIN_SIZE_PX,
    ),
}

# ---------------------------------------------------------------------------
# Navigation state + save/restore registries
# ---------------------------------------------------------------------------

current_fov_idx = [0]
current_cell_idx = [0]

dic_state_by_fov = {}   # (prefix, fov_num) -> dict matching DIC_DEFAULTS keys
cell_state = {}          # (prefix, fov_num, cell_id) -> dict(chlorophyll={...}, bodipy={...}, note=str)


def current_fov_key():
    (prefix, fov_num), _paths = fov_items[current_fov_idx[0]]
    return (prefix, fov_num)


def current_cell_id():
    return current_cell_idx[0] + 1


def current_cell_key():
    prefix, fov_num = current_fov_key()
    return (prefix, fov_num, current_cell_id())


# ---------------------------------------------------------------------------
# Tab 1: DIC segmentation (FOV scale)
# ---------------------------------------------------------------------------

dic_shared_x_range = Range1d(0, SAMPLE_WIDTH)
dic_shared_y_range = Range1d(0, SAMPLE_HEIGHT)


def make_dic_panel(title):
    fig = figure(
        title=title, width=DIC_PANEL_W, height=DIC_PANEL_H,
        # "fixed", not "scale_width" -- scale_width stretches to fill whatever width
        # the parent row/page happens to have, using width/height only for the aspect
        # ratio, so it ignores DIC_PANEL_W as an absolute size and can overflow a
        # narrow browser window regardless of how small DIC_PANEL_W is set. "fixed"
        # renders at exactly the given pixel size.
        sizing_mode="fixed",
        x_range=dic_shared_x_range, y_range=dic_shared_y_range,
        tools="pan,wheel_zoom,reset", match_aspect=True,
        background_fill_color="white", border_fill_color="white",
    )
    fig.axis.visible = False
    fig.grid.visible = False
    fig.toolbar.logo = None
    return fig


dic_a_fig = make_dic_panel("a: DIC (raw)")
dic_b_fig = make_dic_panel("b: binary mask (post Sobel+Otsu+morphology, pre shape-filter)")
dic_c_fig = make_dic_panel("c: accept (green) / reject (red) -- current cell outlined in blue")

dic_a_src = ColumnDataSource(data=dict(image=[]))
dic_mask_src = ColumnDataSource(data=dict(image=[]))
dic_c_bg_src = ColumnDataSource(data=dict(image=[]))

dic_a_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)
dic_mask_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)

dic_a_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=dic_a_src, color_mapper=dic_a_mapper)
dic_b_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=dic_mask_src, color_mapper=dic_mask_mapper)
dic_c_fig.image(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=dic_c_bg_src, color_mapper=dic_a_mapper)

dic_components_src = ColumnDataSource(data=dict(
    xs=[], ys=[], fill_color=[], line_color=[],
    area=[], length_um=[], width_um=[], aspect_ratio=[], solidity=[], body_branch=[], status=[],
))
dic_patches = dic_c_fig.patches(
    xs="xs", ys="ys", source=dic_components_src,
    fill_color="fill_color", fill_alpha=0.45, line_color="line_color", line_width=2,
)
dic_c_fig.add_tools(HoverTool(renderers=[dic_patches], tooltips=[
    ("status", "@status"),
    ("area (px)", "@area"),
    ("length (um)", "@length_um{0.0}"),
    ("width (um)", "@width_um{0.0}"),
    ("aspect ratio", "@aspect_ratio{0.00}"),
    ("solidity", "@solidity{0.000}"),
    ("body branch defect", "@body_branch"),
]))

dic_current_cell_src = ColumnDataSource(data=dict(xs=[], ys=[]))
dic_c_fig.line(x="xs", y="ys", source=dic_current_cell_src, color=CURRENT_CELL_LINE, line_width=3)

dic_status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)
dic_warning_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)
dic_reset_button = Button(label="Reset DIC tab to calibrated defaults", button_type="primary", width=200)

dic_correction_toggle = RadioButtonGroup(
    labels=["Background correction (default)", "No correction (raw DIC)"],
    active=0, width=300,
)
dic_sigma_slider = Slider(title="Gaussian smoothing sigma (px)", start=0.0, end=5.0,
                           value=GAUSSIAN_SIGMA, step=0.1, width=240)
dic_threshold_mult_slider = Slider(title="Otsu threshold multiplier", start=0.05, end=1.5,
                                    value=THRESHOLD_MULT, step=0.05, width=240)
dic_close_radius_slider = Slider(title="Morphological closing radius (px)", start=0, end=30,
                                  value=CLOSE_RADIUS_PX, step=1, width=240)
dic_open_radius_slider = Slider(title="Morphological opening radius (px)", start=0, end=15,
                                 value=OPEN_RADIUS_PX, step=1, width=240)
dic_min_area_slider = Slider(title="Minimum component area (px^2)", start=GEOMETRY_FLOOR_PX, end=2000,
                              value=MIN_COMPONENT_AREA_PX, step=10, width=240)
dic_length_range_slider = RangeSlider(title="Length range (um)", start=0, end=80,
                                       value=LENGTH_UM_RANGE, step=1, width=240)
dic_width_range_slider = RangeSlider(title="Width range (um)", start=0, end=25,
                                      value=WIDTH_UM_RANGE, step=0.5, width=240)
dic_aspect_range_slider = RangeSlider(title="Aspect ratio range", start=1, end=30,
                                       value=ASPECT_RATIO_RANGE, step=0.5, width=240)
dic_min_solidity_slider = Slider(title="Minimum solidity", start=0.0, end=1.0,
                                  value=MIN_SOLIDITY, step=0.01, width=240)


def dic_correction_enabled():
    return dic_correction_toggle.active == 0


def read_dic_state():
    return dict(
        correction=dic_correction_enabled(),
        sigma=dic_sigma_slider.value, threshold_mult=dic_threshold_mult_slider.value,
        close_r=int(dic_close_radius_slider.value), open_r=int(dic_open_radius_slider.value),
        min_area=dic_min_area_slider.value,
        length_range=tuple(dic_length_range_slider.value),
        width_range=tuple(dic_width_range_slider.value),
        aspect_range=tuple(dic_aspect_range_slider.value),
        min_solidity=dic_min_solidity_slider.value,
    )


def apply_dic_state(state):
    dic_correction_toggle.active = 0 if state["correction"] else 1
    dic_sigma_slider.value = state["sigma"]
    dic_threshold_mult_slider.value = state["threshold_mult"]
    dic_close_radius_slider.value = state["close_r"]
    dic_open_radius_slider.value = state["open_r"]
    dic_min_area_slider.value = state["min_area"]
    dic_length_range_slider.value = state["length_range"]
    dic_width_range_slider.value = state["width_range"]
    dic_aspect_range_slider.value = state["aspect_range"]
    dic_min_solidity_slider.value = state["min_solidity"]


def render_dic_mask():
    fov_idx = current_fov_idx[0]
    geom = compute_dic_geometry(fov_idx, dic_sigma_slider.value, dic_threshold_mult_slider.value,
                                 int(dic_close_radius_slider.value), int(dic_open_radius_slider.value),
                                 dic_correction_enabled())
    dic_mask_src.data = dict(image=[np.flipud(geom["mask"]).astype(np.float64)])


def apply_dic_filters_and_render():
    fov_idx = current_fov_idx[0]
    geom = compute_dic_geometry(fov_idx, dic_sigma_slider.value, dic_threshold_mult_slider.value,
                                 int(dic_close_radius_slider.value), int(dic_open_radius_slider.value),
                                 dic_correction_enabled())

    min_area = dic_min_area_slider.value
    length_lo, length_hi = dic_length_range_slider.value
    width_lo, width_hi = dic_width_range_slider.value
    aspect_lo, aspect_hi = dic_aspect_range_slider.value
    min_solidity = dic_min_solidity_slider.value

    xs_list, ys_list, fill_color, line_color = [], [], [], []
    area_list, length_list, width_list, aspect_list, solidity_list, status_list, body_branch_list = (
        [], [], [], [], [], [], [],
    )
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
        solidity_low = comp["solidity"] < min_solidity
        if solidity_low and comp["body_branch"]:
            reasons.append("solidity")
        if comp["touches_border"]:
            reasons.append("border")

        accepted = not reasons
        n_accepted += accepted
        outline_xs, outline_ys = build_outline(comp["ys"], comp["xs"], SAMPLE_HEIGHT)
        xs_list.append(outline_xs)
        ys_list.append(outline_ys)
        fill_color.append(OK_FILL if accepted else REJECT_FILL)
        line_color.append(OK_LINE if accepted else REJECT_LINE)
        area_list.append(comp["area"])
        length_list.append(round(comp["length_um"], 2))
        width_list.append(round(comp["width_um"], 2))
        aspect_list.append(round(comp["aspect_ratio"], 3))
        solidity_list.append(round(comp["solidity"], 4))
        body_branch_list.append("yes" if comp["body_branch"] else "no")
        status = "accepted" if accepted else "rejected: " + ",".join(reasons)
        if solidity_low and not comp["body_branch"]:
            status += " (solidity rescued: curved, no body defect)"
        status_list.append(status)

    dic_components_src.data = dict(
        xs=xs_list, ys=ys_list, fill_color=fill_color, line_color=line_color,
        area=area_list, length_um=length_list, width_um=width_list,
        aspect_ratio=aspect_list, solidity=solidity_list, body_branch=body_branch_list,
        status=status_list,
    )

    prefix, fov_num = current_fov_key()
    dic_warning_div.text = (
        f"<b>Warning:</b> {geom['n_raw_components']} raw components exceeded the "
        f"{DIC_MAX_COMPONENTS} cap -- only the largest {DIC_MAX_COMPONENTS} by area were evaluated."
        if geom["truncated"] else ""
    )
    dic_status_div.text = (
        f"<b>{prefix}FOV{fov_num}</b> &mdash; background correction: "
        f"<b>{'ON' if dic_correction_enabled() else 'OFF'}</b> "
        f"&mdash; {geom['n_raw_components']} raw component(s) &mdash; "
        f"<b>{n_accepted} accepted</b> of {len(geom['components'])} evaluated in this FOV "
        f"(current cell outlined in blue on panel c)"
    )


def render_dic_static():
    fov_idx, cell_idx = current_fov_idx[0], current_cell_idx[0]
    dic_flipped = np.flipud(load_dic_raw(fov_idx))
    dic_norm = (dic_flipped - dic_flipped.min()) / max(dic_flipped.max() - dic_flipped.min(), 1e-9)
    dic_a_src.data = dict(image=[dic_norm])
    dic_c_bg_src.data = dict(image=[dic_norm])

    ys, xs = get_cells_for_fov(fov_idx)[cell_idx]
    outline_xs, outline_ys = build_outline(ys, xs, SAMPLE_HEIGHT)
    dic_current_cell_src.data = dict(xs=outline_xs + [outline_xs[0]], ys=outline_ys + [outline_ys[0]])


def on_dic_structural_change(attr, old, new):
    render_dic_mask()
    apply_dic_filters_and_render()


def on_dic_filter_change(attr, old, new):
    apply_dic_filters_and_render()


def on_dic_reset():
    apply_dic_state(DIC_DEFAULTS)
    render_dic_mask()
    apply_dic_filters_and_render()


dic_reset_button.on_click(on_dic_reset)
dic_correction_toggle.on_change("active", on_dic_structural_change)
for slider in (dic_sigma_slider, dic_threshold_mult_slider, dic_close_radius_slider, dic_open_radius_slider):
    slider.on_change("value_throttled", on_dic_structural_change)
for slider in (dic_min_area_slider, dic_length_range_slider, dic_width_range_slider,
               dic_aspect_range_slider, dic_min_solidity_slider):
    slider.on_change("value", on_dic_filter_change)

dic_instructions = Div(text=(
    "<p>DIC segmentation is a property of the whole FOV, not one cell -- these sliders (and the "
    "parameters saved for this tab) are keyed <b>per FOV</b>, not per cell; see the app docstring. "
    "<b>Structural</b> parameters (background correction, sigma, Otsu multiplier, closing/opening "
    "radius) re-run the segmentation on release; <b>filter</b> parameters (area, length/width/aspect "
    "ranges, solidity) just re-check the already-segmented components, live while dragging. Panel c "
    "colors every evaluated component green (passes all filters) or red (fails at least one) -- hover "
    "for exact numbers -- and outlines the CURRENT cell (the one shown in tabs 2/3) in blue. Panel a "
    "always shows the raw DIC image, a fixed reference regardless of the correction toggle.</p>"
), sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

dic_structural_col = column(
    Div(text="<b>Structural (expensive -- updates on release)</b>", styles=WHITE_STYLE),
    dic_correction_toggle,
    dic_sigma_slider, dic_threshold_mult_slider, dic_close_radius_slider, dic_open_radius_slider,
    styles=WHITE_STYLE,
)
dic_filter_col = column(
    Div(text="<b>Filter (cheap -- updates live)</b>", styles=WHITE_STYLE),
    dic_min_area_slider, dic_length_range_slider, dic_width_range_slider,
    dic_aspect_range_slider, dic_min_solidity_slider,
    styles=WHITE_STYLE,
)

dic_tab_layout = column(
    dic_instructions,
    row(dic_reset_button),
    dic_status_div,
    dic_warning_div,
    row(dic_structural_col, dic_filter_col, sizing_mode="stretch_width"),
    row(dic_a_fig, dic_b_fig, dic_c_fig, sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
    styles=WHITE_STYLE,
)

# ---------------------------------------------------------------------------
# Tabs 2/3: Chlorophyll / BODIPY five-panel view, factory-built since both
# channels share identical structure (see blob_counting_params_explorer_app.py).
# ---------------------------------------------------------------------------


def build_channel_tab(channel, include_focus):
    defaults = CHANNEL_DEFAULTS[channel]
    shared_x_range = Range1d(0, 1)
    shared_y_range = Range1d(0, 1)

    def make_panel(title):
        fig = figure(
            title=title, width=CHANNEL_PANEL_W, height=CHANNEL_PANEL_W,
            # "fixed", not "scale_width" -- see make_dic_panel's comment above; render_static
            # already recomputes exact fig.width/fig.height per cell crop below, so "fixed"
            # just means Bokeh renders at those exact pixels instead of stretching further.
            sizing_mode="fixed",
            x_range=shared_x_range, y_range=shared_y_range,
            tools="pan,wheel_zoom,reset",
            background_fill_color="white", border_fill_color="white",
        )
        fig.axis.visible = False
        fig.grid.visible = False
        fig.toolbar.logo = None
        return fig

    dic_fig = make_panel("a: DIC + outline")
    raw_fig = make_panel(f"b: {channel} raw")
    smooth_fig = make_panel(f"c: {channel} smoothed")
    mask_fig = make_panel("d: bright mask")
    result_fig = make_panel("e: counted/too small")
    all_figs = (dic_fig, raw_fig, smooth_fig, mask_fig, result_fig)

    dic_src = ColumnDataSource(data=dict(image=[np.zeros((1, 1))], dw=[1], dh=[1]))
    raw_src = ColumnDataSource(data=dict(image=[np.zeros((1, 1))], dw=[1], dh=[1]))
    smooth_src = ColumnDataSource(data=dict(image=[np.zeros((1, 1))], dw=[1], dh=[1]))
    mask_src = ColumnDataSource(data=dict(image=[np.zeros((1, 1))], dw=[1], dh=[1]))
    cell_outline_src = ColumnDataSource(data=dict(xs=[], ys=[]))
    peaks_src = ColumnDataSource(data=dict(x=[], y=[]))
    skeleton_src = ColumnDataSource(data=dict(x=[], y=[]))
    skeleton_tips_src = ColumnDataSource(data=dict(x=[], y=[]))

    dic_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)
    channel_mapper = LinearColorMapper(palette="Viridis256", low=0, high=1)
    mask_mapper = LinearColorMapper(palette="Greys256", low=0, high=1)

    dic_fig.image(image="image", x=0, y=0, dw="dw", dh="dh", source=dic_src, color_mapper=dic_mapper)
    dic_fig.line(x="xs", y="ys", source=cell_outline_src, color="#3366CC", line_width=2)
    dic_fig.scatter(x="x", y="y", source=skeleton_src, marker="circle", size=3, fill_color="cyan", line_color=None)
    dic_fig.scatter(x="x", y="y", source=skeleton_tips_src, marker="triangle", size=12, fill_color="lime", line_color="black")
    dic_fig.scatter(x="x", y="y", source=peaks_src, marker="x", size=10, line_color="red", line_width=2)
    raw_fig.image(image="image", x=0, y=0, dw="dw", dh="dh", source=raw_src, color_mapper=channel_mapper)
    smooth_fig.image(image="image", x=0, y=0, dw="dw", dh="dh", source=smooth_src, color_mapper=channel_mapper)
    mask_fig.image(image="image", x=0, y=0, dw="dw", dh="dh", source=mask_src, color_mapper=mask_mapper)
    mask_fig.scatter(x="x", y="y", source=peaks_src, marker="x", size=10, line_color="red", line_width=2)
    result_fig.scatter(x="x", y="y", source=peaks_src, marker="x", size=10, line_color="black", line_width=2)

    blobs_src = ColumnDataSource(data=dict(
        xs=[], ys=[], fill_color=[], line_color=[], size_px=[], size_um2=[], status=[],
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

    registration_toggle = RadioButtonGroup(
        labels=["Registration correction (default)", "No correction (raw channel)"],
        active=0, width=300,
    )
    connectivity_toggle = RadioButtonGroup(
        labels=["8-connected (default)", "4-connected"], active=0, width=240,
    )
    method_toggle = RadioButtonGroup(
        labels=["Simple threshold", "Watershed (default)", "Skeleton-clustered (prototype)"],
        active=["threshold", "watershed", "skeleton"].index(defaults["method"]), width=360,
    )
    sigma_slider = Slider(title="Smoothing sigma (px)", start=0.0, end=5.0,
                           value=defaults["sigma"], step=0.1, width=240)
    threshold_mult_slider = Slider(title="Otsu threshold multiplier", start=0.2, end=3.0,
                                    value=defaults["threshold_mult"], step=0.05, width=240)
    watershed_min_distance_slider = Slider(title="Watershed minimum peak distance (px)", start=1, end=30,
                                            value=defaults["min_distance"], step=1, width=240)
    watershed_min_prominence_slider = Slider(title="Watershed minimum peak prominence (intensity units)",
                                              start=0, end=1500, value=defaults["min_prominence"], step=10, width=240)
    skeleton_cluster_gap_slider = Slider(title="Skeleton cluster gap (px, prototype)", start=1, end=60,
                                          value=defaults["cluster_gap"], step=1, width=240)
    min_size_slider = Slider(title="Minimum blob size (px^2)", start=BLOB_FLOOR_PX, end=100,
                              value=defaults["min_size"], step=1, width=240)

    status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)
    warning_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)
    skeleton_info_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)
    reset_button = Button(label=f"Reset {channel} tab to calibrated defaults", button_type="primary", width=200)

    if include_focus:
        focus_presmooth_slider = Slider(title="Pre-Laplacian smoothing sigma (px, 0 = production default)",
                                         start=0.0, end=5.0, value=defaults["focus_presmooth_sigma"], step=0.1, width=240)
        focus_threshold_slider = Slider(title="Focus score threshold (in_focus cutoff)",
                                         start=0, end=3000, value=defaults["focus_threshold"], step=10, width=240)
        focus_status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)
    else:
        focus_presmooth_slider = focus_threshold_slider = focus_status_div = None

    last_result = {"n_counted": 0, "focus_score": None, "in_focus": None}

    def registration_value():
        return registration_toggle.active == 0

    def connectivity_value():
        return 8 if connectivity_toggle.active == 0 else 4

    def method_value():
        return ["threshold", "watershed", "skeleton"][method_toggle.active]

    def current_geometry():
        fov_idx, cell_idx = current_fov_idx[0], current_cell_idx[0]
        return compute_blob_geometry(
            fov_idx, cell_idx, channel, sigma_slider.value, threshold_mult_slider.value,
            connectivity_value(), method_value(), watershed_min_distance_slider.value,
            watershed_min_prominence_slider.value, skeleton_cluster_gap_slider.value, registration_value(),
        )

    def apply_filters_and_render():
        fov_idx, cell_idx = current_fov_idx[0], current_cell_idx[0]
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
        last_result["n_counted"] = n_counted

        (prefix, fov_num), _paths = fov_items[fov_idx]
        noun = "plastid" if channel == "Chlorophyll" else "lipid body"
        noun_plural = "plastids" if channel == "Chlorophyll" else "lipid bodies"
        warning_div.text = (
            f"<b>Warning:</b> {geom['n_raw_blobs']} raw bright blob(s) exceeded the "
            f"{BLOB_MAX_COUNT} cap -- only the largest {BLOB_MAX_COUNT} by size were evaluated."
            if geom["truncated"] else ""
        )
        status_div.text = (
            f"<b>{prefix}FOV{fov_num}</b> cell {cell_idx + 1} &mdash; channel: <b>{channel}</b> "
            f"&mdash; registration correction: <b>{'ON' if registration_value() else 'OFF'}</b> "
            f"&mdash; per-cell Otsu threshold (smoothed): {geom['thresh']:.1f} &times; {threshold_mult_slider.value:.2f} "
            f"&mdash; {geom['n_raw_blobs']} raw bright blob(s) &mdash; "
            f"<b>{n_counted} {noun if n_counted == 1 else noun_plural}</b> "
            f"(&ge; {min_size:.0f}px, {connectivity}-connected, method: {method}, "
            f"min prominence: {watershed_min_prominence_slider.value:.0f})"
        )

        if method == "skeleton" and len(geom["peak_ys"]):
            n_clusters = len(set(geom["cluster_ids"].tolist()))
            rows_txt = "; ".join(
                f"peak{i + 1}: arc={arc:.0f}px side={side:+.0f} &rarr; cluster {cid}"
                for i, (arc, side, cid) in enumerate(zip(geom["arc_lens"], geom["sides"], geom["cluster_ids"]))
            )
            skeleton_info_div.text = (
                f"<b>Skeleton clustering:</b> {len(geom['peak_ys'])} raw peak(s) &rarr; "
                f"<b>{n_clusters} cluster(s)</b> (gap={skeleton_cluster_gap_slider.value:.0f}px). {rows_txt}"
            )
        else:
            skeleton_info_div.text = ""

        if include_focus:
            score = compute_focus(fov_idx, cell_idx, registration_value(), focus_presmooth_slider.value)
            in_focus = score >= focus_threshold_slider.value
            last_result["focus_score"] = score
            last_result["in_focus"] = in_focus
            focus_status_div.text = (
                f"<b>Focus score</b> (variance of Laplacian, pre-smoothing sigma={focus_presmooth_slider.value:.1f}): "
                f"<b>{score:.1f}</b> &mdash; threshold {focus_threshold_slider.value:.0f} &mdash; "
                f"<b>{'IN FOCUS' if in_focus else 'OUT OF FOCUS'}</b>"
            )

    def render_static():
        fov_idx, cell_idx = current_fov_idx[0], current_cell_idx[0]
        geom = current_geometry()
        r0, r1, c0, c1 = geom["crop_bounds"]
        crop_h, crop_w = r1 - r0, c1 - c0
        do_rotate = crop_w > crop_h
        disp_h, disp_w = (crop_w, crop_h) if do_rotate else (crop_h, crop_w)

        if disp_w >= disp_h:
            fig_w, fig_h = CHANNEL_PANEL_W, max(1, round(CHANNEL_PANEL_W * disp_h / disp_w))
        else:
            fig_w, fig_h = max(1, round(CHANNEL_PANEL_W * disp_w / disp_h)), CHANNEL_PANEL_W
        for fig in all_figs:
            fig.width, fig.height = fig_w, fig_h

        shared_x_range.start, shared_x_range.end = 0, disp_w
        shared_y_range.start, shared_y_range.end = 0, disp_h

        dic_corr = load_dic_corrected(fov_idx)
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
        channel_mapper.low, channel_mapper.high = float(raw_crop.min()), float(raw_crop.max())

        mask_crop_disp = rotate_crop_for_display(geom["bright_mask_crop"], do_rotate)
        mask_src.data = dict(image=[np.flipud(mask_crop_disp)], dw=[disp_w], dh=[disp_h])

        if len(geom["peak_ys"]):
            peak_r, peak_c = rotate_point_for_display(geom["peak_ys"] - r0, geom["peak_xs"] - c0, crop_w, do_rotate)
            peaks_src.data = dict(x=peak_c.tolist(), y=height_to_bokeh_y(peak_r, disp_h).tolist())
        else:
            peaks_src.data = dict(x=[], y=[])

        if method_value() == "skeleton":
            skel_info = compute_cell_skeleton(fov_idx, cell_idx)
            skel_ys_local, skel_xs_local = np.where(skel_info["skel"])
            skel_r_full = skel_ys_local + skel_info["y0"] - 1
            skel_c_full = skel_xs_local + skel_info["x0"] - 1
            skel_r, skel_c = rotate_point_for_display(skel_r_full - r0, skel_c_full - c0, crop_w, do_rotate)
            skeleton_src.data = dict(x=skel_c.tolist(), y=height_to_bokeh_y(skel_r, disp_h).tolist())

            tip_rs_full = np.array([skel_info["tip_a"][0], skel_info["tip_b"][0]]) + skel_info["y0"] - 1
            tip_cs_full = np.array([skel_info["tip_a"][1], skel_info["tip_b"][1]]) + skel_info["x0"] - 1
            tip_r, tip_c = rotate_point_for_display(tip_rs_full - r0, tip_cs_full - c0, crop_w, do_rotate)
            skeleton_tips_src.data = dict(x=tip_c.tolist(), y=height_to_bokeh_y(tip_r, disp_h).tolist())
        else:
            skeleton_src.data = dict(x=[], y=[])
            skeleton_tips_src.data = dict(x=[], y=[])

    def read_state():
        state = dict(
            registration=registration_value(), connectivity=connectivity_value(), method=method_value(),
            sigma=sigma_slider.value, threshold_mult=threshold_mult_slider.value,
            min_distance=watershed_min_distance_slider.value, min_prominence=watershed_min_prominence_slider.value,
            cluster_gap=skeleton_cluster_gap_slider.value, min_size=min_size_slider.value,
            n_blobs=last_result["n_counted"],
        )
        if include_focus:
            state["focus_presmooth_sigma"] = focus_presmooth_slider.value
            state["focus_threshold"] = focus_threshold_slider.value
            state["focus_score"] = last_result["focus_score"]
            state["in_focus"] = last_result["in_focus"]
        return state

    def apply_state(state):
        registration_toggle.active = 0 if state.get("registration", defaults["registration"]) else 1
        connectivity_toggle.active = 0 if state.get("connectivity", defaults["connectivity"]) == 8 else 1
        method_toggle.active = ["threshold", "watershed", "skeleton"].index(state.get("method", defaults["method"]))
        sigma_slider.value = state.get("sigma", defaults["sigma"])
        threshold_mult_slider.value = state.get("threshold_mult", defaults["threshold_mult"])
        watershed_min_distance_slider.value = state.get("min_distance", defaults["min_distance"])
        watershed_min_prominence_slider.value = state.get("min_prominence", defaults["min_prominence"])
        skeleton_cluster_gap_slider.value = state.get("cluster_gap", defaults["cluster_gap"])
        min_size_slider.value = state.get("min_size", defaults["min_size"])
        if include_focus:
            focus_presmooth_slider.value = state.get("focus_presmooth_sigma", defaults["focus_presmooth_sigma"])
            focus_threshold_slider.value = state.get("focus_threshold", defaults["focus_threshold"])

    def reset():
        apply_state(defaults)
        render_static()
        apply_filters_and_render()

    def on_structural_change(attr, old, new):
        render_static()
        apply_filters_and_render()

    def on_filter_change(attr, old, new):
        apply_filters_and_render()

    reset_button.on_click(reset)
    registration_toggle.on_change("active", on_structural_change)
    connectivity_toggle.on_change("active", on_structural_change)
    method_toggle.on_change("active", on_structural_change)
    for slider in (sigma_slider, threshold_mult_slider, watershed_min_distance_slider,
                   watershed_min_prominence_slider, skeleton_cluster_gap_slider):
        slider.on_change("value_throttled", on_structural_change)
    min_size_slider.on_change("value", on_filter_change)
    if include_focus:
        focus_presmooth_slider.on_change("value_throttled", on_structural_change)
        focus_threshold_slider.on_change("value", on_filter_change)

    instructions = Div(text=(
        f"<p><b>{channel}</b> tab -- fixed to the {channel} channel; parameters here are saved <b>per "
        "cell</b>. <b>Registration correction</b> translates the fluorescence channel to align with DIC "
        "before anything else runs (matches quantify_cells_shifted.py). Production's own method is "
        f"<b>watershed</b>; <b>Simple threshold</b> is kept for comparison. This tab's own starting default "
        f"is <b>{['Simple threshold', 'Watershed', 'Watershed, skeleton-clustered (prototype)'][['threshold', 'watershed', 'skeleton'].index(defaults['method'])]}</b> "
        f"with minimum peak prominence {defaults['min_prominence']:.0f} -- a starting point for manual "
        "estimation in this review tool, not necessarily production's own setting. "
        "<b>Structural</b> parameters re-run thresholding/labeling on release; the <b>filter</b> parameter "
        "(minimum blob size) just re-checks already-labeled blobs, live while dragging. Panel e colors "
        "every blob green (counted) or red (too small) -- hover for its exact size. Panel a shows the DIC "
        "channel with this cell's mask outline for orientation only." +
        (" The focus-score sliders below test whether pre-smoothing before the Laplacian (0 = production's "
         "raw/unsmoothed default) changes which cells clear the in-focus threshold." if include_focus else "") +
        "</p>"
    ), sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

    structural_children = [
        Div(text="<b>Structural (expensive -- updates on release)</b>", styles=WHITE_STYLE),
        registration_toggle, connectivity_toggle, method_toggle,
        sigma_slider, threshold_mult_slider, watershed_min_distance_slider, watershed_min_prominence_slider,
        skeleton_cluster_gap_slider,
    ]
    structural_col = column(*structural_children, styles=WHITE_STYLE)
    filter_col = column(
        Div(text="<b>Filter (cheap -- updates live)</b>", styles=WHITE_STYLE),
        min_size_slider,
        styles=WHITE_STYLE,
    )

    focus_col = None
    if include_focus:
        focus_col = column(
            Div(text="<b>Focus-score testing (Chlorophyll only)</b>", styles=WHITE_STYLE),
            focus_presmooth_slider, focus_threshold_slider, focus_status_div,
            styles=WHITE_STYLE,
        )

    controls_row = row(structural_col, filter_col, sizing_mode="stretch_width") if focus_col is None else \
        row(structural_col, filter_col, focus_col, sizing_mode="stretch_width")

    layout = column(
        instructions,
        row(reset_button),
        status_div,
        warning_div,
        skeleton_info_div,
        controls_row,
        row(dic_fig, raw_fig, smooth_fig, mask_fig, result_fig, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
        styles=WHITE_STYLE,
    )

    return dict(
        layout=layout, render_static=render_static, apply_filters_and_render=apply_filters_and_render,
        read_state=read_state, apply_state=apply_state, reset=reset,
    )


chl_tab = build_channel_tab("Chlorophyll", include_focus=True)
bod_tab = build_channel_tab("BODIPY", include_focus=False)

# ---------------------------------------------------------------------------
# Navigation, commit/restore, note field, CSV save/export
# ---------------------------------------------------------------------------

fov_options = [
    (str(i), f"{prefix}FOV{fov_num} ({i + 1}/{len(fov_items)})")
    for i, ((prefix, fov_num), _paths) in enumerate(fov_items)
]
fov_select = Select(title="Field of view", options=fov_options, value="0", width=200)
cell_select = Select(title="Cell in this FOV", options=[("0", "-")], value="0", width=140)
prev_button = Button(label="< Previous cell", width=115)
next_button = Button(label="Next cell >", width=115)
overall_status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

note_input = TextAreaInput(title="Note for this cell", value="", rows=3,
                            sizing_mode="stretch_width", styles=WHITE_STYLE)

csv_name_input = TextInput(title="Output CSV filename", value=DEFAULT_CSV_NAME, width=240)
export_button = Button(label="Export CSV", button_type="primary", width=120)
export_status_div = Div(text="", sizing_mode="stretch_width", styles=WHITE_STYLE)
load_status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

_started = [False]


def csv_path():
    return os.path.join(OUTPUT_DIR, csv_name_input.value.strip() or DEFAULT_CSV_NAME)


def commit_current():
    if not _started[0]:
        return
    fov_key = current_fov_key()
    cell_key = current_cell_key()
    dic_state_by_fov[fov_key] = read_dic_state()
    cell_state[cell_key] = dict(
        chlorophyll=chl_tab["read_state"](),
        bodipy=bod_tab["read_state"](),
        note=note_input.value,
    )


def restore_for(fov_idx, cell_idx):
    fov_key, _ = fov_items[fov_idx]
    cell_id = cell_idx + 1
    cell_key = (fov_key[0], fov_key[1], cell_id)

    apply_dic_state(dic_state_by_fov.get(fov_key, DIC_DEFAULTS))
    saved = cell_state.get(cell_key)
    chl_tab["apply_state"](saved["chlorophyll"] if saved else CHANNEL_DEFAULTS["Chlorophyll"])
    bod_tab["apply_state"](saved["bodipy"] if saved else CHANNEL_DEFAULTS["BODIPY"])
    note_input.value = saved["note"] if saved else ""


def render_all():
    render_dic_static()
    render_dic_mask()
    apply_dic_filters_and_render()
    chl_tab["render_static"]()
    chl_tab["apply_filters_and_render"]()
    bod_tab["render_static"]()
    bod_tab["apply_filters_and_render"]()


def show_cell(fov_idx, cell_idx):
    fov_idx = max(0, min(len(fov_items) - 1, fov_idx))
    n_cells = len(get_cells_for_fov(fov_idx))
    if n_cells == 0:
        return
    cell_idx = max(0, min(n_cells - 1, cell_idx))

    commit_current()
    current_fov_idx[0], current_cell_idx[0] = fov_idx, cell_idx
    restore_for(fov_idx, cell_idx)

    fov_select.value = str(fov_idx)
    cell_select.options = [(str(i), f"Cell {i + 1} of {n_cells}") for i in range(n_cells)]
    cell_select.value = str(cell_idx)

    render_all()

    prefix, fov_num = current_fov_key()
    cell_key = current_cell_key()
    reviewed = " -- already reviewed (values restored)" if cell_key in cell_state else " -- not yet reviewed"
    overall_status_div.text = (
        f"<b>{prefix}FOV{fov_num}</b> ({fov_idx + 1}/{len(fov_items)}) &mdash; "
        f"cell {cell_idx + 1} of {n_cells}{reviewed}"
    )


def first_fov_with_cells(start, step):
    idx = start
    while 0 <= idx < len(fov_items):
        if len(get_cells_for_fov(idx)) > 0:
            return idx
        idx += step
    return None


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
    if fov_idx == current_fov_idx[0]:
        return
    if len(get_cells_for_fov(fov_idx)) == 0:
        target = first_fov_with_cells(fov_idx, 1) or first_fov_with_cells(fov_idx, -1)
        if target is not None:
            show_cell(target, 0)
        return
    show_cell(fov_idx, 0)


def on_cell_select(attr, old, new):
    if int(new) == current_cell_idx[0]:
        return
    show_cell(current_fov_idx[0], int(new))


prev_button.on_click(on_prev)
next_button.on_click(on_next)
fov_select.on_change("value", on_fov_select)
cell_select.on_change("value", on_cell_select)

# ---------------------------------------------------------------------------
# CSV export / load-back
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "sample", "fov", "cell_id",
    "dic_background_correction", "dic_sigma", "dic_threshold_mult",
    "dic_close_radius_px", "dic_open_radius_px", "dic_min_area_px",
    "dic_length_min_um", "dic_length_max_um", "dic_width_min_um", "dic_width_max_um",
    "dic_aspect_min", "dic_aspect_max", "dic_min_solidity",
    "chl_registration", "chl_connectivity", "chl_method", "chl_sigma", "chl_threshold_mult",
    "chl_watershed_min_distance_px", "chl_watershed_min_prominence", "chl_cluster_gap_px", "chl_min_size_px",
    "chl_focus_presmooth_sigma", "chl_focus_threshold", "chl_n_plastids", "chl_focus_score", "chl_in_focus",
    "bod_registration", "bod_connectivity", "bod_method", "bod_sigma", "bod_threshold_mult",
    "bod_watershed_min_distance_px", "bod_watershed_min_prominence", "bod_cluster_gap_px", "bod_min_size_px",
    "bod_n_lipid_bodies",
    "note",
]


def build_export_rows():
    rows = []
    for (prefix, fov_num, cell_id), state in sorted(cell_state.items()):
        dic_vals = dic_state_by_fov.get((prefix, fov_num), DIC_DEFAULTS)
        chl, bod = state.get("chlorophyll", {}), state.get("bodipy", {})
        rows.append(dict(
            sample=prefix, fov=fov_num, cell_id=cell_id,
            dic_background_correction=dic_vals["correction"], dic_sigma=dic_vals["sigma"],
            dic_threshold_mult=dic_vals["threshold_mult"],
            dic_close_radius_px=dic_vals["close_r"], dic_open_radius_px=dic_vals["open_r"],
            dic_min_area_px=dic_vals["min_area"],
            dic_length_min_um=dic_vals["length_range"][0], dic_length_max_um=dic_vals["length_range"][1],
            dic_width_min_um=dic_vals["width_range"][0], dic_width_max_um=dic_vals["width_range"][1],
            dic_aspect_min=dic_vals["aspect_range"][0], dic_aspect_max=dic_vals["aspect_range"][1],
            dic_min_solidity=dic_vals["min_solidity"],
            chl_registration=chl.get("registration"), chl_connectivity=chl.get("connectivity"),
            chl_method=chl.get("method"), chl_sigma=chl.get("sigma"), chl_threshold_mult=chl.get("threshold_mult"),
            chl_watershed_min_distance_px=chl.get("min_distance"), chl_watershed_min_prominence=chl.get("min_prominence"),
            chl_cluster_gap_px=chl.get("cluster_gap"), chl_min_size_px=chl.get("min_size"),
            chl_focus_presmooth_sigma=chl.get("focus_presmooth_sigma"), chl_focus_threshold=chl.get("focus_threshold"),
            chl_n_plastids=chl.get("n_blobs"), chl_focus_score=chl.get("focus_score"), chl_in_focus=chl.get("in_focus"),
            bod_registration=bod.get("registration"), bod_connectivity=bod.get("connectivity"),
            bod_method=bod.get("method"), bod_sigma=bod.get("sigma"), bod_threshold_mult=bod.get("threshold_mult"),
            bod_watershed_min_distance_px=bod.get("min_distance"), bod_watershed_min_prominence=bod.get("min_prominence"),
            bod_cluster_gap_px=bod.get("cluster_gap"), bod_min_size_px=bod.get("min_size"),
            bod_n_lipid_bodies=bod.get("n_blobs"),
            note=state.get("note", ""),
        ))
    return rows


def on_export():
    commit_current()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = build_export_rows()
    if not rows:
        export_status_div.text = "No reviewed cells yet -- nothing to export."
        return
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path(), index=False)
    export_status_div.text = f"Exported {len(rows)} cell row(s) to {csv_path()}"


def _clean(value):
    return None if value == "" else value


def load_existing_csv():
    path = csv_path()
    if not os.path.exists(path):
        load_status_div.text = f"No existing file at {path}."
        return
    df = pd.read_csv(path, keep_default_na=False)
    for _, r in df.iterrows():
        fov_key = (str(r["sample"]), int(r["fov"]))
        cell_key = (fov_key[0], fov_key[1], int(r["cell_id"]))
        dic_state_by_fov[fov_key] = dict(
            correction=bool(r["dic_background_correction"]), sigma=float(r["dic_sigma"]),
            threshold_mult=float(r["dic_threshold_mult"]),
            close_r=int(float(r["dic_close_radius_px"])), open_r=int(float(r["dic_open_radius_px"])),
            min_area=float(r["dic_min_area_px"]),
            length_range=(float(r["dic_length_min_um"]), float(r["dic_length_max_um"])),
            width_range=(float(r["dic_width_min_um"]), float(r["dic_width_max_um"])),
            aspect_range=(float(r["dic_aspect_min"]), float(r["dic_aspect_max"])),
            min_solidity=float(r["dic_min_solidity"]),
        )
        cell_state[cell_key] = dict(
            chlorophyll=dict(
                registration=bool(r["chl_registration"]), connectivity=int(r["chl_connectivity"]),
                method=str(r["chl_method"]), sigma=float(r["chl_sigma"]), threshold_mult=float(r["chl_threshold_mult"]),
                min_distance=float(r["chl_watershed_min_distance_px"]),
                min_prominence=float(r["chl_watershed_min_prominence"]),
                cluster_gap=float(r["chl_cluster_gap_px"]), min_size=float(r["chl_min_size_px"]),
                focus_presmooth_sigma=float(r["chl_focus_presmooth_sigma"]),
                focus_threshold=float(r["chl_focus_threshold"]),
                n_blobs=_clean(r["chl_n_plastids"]) and int(r["chl_n_plastids"]),
                focus_score=_clean(r["chl_focus_score"]) and float(r["chl_focus_score"]),
                in_focus=_clean(r["chl_in_focus"]) and bool(r["chl_in_focus"]),
            ),
            bodipy=dict(
                registration=bool(r["bod_registration"]), connectivity=int(r["bod_connectivity"]),
                method=str(r["bod_method"]), sigma=float(r["bod_sigma"]), threshold_mult=float(r["bod_threshold_mult"]),
                min_distance=float(r["bod_watershed_min_distance_px"]),
                min_prominence=float(r["bod_watershed_min_prominence"]),
                cluster_gap=float(r["bod_cluster_gap_px"]), min_size=float(r["bod_min_size_px"]),
                n_blobs=_clean(r["bod_n_lipid_bodies"]) and int(r["bod_n_lipid_bodies"]),
            ),
            note=str(r.get("note", "")),
        )
    load_status_div.text = f"Loaded {len(df)} cell row(s) from {path}."


def on_csv_filename_change(attr, old, new):
    load_existing_csv()
    restore_for(current_fov_idx[0], current_cell_idx[0])
    render_all()


export_button.on_click(on_export)
csv_name_input.on_change("value", on_csv_filename_change)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

instructions = Div(text=(
    "<p>Walk the dataset cell by cell (Prev/Next roll into the neighboring FOV once you run off either "
    "end) and estimate parameters on each of the three tabs below. Navigating to another cell "
    "auto-commits the current sliders into memory -- Chlorophyll/BODIPY parameters and the note are "
    "saved per cell, DIC parameters are saved per FOV (DIC segmentation runs once per FOV; see the app "
    "docstring). Revisiting an already-committed cell/FOV restores those values instead of resetting to "
    "calibrated defaults. Nothing is written to disk until you click <b>Export CSV</b>; the CSV filename "
    "below is checked for an existing file at startup and whenever changed, and its contents are loaded "
    "back into memory (not wiping out whatever's already committed in this session).</p>"
), sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

tabs = Tabs(tabs=[
    TabPanel(child=dic_tab_layout, title="1. DIC segmentation"),
    TabPanel(child=chl_tab["layout"], title="2. Chlorophyll (plastids)"),
    TabPanel(child=bod_tab["layout"], title="3. BODIPY (lipid bodies)"),
])

layout = column(
    instructions,
    row(prev_button, next_button, fov_select, cell_select, sizing_mode="stretch_width"),
    overall_status_div,
    tabs,
    note_input,
    row(csv_name_input, export_button, sizing_mode="stretch_width"),
    load_status_div,
    export_status_div,
    sizing_mode="stretch_width",
    styles=dict(WHITE_STYLE, padding="12px"),
)

_start_fov = first_fov_with_cells(0, 1)
if _start_fov is None:
    raise SystemExit("No accepted cells found in any FOV -- nothing to review.")

load_existing_csv()
show_cell(_start_fov, 0)
_started[0] = True

curdoc().add_root(layout)
curdoc().title = "Per-cell parameter review (DIC / Chlorophyll / BODIPY)"
