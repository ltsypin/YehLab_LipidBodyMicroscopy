#!/usr/bin/env python3
"""
Interactive Bokeh server app for visually reviewing quantify_cells.py's
segmentation, FOV by FOV, flagging cells whose segmentation looks wrong,
marking good cells the pipeline missed, and sanity-checking the Step 4
fluorescence-to-DIC registration correction.

Panels: a=DIC, b=Chlorophyll, c=BODIPY, d=Chlorophyll+BODIPY overlay (same
global per-channel normalization as composite_figure.py, computed once over
the whole input directory), e=DIC with each accepted cell's mask boundary
drawn as a clickable region, f=registration-corrected Chlorophyll+BODIPY
overlay with the same (DIC-derived) mask boundaries drawn on top, to check
that corrected fluorescence signal actually falls inside each outline.

Panel e: click a cell's outline to flag it as poorly segmented (click again
to unflag); flagged cells turn red. Use the Freehand Draw tool (toolbar
icon) to lasso a good cell the pipeline missed -- drag to trace an outline;
use the "Remove ROI" button next to its note field to delete one drawn
poorly. Both flagged cells and missed-cell ROIs get a note field below the
panels ("why did you select this?"), and "Export ROIs" writes everything
(with notes) to <output_dir>/<flagged filename> and
<output_dir>/<missed filename> -- both filenames are set in the app itself
(defaulting to flagged_rois.csv/missed_cell_rois.csv) and are checked for
an existing file to load back in, both at startup and whenever changed.

Segmentation reuses quantify_cells.segment_dic/accepted_cells/
count_lipid_bodies/compute_dic_background/correct_dic_background verbatim --
this app never re-implements or approximates the pipeline, so what you
review here is exactly what the CSVs contain. (Previously this app called
segment_dic on the raw, uncorrected DIC image, so its cell population could
disagree with quantify_cells.py's own main() wherever the fixed-pattern
background correction matters -- fixed to apply the same correction first.)

Run:
    bokeh serve --show qc_review_app.py --args <input_dir> [<output_dir>]

Defaults to ./renamed_composites and ./quantification if omitted.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
from skimage.measure import find_contours

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantify_cells import (
    CHANNELS, segment_dic, accepted_cells, count_lipid_bodies, LIPID_SMOOTH_SIGMA,
    compute_dic_background, correct_dic_background,
)
from quantify_cells_shifted import SHIFT_DY, SHIFT_DX
from composite_figure import (
    find_channel_files, compute_global_ranges, normalize, to_rgb, group_fovs,
    SCALE_BAR_PX, SCALE_BAR_UM,
)

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Button, Select, Div, HoverTool, TapTool, TextAreaInput, TextInput,
    FreehandDrawTool, Range1d,
)
from bokeh.plotting import figure

INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "renamed_composites"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "quantification"
DEFAULT_FLAGGED_NAME = "flagged_rois.csv"
DEFAULT_MISSED_NAME = "missed_cell_rois.csv"

FLAGGED_FILL, FLAGGED_LINE = "#E24B4A", "#A32D2D"
OK_FILL, OK_LINE = "#5DCAA5", "#0F6E56"
REG_OUTLINE = "#FAC775"
MISSED_FILL, MISSED_LINE = "#F0997B", "#993C1D"

WHITE_STYLE = {"background-color": "white"}
FULL_WIDTH_TEXT_STYLE = {"background-color": "white", "margin": "0"}

# ---------------------------------------------------------------------------
# Data loading: reuse the pipeline's own functions so review == what's in the CSVs.
# ---------------------------------------------------------------------------

print(f"Scanning {INPUT_DIR} ...")
files_by_channel = find_channel_files(INPUT_DIR)
missing = [c for c in CHANNELS if c not in files_by_channel]
if missing:
    raise SystemExit(f"No files found for channel(s): {missing} in {INPUT_DIR}")
print("Computing global per-channel intensity ranges (same as composite_figure.py)...")
GLOBAL_RANGES = compute_global_ranges(files_by_channel)
for channel, (vmin, vmax) in GLOBAL_RANGES.items():
    print(f"  {channel}: min={vmin}, max={vmax}")

all_fovs = group_fovs(INPUT_DIR)
fov_items = sorted(all_fovs.items())
fov_items = [(key, paths) for key, paths in fov_items if all(c in paths for c in CHANNELS)]
if not fov_items:
    raise SystemExit(f"No complete FOVs (all 3 channels) found in {INPUT_DIR}")
print(f"Found {len(fov_items)} complete FOVs.")

# Same fixed-pattern background correction quantify_cells.py's own main() and
# quantify_cells_shifted.py apply before segment_dic -- always estimated from every
# DIC image in INPUT_DIR (not just the complete-triplet FOVs above), matching those
# scripts exactly, since it's estimating a stationary instrument artifact, not
# anything tied to which channels a given FOV happens to have.
_dic_paths_for_background = [paths["DIC"] for paths in all_fovs.values() if "DIC" in paths]
DIC_BACKGROUND = compute_dic_background(_dic_paths_for_background)
print(f"Computed DIC background from {len(_dic_paths_for_background)} FOV(s) in {INPUT_DIR}")

_cache = {}
flagged_registry = {}      # (prefix, fov_num, cell_id) -> measurement dict (incl. "note")
missed_rois_registry = {}  # (prefix, fov_num) -> dict(xs=[[...], ...], ys=[[...], ...], note=[...])


def height_to_bokeh_y(row_coords, height):
    return height - row_coords


def load_fov(idx):
    if idx in _cache:
        return _cache[idx]

    (prefix, fov_num), paths = fov_items[idx]
    dic = tifffile.imread(paths["DIC"])
    chl = tifffile.imread(paths["Chlorophyll"]).astype(np.float64)
    bod = tifffile.imread(paths["BODIPY"]).astype(np.float64)
    height, width = dic.shape

    dic_norm = normalize(dic, *GLOBAL_RANGES["DIC"])
    chl_norm = normalize(chl, *GLOBAL_RANGES["Chlorophyll"])
    bod_norm = normalize(bod, *GLOBAL_RANGES["BODIPY"])

    dic_rgb = to_rgb(dic_norm, "gray")
    chl_rgb = to_rgb(chl_norm, "magenta")
    bod_rgb = to_rgb(bod_norm, "cyan")
    overlay_rgb = np.clip(chl_rgb + bod_rgb, 0.0, 1.0)

    chl_corr = ndi.shift(chl, shift=(SHIFT_DY, SHIFT_DX), order=1)
    bod_corr = ndi.shift(bod, shift=(SHIFT_DY, SHIFT_DX), order=1)
    chl_corr_norm = normalize(chl_corr, *GLOBAL_RANGES["Chlorophyll"])
    bod_corr_norm = normalize(bod_corr, *GLOBAL_RANGES["BODIPY"])
    reg_overlay_rgb = np.clip(to_rgb(chl_corr_norm, "magenta") + to_rgb(bod_corr_norm, "cyan"), 0.0, 1.0)

    dic_corrected = correct_dic_background(dic, DIC_BACKGROUND)
    labeled, n_components = segment_dic(dic_corrected)
    cells = list(accepted_cells(labeled, n_components, dic.shape))
    bod_smooth = ndi.gaussian_filter(bod, sigma=LIPID_SMOOTH_SIGMA)

    cell_rows = []
    for cell_id, (ys, xs, props) in enumerate(cells, start=1):
        mask = np.zeros(dic.shape, dtype=bool)
        mask[ys, xs] = True
        contours = find_contours(mask.astype(np.float64), 0.5)
        contour = max(contours, key=len)
        cell_rows.append(dict(
            cell_id=cell_id,
            xs=contour[:, 1].tolist(),
            ys=height_to_bokeh_y(contour[:, 0], height).tolist(),
            n_lipid_bodies=count_lipid_bodies(bod_smooth, ys, xs),
            **props,
        ))

    result = dict(
        prefix=prefix, fov_num=fov_num, height=height, width=width,
        dic_rgb=np.flipud(dic_rgb), chl_rgb=np.flipud(chl_rgb),
        bod_rgb=np.flipud(bod_rgb), overlay_rgb=np.flipud(overlay_rgb),
        reg_overlay_rgb=np.flipud(reg_overlay_rgb),
        cells=cell_rows,
    )
    _cache[idx] = result
    return result


def to_rgba_uint32(rgb):
    h, w, _ = rgb.shape
    img = np.empty((h, w), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((h, w, 4))
    view[..., 0] = np.clip(rgb[..., 0] * 255, 0, 255).astype(np.uint8)
    view[..., 1] = np.clip(rgb[..., 1] * 255, 0, 255).astype(np.uint8)
    view[..., 2] = np.clip(rgb[..., 2] * 255, 0, 255).astype(np.uint8)
    view[..., 3] = 255
    return img


# ---------------------------------------------------------------------------
# Bokeh figures
# ---------------------------------------------------------------------------

SAMPLE_HEIGHT, SAMPLE_WIDTH = tifffile.imread(fov_items[0][1]["DIC"]).shape
ASPECT = SAMPLE_HEIGHT / SAMPLE_WIDTH

LARGE_W = 600
LARGE_H = int(LARGE_W * ASPECT)
SMALL_W = LARGE_W // 4
SMALL_H = int(SMALL_W * ASPECT)


def add_scale_bar(fig, width, height, show_label=True):
    """White SCALE_BAR_PX = SCALE_BAR_UM um scale bar, bottom-right, matching composite_figure.py's.
    Label is horizontally centered over the bar (bokeh's own text_align, not the bar's right edge)."""
    margin = 0.03 * width
    bar_height = max(2, round(0.006 * height))
    x0 = width - margin - SCALE_BAR_PX
    y0 = margin
    fig.quad(
        left=[x0], right=[x0 + SCALE_BAR_PX], bottom=[y0], top=[y0 + bar_height],
        fill_color="white", line_color="white",
    )
    if show_label:
        fig.text(
            x=[x0 + SCALE_BAR_PX / 2], y=[y0 + bar_height * 2.5], text=[f"{SCALE_BAR_UM} µm"],
            text_color="white", text_align="center", text_baseline="bottom",
            text_font_size="10pt", text_font_style="bold",
        )


def make_image_figure(title, width, height, x_range, y_range):
    fig = figure(
        title=title, width=width, height=height,
        sizing_mode="scale_width",
        x_range=x_range, y_range=y_range,
        tools="pan,wheel_zoom,reset", match_aspect=True,
        background_fill_color="white", border_fill_color="white",
    )
    fig.axis.visible = False
    fig.grid.visible = False
    fig.toolbar.logo = None
    return fig


# shared Range1d instances link pan/zoom across a-d and, separately, across e-f
small_x_range = Range1d(0, SAMPLE_WIDTH)
small_y_range = Range1d(0, SAMPLE_HEIGHT)
large_x_range = Range1d(0, SAMPLE_WIDTH)
large_y_range = Range1d(0, SAMPLE_HEIGHT)

dic_fig = make_image_figure("a: DIC", SMALL_W, SMALL_H, small_x_range, small_y_range)
chl_fig = make_image_figure("b: Chlorophyll", SMALL_W, SMALL_H, small_x_range, small_y_range)
bod_fig = make_image_figure("c: BODIPY", SMALL_W, SMALL_H, small_x_range, small_y_range)
overlay_fig = make_image_figure("d: Chlorophyll + BODIPY", SMALL_W, SMALL_H, small_x_range, small_y_range)
seg_fig = make_image_figure(
    "e: DIC + segmentation (click=flag, drag=lasso missed-cell ROI)", LARGE_W, LARGE_H,
    large_x_range, large_y_range,
)
reg_fig = make_image_figure(
    "f: registration-corrected Chlorophyll+BODIPY + mask outline", LARGE_W, LARGE_H,
    large_x_range, large_y_range,
)

dic_src = ColumnDataSource(data=dict(image=[]))
chl_src = ColumnDataSource(data=dict(image=[]))
bod_src = ColumnDataSource(data=dict(image=[]))
overlay_src = ColumnDataSource(data=dict(image=[]))
seg_bg_src = ColumnDataSource(data=dict(image=[]))
reg_bg_src = ColumnDataSource(data=dict(image=[]))

for fig, src in [(dic_fig, dic_src), (chl_fig, chl_src), (bod_fig, bod_src),
                  (overlay_fig, overlay_src), (seg_fig, seg_bg_src), (reg_fig, reg_bg_src)]:
    fig.image_rgba(image="image", x=0, y=0, dw=SAMPLE_WIDTH, dh=SAMPLE_HEIGHT, source=src)

cells_src = ColumnDataSource(data=dict(
    xs=[], ys=[], cell_id=[], length_um=[], width_um=[], aspect_ratio=[],
    solidity=[], n_lipid_bodies=[], fill_color=[], line_color=[],
))
patches = seg_fig.patches(
    xs="xs", ys="ys", source=cells_src,
    fill_color="fill_color", fill_alpha=0.45,
    line_color="line_color", line_width=2,
    selection_fill_alpha=0.45, nonselection_fill_alpha=0.45,
)
seg_fig.add_tools(TapTool(renderers=[patches]))
seg_fig.add_tools(HoverTool(renderers=[patches], tooltips=[
    ("cell", "@cell_id"),
    ("length (um)", "@length_um{0.0}"),
    ("width (um)", "@width_um{0.0}"),
    ("aspect ratio", "@aspect_ratio{0.0}"),
    ("solidity", "@solidity{0.000}"),
    ("lipid bodies", "@n_lipid_bodies"),
]))

# manually-lassoed "missed good cell" ROIs, on the same panel
missed_src = ColumnDataSource(data=dict(xs=[], ys=[], note=[]))
missed_patches = seg_fig.patches(
    xs="xs", ys="ys", source=missed_src,
    fill_alpha=0.25, fill_color=MISSED_FILL, line_color=MISSED_LINE, line_width=2,
)
freehand_tool = FreehandDrawTool(renderers=[missed_patches], empty_value="")
seg_fig.add_tools(freehand_tool)

# panel f: same accepted-cell outlines, drawn as an unfilled overlay to check registration
reg_fig.patches(
    xs="xs", ys="ys", source=cells_src,
    fill_alpha=0, line_color=REG_OUTLINE, line_width=2,
)

# scale bars: segmentation panel, plus the two Chlorophyll+BODIPY overlays (raw and registration-corrected)
add_scale_bar(seg_fig, SAMPLE_WIDTH, SAMPLE_HEIGHT)
add_scale_bar(overlay_fig, SAMPLE_WIDTH, SAMPLE_HEIGHT, show_label=False)
add_scale_bar(reg_fig, SAMPLE_WIDTH, SAMPLE_HEIGHT)

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

fov_options = [
    (str(i), f"{prefix}FOV{fov_num} ({i + 1}/{len(fov_items)})")
    for i, ((prefix, fov_num), _paths) in enumerate(fov_items)
]
fov_select = Select(title="Field of view", options=fov_options, value="0", width=320)
prev_button = Button(label="< Previous", width=100)
next_button = Button(label="Next >", width=100)
status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)
flagged_div = Div(text="", width=900, styles=WHITE_STYLE)
export_button = Button(label="Export ROIs", button_type="primary", width=180)
export_status_div = Div(text="", width=700, styles=WHITE_STYLE)
notes_column = column(styles=WHITE_STYLE)

flagged_csv_input = TextInput(title="Flagged cells CSV filename", value=DEFAULT_FLAGGED_NAME, width=320)
missed_csv_input = TextInput(title="Missed-cell ROIs CSV filename", value=DEFAULT_MISSED_NAME, width=320)
load_status_div = Div(text="", sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

current_idx = [0]


def flagged_csv_path():
    return os.path.join(OUTPUT_DIR, flagged_csv_input.value.strip() or DEFAULT_FLAGGED_NAME)


def missed_csv_path():
    return os.path.join(OUTPUT_DIR, missed_csv_input.value.strip() or DEFAULT_MISSED_NAME)


def iter_missed_rois():
    for (prefix, fov_num), rois in missed_rois_registry.items():
        notes = rois.get("note", [])
        for i in range(len(rois.get("xs", []))):
            yield prefix, fov_num, i, (notes[i] if i < len(notes) else "")


def render_flagged_list():
    parts = []
    if flagged_registry:
        rows = "".join(
            f"<li>{p}FOV{f} cell {c} "
            f"(length={m['length_um']:.1f}um, width={m['width_um']:.1f}um, "
            f"aspect={m['aspect_ratio']:.1f}, solidity={m['solidity']:.3f}, "
            f"lipid_bodies={m['n_lipid_bodies']}"
            f"{', note: ' + m['note'] if m.get('note') else ''})</li>"
            for (p, f, c), m in sorted(flagged_registry.items())
        )
        parts.append(f"<b>Flagged cells ({len(flagged_registry)}):</b><ul>{rows}</ul>")
    else:
        parts.append("<b>Flagged cells:</b> none yet.")

    missed_list = sorted(iter_missed_rois())
    if missed_list:
        rows = "".join(
            f"<li>{p}FOV{f} ROI {i + 1}{', note: ' + note if note else ''}</li>"
            for p, f, i, note in missed_list
        )
        parts.append(f"<b>Missed-cell ROIs ({len(missed_list)}):</b><ul>{rows}</ul>")
    else:
        parts.append("<b>Missed-cell ROIs:</b> none yet.")

    flagged_div.text = "".join(parts)


def load_existing_exports():
    """Check flagged/missed CSVs at the current filenames; if present, merge their contents
    into the in-memory registries (existing in-session entries are not cleared)."""
    messages = []

    fpath = flagged_csv_path()
    if os.path.exists(fpath):
        df = pd.read_csv(fpath, keep_default_na=False)
        for _, csv_row in df.iterrows():
            key = (str(csv_row["sample"]), int(csv_row["fov"]), int(csv_row["cell_id"]))
            measurements = csv_row.drop(labels=["sample", "fov", "cell_id"]).to_dict()
            measurements["n_lipid_bodies"] = int(measurements["n_lipid_bodies"])
            flagged_registry[key] = measurements
        messages.append(f"Loaded {len(df)} flagged cell(s) from {fpath}.")
    else:
        messages.append(f"No existing file at {fpath}.")

    mpath = missed_csv_path()
    if os.path.exists(mpath):
        df = pd.read_csv(mpath, keep_default_na=False)
        n_loaded = 0
        for (sample, fov), group in df.groupby(["sample", "fov"]):
            group = group.sort_values("roi_index")
            xs_list, ys_list, notes_list = [], [], []
            for _, csv_row in group.iterrows():
                rc_pairs = json.loads(csv_row["polygon_row_col"])
                xs_list.append([c for r, c in rc_pairs])
                ys_list.append([SAMPLE_HEIGHT - r for r, c in rc_pairs])
                notes_list.append(csv_row.get("note", ""))
            missed_rois_registry[(str(sample), int(fov))] = dict(xs=xs_list, ys=ys_list, note=notes_list)
            n_loaded += len(group)
        messages.append(f"Loaded {n_loaded} missed-cell ROI(s) from {mpath}.")
    else:
        messages.append(f"No existing file at {mpath}.")

    load_status_div.text = "<br>".join(messages)
    show_fov(current_idx[0])
    render_flagged_list()


def _cell_note_callback(key):
    def cb(attr, old, new):
        if key in flagged_registry:
            flagged_registry[key]["note"] = new
    return cb


def _roi_note_callback(i):
    def cb(attr, old, new):
        notes = list(missed_src.data.get("note", []))
        if i < len(notes):
            notes[i] = new
            new_data = dict(missed_src.data)
            new_data["note"] = notes
            missed_src.data = new_data
    return cb


def _remove_roi_callback(i):
    def cb():
        xs_list = list(missed_src.data.get("xs", []))
        ys_list = list(missed_src.data.get("ys", []))
        notes_list = list(missed_src.data.get("note", []))
        if i < len(xs_list):
            del xs_list[i]
            del ys_list[i]
            del notes_list[i]
            missed_src.data = dict(xs=xs_list, ys=ys_list, note=notes_list)
    return cb


def rebuild_notes_panel():
    idx = current_idx[0]
    data = load_fov(idx)
    prefix, fov_num = data["prefix"], data["fov_num"]
    widgets = []
    for cell in data["cells"]:
        key = (prefix, fov_num, cell["cell_id"])
        if key in flagged_registry:
            note_val = flagged_registry[key].get("note", "")
            ti = TextAreaInput(
                value=note_val, rows=2, width=560,
                title=f"Note -- flagged cell {cell['cell_id']} (why is this segmentation wrong?)",
                styles=WHITE_STYLE,
            )
            ti.on_change("value", _cell_note_callback(key))
            widgets.append(ti)

    notes_list = missed_src.data.get("note", [])
    for i in range(len(missed_src.data.get("xs", []))):
        ti = TextAreaInput(
            value=notes_list[i] if i < len(notes_list) else "", rows=2, width=460,
            title=f"Note -- missed-cell ROI {i + 1} (why should this be a cell?)",
            styles=WHITE_STYLE,
        )
        ti.on_change("value", _roi_note_callback(i))
        remove_button = Button(label="Remove ROI", button_type="danger", width=100, margin=(24, 0, 0, 10))
        remove_button.on_click(_remove_roi_callback(i))
        widgets.append(row(ti, remove_button, styles=WHITE_STYLE))

    if not widgets:
        widgets = [Div(text="<i>No flagged cells or missed-cell ROIs on this FOV yet.</i>", styles=WHITE_STYLE)]
    notes_column.children = widgets


def save_missed_rois_for_current_fov():
    idx = current_idx[0]
    data = load_fov(idx)
    key = (data["prefix"], data["fov_num"])
    missed_rois_registry[key] = {k: list(v) for k, v in missed_src.data.items()}


def _structural_change(old_data, new_data):
    for key in ("xs", "ys"):
        if list(old_data.get(key, [])) != list(new_data.get(key, [])):
            return True
    return False


def show_fov(idx):
    idx = max(0, min(len(fov_items) - 1, idx))
    current_idx[0] = idx
    fov_select.value = str(idx)

    data = load_fov(idx)
    dic_src.data = dict(image=[to_rgba_uint32(data["dic_rgb"])])
    chl_src.data = dict(image=[to_rgba_uint32(data["chl_rgb"])])
    bod_src.data = dict(image=[to_rgba_uint32(data["bod_rgb"])])
    overlay_src.data = dict(image=[to_rgba_uint32(data["overlay_rgb"])])
    seg_bg_src.data = dict(image=[to_rgba_uint32(data["dic_rgb"])])
    reg_bg_src.data = dict(image=[to_rgba_uint32(data["reg_overlay_rgb"])])

    prefix, fov_num = data["prefix"], data["fov_num"]
    xs, ys, cell_id, length_um, width_um, aspect_ratio, solidity, n_lipid, fill_color, line_color = (
        [], [], [], [], [], [], [], [], [], [],
    )
    for cell in data["cells"]:
        key = (prefix, fov_num, cell["cell_id"])
        is_flagged = key in flagged_registry
        xs.append(cell["xs"]); ys.append(cell["ys"]); cell_id.append(cell["cell_id"])
        length_um.append(cell["length_um"]); width_um.append(cell["width_um"])
        aspect_ratio.append(cell["aspect_ratio"]); solidity.append(cell["solidity"])
        n_lipid.append(cell["n_lipid_bodies"])
        fill_color.append(FLAGGED_FILL if is_flagged else OK_FILL)
        line_color.append(FLAGGED_LINE if is_flagged else OK_LINE)

    cells_src.data = dict(
        xs=xs, ys=ys, cell_id=cell_id, length_um=length_um, width_um=width_um,
        aspect_ratio=aspect_ratio, solidity=solidity, n_lipid_bodies=n_lipid,
        fill_color=fill_color, line_color=line_color,
    )
    cells_src.selected.indices = []

    saved_rois = missed_rois_registry.get((prefix, fov_num), dict(xs=[], ys=[], note=[]))
    missed_src.data = {k: list(v) for k, v in saved_rois.items()}

    status_div.text = (
        f"<b>{prefix}FOV{fov_num}</b> &mdash; FOV {idx + 1} of {len(fov_items)} "
        f"&mdash; {len(data['cells'])} accepted cell(s)"
    )
    rebuild_notes_panel()


def on_selected_change(attr, old, new):
    if not new:
        return
    data = load_fov(current_idx[0])
    prefix, fov_num = data["prefix"], data["fov_num"]
    fill_colors = list(cells_src.data["fill_color"])
    line_colors = list(cells_src.data["line_color"])
    for i in new:
        cell = data["cells"][i]
        key = (prefix, fov_num, cell["cell_id"])
        if key in flagged_registry:
            del flagged_registry[key]
            fill_colors[i], line_colors[i] = OK_FILL, OK_LINE
        else:
            note_free_cell = {k: v for k, v in cell.items() if k not in ("xs", "ys", "cell_id")}
            flagged_registry[key] = dict(note_free_cell, note="")
            fill_colors[i], line_colors[i] = FLAGGED_FILL, FLAGGED_LINE
    cells_src.data["fill_color"] = fill_colors
    cells_src.data["line_color"] = line_colors
    cells_src.selected.indices = []
    render_flagged_list()
    rebuild_notes_panel()


def on_missed_data_change(attr, old, new):
    save_missed_rois_for_current_fov()
    if _structural_change(old, new):
        rebuild_notes_panel()
        render_flagged_list()


def on_prev():
    show_fov(current_idx[0] - 1)


def on_next():
    show_fov(current_idx[0] + 1)


def on_select(attr, old, new):
    show_fov(int(new))


def on_export():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_cells = n_rois = 0
    fpath, mpath = flagged_csv_path(), missed_csv_path()

    if flagged_registry:
        rows = [
            dict(sample=prefix, fov=fov_num, cell_id=cell_id, **measurements)
            for (prefix, fov_num, cell_id), measurements in sorted(flagged_registry.items())
        ]
        pd.DataFrame(rows).to_csv(fpath, index=False)
        n_cells = len(rows)

    missed_list = list(iter_missed_rois())
    if missed_list:
        rows = []
        for prefix, fov_num, i, note in missed_list:
            rois = missed_rois_registry[(prefix, fov_num)]
            xs_i, ys_i = rois["xs"][i], rois["ys"][i]
            rows_rc = [[round(SAMPLE_HEIGHT - y, 2), round(x, 2)] for x, y in zip(xs_i, ys_i)]
            rows.append(dict(
                sample=prefix, fov=fov_num, roi_index=i + 1,
                row_min=min(r for r, c in rows_rc), row_max=max(r for r, c in rows_rc),
                col_min=min(c for r, c in rows_rc), col_max=max(c for r, c in rows_rc),
                polygon_row_col=json.dumps(rows_rc), note=note,
            ))
        pd.DataFrame(rows).to_csv(mpath, index=False)
        n_rois = len(rows)

    if n_cells == 0 and n_rois == 0:
        export_status_div.text = "Nothing flagged or marked yet -- nothing to export."
        return
    export_status_div.text = (
        f"Exported {n_cells} flagged cell(s) to {fpath} and "
        f"{n_rois} missed-cell ROI(s) to {mpath}"
    )


def on_output_filename_change(attr, old, new):
    load_existing_exports()


prev_button.on_click(on_prev)
next_button.on_click(on_next)
fov_select.on_change("value", on_select)
cells_src.selected.on_change("indices", on_selected_change)
missed_src.on_change("data", on_missed_data_change)
export_button.on_click(on_export)
flagged_csv_input.on_change("value", on_output_filename_change)
missed_csv_input.on_change("value", on_output_filename_change)

instructions = Div(text=(
    "<p>Panel <b>e</b>: click a cell outline to flag it as poorly segmented "
    "(red = flagged; click again to unflag). Select the <b>Freehand Draw</b> "
    "tool (toolbar icon on panel e) to lasso a good cell the pipeline "
    "missed; drew one poorly? Use the <b>Remove ROI</b> button next to its "
    "note field below the panels, rather than the tool's own tap-to-select. "
    "Panel <b>f</b> shows the registration-corrected fluorescence with the "
    "same DIC mask outlines, to check the correction is centering signal "
    "inside each cell rather than clipping an edge. Add a note to any "
    "flagged cell or missed-cell ROI below the panels, then use "
    "<b>Export ROIs</b> to write everything to the two output filenames set "
    "below -- if either already exists (e.g. from a previous session), its "
    "contents are loaded back into this dashboard automatically.</p>"
), sizing_mode="stretch_width", styles=FULL_WIDTH_TEXT_STYLE)

layout = column(
    instructions,
    row(flagged_csv_input, missed_csv_input, sizing_mode="stretch_width"),
    load_status_div,
    row(prev_button, next_button, fov_select),
    status_div,
    row(dic_fig, chl_fig, bod_fig, overlay_fig, sizing_mode="stretch_width"),
    row(seg_fig, sizing_mode="stretch_width"),
    row(reg_fig, sizing_mode="stretch_width"),
    row(export_button, export_status_div),
    flagged_div,
    Div(text="<b>Notes on the current FOV's ROIs</b>", styles=WHITE_STYLE),
    notes_column,
    sizing_mode="stretch_width",
    styles=dict(WHITE_STYLE, padding="12px"),
)

show_fov(0)
load_existing_exports()

curdoc().add_root(layout)
curdoc().title = "Cell segmentation QC review"
