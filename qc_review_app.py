#!/usr/bin/env python3
"""
Interactive Bokeh server app for visually reviewing quantify_cells.py's
segmentation, FOV by FOV, and flagging cells whose segmentation looks wrong.

Panels: a=DIC, b=Chlorophyll, c=BODIPY, d=Chlorophyll+BODIPY overlay (same
global per-channel normalization as composite_figure.py, computed once over
the whole input directory), e=DIC with each accepted cell's mask boundary
drawn as a clickable region. Click a cell's outline in panel e to flag it as
poorly segmented (click again to unflag); flagged cells turn red. "Export
flagged ROIs" writes every currently-flagged cell (sample/fov/cell_id plus
its quantify_cells.py measurements) to <output_dir>/flagged_rois.csv.

Segmentation reuses quantify_cells.segment_dic/accepted_cells/
count_lipid_bodies verbatim -- this app never re-implements or approximates
the pipeline, so what you review here is exactly what the CSVs contain.

Run:
    bokeh serve --show qc_review_app.py --args <input_dir> [<output_dir>]

Defaults to ./renamed_composites and ./quantification if omitted.
"""

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
)
from composite_figure import find_channel_files, compute_global_ranges, normalize, to_rgb, group_fovs

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Button, Select, Div, HoverTool, TapTool
from bokeh.plotting import figure

INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "renamed_composites"
OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "quantification"
FLAGGED_CSV = os.path.join(OUTPUT_DIR, "flagged_rois.csv")

FLAGGED_FILL, FLAGGED_LINE = "#E24B4A", "#A32D2D"
OK_FILL, OK_LINE = "#5DCAA5", "#0F6E56"

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

fov_items = sorted(group_fovs(INPUT_DIR).items())
fov_items = [(key, paths) for key, paths in fov_items if all(c in paths for c in CHANNELS)]
if not fov_items:
    raise SystemExit(f"No complete FOVs (all 3 channels) found in {INPUT_DIR}")
print(f"Found {len(fov_items)} complete FOVs.")

_cache = {}
flagged_registry = {}  # (prefix, fov_num, cell_id) -> measurement dict


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

    labeled, _n_components = segment_dic(dic)
    cells = list(accepted_cells(labeled, dic.shape))
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

SMALL_W = 300
SMALL_H = int(SMALL_W * ASPECT)
LARGE_W = 520
LARGE_H = int(LARGE_W * ASPECT)


def make_image_figure(title, width, height):
    fig = figure(
        title=title, width=width, height=height,
        x_range=(0, SAMPLE_WIDTH), y_range=(0, SAMPLE_HEIGHT),
        tools="pan,wheel_zoom,reset", match_aspect=True,
    )
    fig.axis.visible = False
    fig.grid.visible = False
    fig.toolbar.logo = None
    return fig


dic_fig = make_image_figure("a: DIC", SMALL_W, SMALL_H)
chl_fig = make_image_figure("b: Chlorophyll", SMALL_W, SMALL_H)
bod_fig = make_image_figure("c: BODIPY", SMALL_W, SMALL_H)
overlay_fig = make_image_figure("d: Chlorophyll + BODIPY", SMALL_W, SMALL_H)
seg_fig = make_image_figure("e: DIC + segmentation (click a cell to flag)", LARGE_W, LARGE_H)

dic_src = ColumnDataSource(data=dict(image=[]))
chl_src = ColumnDataSource(data=dict(image=[]))
bod_src = ColumnDataSource(data=dict(image=[]))
overlay_src = ColumnDataSource(data=dict(image=[]))
seg_bg_src = ColumnDataSource(data=dict(image=[]))

for fig, src in [(dic_fig, dic_src), (chl_fig, chl_src), (bod_fig, bod_src),
                  (overlay_fig, overlay_src), (seg_fig, seg_bg_src)]:
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
status_div = Div(text="", width=700)
flagged_div = Div(text="", width=900)
export_button = Button(label="Export flagged ROIs", button_type="primary", width=180)
export_status_div = Div(text="", width=700)

current_idx = [0]


def render_flagged_list():
    if not flagged_registry:
        flagged_div.text = "<b>Flagged for review:</b> none yet."
        return
    rows = "".join(
        f"<li>{p}FOV{f} cell {c} "
        f"(length={m['length_um']:.1f}um, width={m['width_um']:.1f}um, "
        f"aspect={m['aspect_ratio']:.1f}, solidity={m['solidity']:.3f}, "
        f"lipid_bodies={m['n_lipid_bodies']})</li>"
        for (p, f, c), m in sorted(flagged_registry.items())
    )
    flagged_div.text = f"<b>Flagged for review ({len(flagged_registry)}):</b><ul>{rows}</ul>"


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

    status_div.text = (
        f"<b>{prefix}FOV{fov_num}</b> &mdash; FOV {idx + 1} of {len(fov_items)} "
        f"&mdash; {len(data['cells'])} accepted cell(s)"
    )


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
            flagged_registry[key] = {k: v for k, v in cell.items() if k not in ("xs", "ys", "cell_id")}
            fill_colors[i], line_colors[i] = FLAGGED_FILL, FLAGGED_LINE
    cells_src.data["fill_color"] = fill_colors
    cells_src.data["line_color"] = line_colors
    cells_src.selected.indices = []
    render_flagged_list()


def on_prev():
    show_fov(current_idx[0] - 1)


def on_next():
    show_fov(current_idx[0] + 1)


def on_select(attr, old, new):
    show_fov(int(new))


def on_export():
    if not flagged_registry:
        export_status_div.text = "Nothing flagged yet -- nothing to export."
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for (prefix, fov_num, cell_id), measurements in sorted(flagged_registry.items()):
        rows.append(dict(sample=prefix, fov=fov_num, cell_id=cell_id, **measurements))
    pd.DataFrame(rows).to_csv(FLAGGED_CSV, index=False)
    export_status_div.text = f"Exported {len(rows)} flagged ROI(s) to {FLAGGED_CSV}"


prev_button.on_click(on_prev)
next_button.on_click(on_next)
fov_select.on_change("value", on_select)
cells_src.selected.on_change("indices", on_selected_change)
export_button.on_click(on_export)

instructions = Div(text=(
    "<p>Click a cell outline in panel <b>e</b> to flag it as poorly segmented "
    "(red = flagged); click again to unflag. Flags persist as you navigate "
    "between FOVs. Use <b>Export flagged ROIs</b> to write them all to "
    f"<code>{FLAGGED_CSV}</code> for offline review.</p>"
))

layout = column(
    instructions,
    row(prev_button, next_button, fov_select),
    status_div,
    row(dic_fig, chl_fig, bod_fig),
    row(overlay_fig, seg_fig),
    row(export_button, export_status_div),
    flagged_div,
)

show_fov(0)

curdoc().add_root(layout)
curdoc().title = "Cell segmentation QC review"
