#!/usr/bin/env python3
"""
Build a 2x2 quantitative composite figure (DIC / Chlorophyll / BODIPY / overlay)
for every field of view found in a directory of renamed microscopy images
(see rename_microscopy_images.py for the naming scheme:
"<prefix>FOV<n>_<Channel>.tiff", Channel in {DIC, Chlorophyll, BODIPY}).

For each channel, pixel intensities are normalized using the minimum and
maximum observed across every image of that channel in the input directory,
so brightness is directly comparable between samples/replicates/FOVs (the
exposure for a given channel was held constant across the whole dataset).

Panel layout (2x2):
    a: DIC (grayscale)          b: Chlorophyll (magenta)
    c: BODIPY (cyan)            d: Chlorophyll + BODIPY overlay

Panels are spaced exactly 0.5 mm apart. A white 65 px (10 um) scale bar with
a "10 um" label is drawn in panel a.

Use --days/--reps to restrict to specific Day/replicate numbers, parsed from
the "<Condition>_Day<N>_rep<M>" filename prefix (e.g. --days 3 --reps 1,2).
This restricts BOTH which composite figures get rendered AND which images
feed the global per-channel min/max used for normalization -- i.e. a
filtered run is a fully self-contained analysis of that subset, with its own
normalization baseline, not a partial regeneration against a full-dataset
baseline computed elsewhere.

Usage:
    python composite_figure.py <input_dir> <output_dir> [--days 3] [--reps 1,2]
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

CHANNELS = ["DIC", "Chlorophyll", "BODIPY"]
FOV_FILE_RE = re.compile(r"^(?P<prefix>.+?)FOV(?P<fov>\d+)_(?P<channel>DIC|Chlorophyll|BODIPY)\.tiff$")
PREFIX_DAY_REP_RE = re.compile(r"_Day(?P<day>\d+)_rep(?P<rep>\d+)$")

MM_PER_INCH = 25.4
PANEL_GAP_MM = 0.5
SCALE_BAR_PX = 65
SCALE_BAR_UM = 10


def parse_day_rep(prefix):
    """Extract (day, rep) ints from a '<Condition>_Day<N>_rep<M>' prefix; (None, None) if it doesn't match."""
    match = PREFIX_DAY_REP_RE.search(prefix)
    if not match:
        return None, None
    return int(match.group("day")), int(match.group("rep"))


def parse_int_list(value):
    """Parse a CLI value like '1,3' into [1, 3]; None stays None (meaning "no filter")."""
    if value is None:
        return None
    return [int(v) for v in value.split(",") if v.strip()]


def matches_day_rep(prefix, days, reps):
    day, rep = parse_day_rep(prefix)
    if days is not None and day not in days:
        return False
    if reps is not None and rep not in reps:
        return False
    return True


def find_channel_files(directory, days=None, reps=None):
    """Map channel name -> list of file paths for every image of that channel."""
    files_by_channel = defaultdict(list)
    for path in glob.glob(os.path.join(directory, "*.tiff")):
        match = FOV_FILE_RE.match(os.path.basename(path))
        if match and matches_day_rep(match.group("prefix"), days, reps):
            files_by_channel[match.group("channel")].append(path)
    return files_by_channel


def compute_global_ranges(files_by_channel):
    """For each channel, find the min/max pixel value across all of its images."""
    ranges = {}
    for channel, paths in files_by_channel.items():
        vmin, vmax = np.inf, -np.inf
        for path in paths:
            img = tifffile.imread(path)
            vmin = min(vmin, float(img.min()))
            vmax = max(vmax, float(img.max()))
        ranges[channel] = (vmin, vmax)
    return ranges


def group_fovs(directory, days=None, reps=None):
    """Map (prefix, fov_number) -> {channel: path}."""
    fovs = defaultdict(dict)
    for path in glob.glob(os.path.join(directory, "*.tiff")):
        match = FOV_FILE_RE.match(os.path.basename(path))
        if match and matches_day_rep(match.group("prefix"), days, reps):
            key = (match.group("prefix"), int(match.group("fov")))
            fovs[key][match.group("channel")] = path
    return fovs


def normalize(img, vmin, vmax):
    return np.clip((img.astype(np.float64) - vmin) / (vmax - vmin), 0.0, 1.0)


def to_rgb(norm_img, color):
    rgb = np.zeros((*norm_img.shape, 3), dtype=np.float64)
    if color == "gray":
        rgb[..., 0] = rgb[..., 1] = rgb[..., 2] = norm_img
    elif color == "magenta":
        rgb[..., 0] = norm_img
        rgb[..., 2] = norm_img
    elif color == "cyan":
        rgb[..., 1] = norm_img
        rgb[..., 2] = norm_img
    else:
        raise ValueError(f"Unknown color: {color}")
    return rgb


def add_scale_bar(ax, image_shape):
    height, width = image_shape
    margin = 0.03 * width
    bar_height = max(2, round(0.006 * height))
    x0 = width - margin - SCALE_BAR_PX
    y0 = height - margin - bar_height
    ax.add_patch(Rectangle((x0, y0), SCALE_BAR_PX, bar_height, color="white", linewidth=0))
    ax.text(
        x0 + SCALE_BAR_PX, y0 - bar_height * 1.5, f"{SCALE_BAR_UM} µm",
        color="white", ha="right", va="bottom", fontsize=8, fontweight="bold",
    )


def build_figure(images, panel_size_in, dpi):
    """images: dict with keys 'DIC', 'Chlorophyll', 'BODIPY' -> normalized [0, 1] arrays."""
    height, width = images["DIC"].shape
    aspect = height / width

    panel_w_in = panel_size_in
    panel_h_in = panel_size_in * aspect
    gap_in = PANEL_GAP_MM / MM_PER_INCH
    margin_in = 0.25  # outer margin for panel labels, not part of the inter-panel gap

    fig_w_in = 2 * panel_w_in + gap_in + 2 * margin_in
    fig_h_in = 2 * panel_h_in + gap_in + 2 * margin_in

    wspace = gap_in / panel_w_in
    hspace = gap_in / panel_h_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    gs = GridSpec(
        2, 2, figure=fig,
        left=margin_in / fig_w_in, right=1 - margin_in / fig_w_in,
        top=1 - margin_in / fig_h_in, bottom=margin_in / fig_h_in,
        wspace=wspace, hspace=hspace,
    )

    dic_rgb = to_rgb(images["DIC"], "gray")
    chl_rgb = to_rgb(images["Chlorophyll"], "magenta")
    bod_rgb = to_rgb(images["BODIPY"], "cyan")
    overlay_rgb = np.clip(chl_rgb + bod_rgb, 0.0, 1.0)

    panels = [("a", dic_rgb, True), ("b", chl_rgb, False), ("c", bod_rgb, False), ("d", overlay_rgb, False)]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (label, rgb, draw_scale_bar), (row, col) in zip(panels, positions):
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(
            0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold", color="white",
            path_effects=[patheffects.withStroke(linewidth=2, foreground="black")],
        )
        if draw_scale_bar:
            add_scale_bar(ax, rgb.shape[:2])

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Directory of renamed <prefix>FOV<n>_<Channel>.tiff files")
    parser.add_argument("output_dir", help="Directory to write *_quantitative_composite.png files")
    parser.add_argument("--panel-size-in", type=float, default=3.0, help="Width of each panel, in inches")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--days", type=str, default=None,
                         help="Comma-separated Day numbers to include, e.g. '3' or '1,3' (default: all days present)")
    parser.add_argument("--reps", type=str, default=None,
                         help="Comma-separated replicate numbers to include, e.g. '1,2' (default: all reps present)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    days, reps = parse_int_list(args.days), parse_int_list(args.reps)

    files_by_channel = find_channel_files(args.input_dir, days=days, reps=reps)
    missing = [c for c in CHANNELS if c not in files_by_channel]
    if missing:
        raise SystemExit(f"No files found for channel(s): {missing} (--days={args.days} --reps={args.reps})")

    print("Computing global intensity ranges per channel...")
    ranges = compute_global_ranges(files_by_channel)
    for channel, (vmin, vmax) in ranges.items():
        print(f"  {channel}: min={vmin}, max={vmax}")

    fovs = group_fovs(args.input_dir, days=days, reps=reps)
    for (prefix, fov_num), channel_paths in sorted(fovs.items()):
        missing_channels = [c for c in CHANNELS if c not in channel_paths]
        if missing_channels:
            print(f"SKIPPING {prefix} FOV{fov_num}: missing {missing_channels}")
            continue

        images = {}
        for channel in CHANNELS:
            img = tifffile.imread(channel_paths[channel])
            vmin, vmax = ranges[channel]
            images[channel] = normalize(img, vmin, vmax)

        fig = build_figure(images, panel_size_in=args.panel_size_in, dpi=args.dpi)
        out_name = f"{prefix}_FOV{fov_num}_quantitative_composite.png"
        out_path = os.path.join(args.output_dir, out_name)
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
