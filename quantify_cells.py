#!/usr/bin/env python3
"""
Segment Phaeodactylum tricornutum cells from the DIC channel and measure
per-cell Chlorophyll and BODIPY fluorescence, pooled across all replicates
and fields of view for each condition.

Segmentation: Sobel gradient magnitude of the DIC image -> Otsu threshold
-> morphological closing (bridges the two faint edge lines of a thin cell
into a filled body) -> hole filling -> opening (denoise) -> connected
components. Components are kept only if their fitted ellipse dimensions and
solidity fall within the expected P. tricornutum fusiform morphology; the
size/aspect-ratio bounds below were calibrated against this dataset (see
project conversation) rather than taken as fixed biological constants.

For each accepted cell, "total" fluorescence is the sum of raw pixel
intensity within the cell mask; "average" is that total divided by the
cell's pixel area (i.e. mean intensity per pixel in the cell).

Lipid bodies are counted by smoothing the BODIPY channel, Otsu-thresholding
it within each cell mask, and counting connected bright components above a
minimum size. Droplets that touch or overlap enough to merge into one
connected component are counted as a single lipid body.

Usage:
    python quantify_cells.py <input_dir> <output_dir>
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import tifffile
import scipy.ndimage as ndi
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANNELS = ["DIC", "Chlorophyll", "BODIPY"]
FOV_FILE_RE = re.compile(r"^(?P<prefix>.+?)FOV(?P<fov>\d+)_(?P<channel>DIC|Chlorophyll|BODIPY)\.tiff$")

UM_PER_PX = 10.0 / 65.0  # matches the 65 px = 10 um scale bar used in composite_figure.py

GAUSSIAN_SIGMA = 1.0
THRESHOLD_MULT = 0.4
CLOSE_RADIUS_PX = 10
OPEN_RADIUS_PX = 2
MIN_COMPONENT_AREA_PX = 300

LENGTH_UM_RANGE = (15.0, 40.0)
WIDTH_UM_RANGE = (2.0, 7.0)
ASPECT_RATIO_RANGE = (5.0, 16.0)
MIN_SOLIDITY = 0.75

LIPID_SMOOTH_SIGMA = 1.0
LIPID_MIN_SIZE_PX = 3

CONDITION_ORDER = ["Nitrate", "Arginine", "Urea"]


def disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2) <= radius ** 2


def otsu_threshold(values, nbins=256):
    hist, edges = np.histogram(values, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    hist = hist.astype(np.float64)
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * centers) / np.clip(w1, 1, None)
    m2 = (np.cumsum((hist * centers)[::-1])[::-1]) / np.clip(w2, 1, None)
    var_between = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return centers[np.argmax(var_between)]


def segment_dic(dic_img):
    smooth = ndi.gaussian_filter(dic_img.astype(np.float64), sigma=GAUSSIAN_SIGMA)
    grad_mag = np.hypot(ndi.sobel(smooth, axis=1), ndi.sobel(smooth, axis=0))
    thresh = otsu_threshold(grad_mag.ravel())
    mask = grad_mag > thresh * THRESHOLD_MULT
    mask = ndi.binary_closing(mask, structure=disk(CLOSE_RADIUS_PX))
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, structure=disk(OPEN_RADIUS_PX))
    return ndi.label(mask)


def fit_ellipse(ys, xs):
    """Major/minor axis lengths (px) of the ellipse with the same second moments as the region."""
    dy, dx = ys - ys.mean(), xs - xs.mean()
    n = len(ys)
    muyy, muxx, muxy = np.sum(dy * dy) / n, np.sum(dx * dx) / n, np.sum(dy * dx) / n
    common = np.sqrt((muxx - muyy) ** 2 + 4 * muxy ** 2)
    l1 = (muxx + muyy + common) / 2
    l2 = max((muxx + muyy - common) / 2, 0)
    return 4 * np.sqrt(l1), 4 * np.sqrt(max(l2, 1e-6))


def accepted_cells(labeled, n_components, image_shape):
    """Yield (ys, xs) pixel coordinates for each component that passes the morphology filters."""
    height, width = image_shape
    for label_id in range(1, n_components + 1):
        ys, xs = np.where(labeled == label_id)
        area = len(ys)
        if area < MIN_COMPONENT_AREA_PX:
            continue
        if ys.min() == 0 or xs.min() == 0 or ys.max() == height - 1 or xs.max() == width - 1:
            continue

        major_px, minor_px = fit_ellipse(ys, xs)
        if minor_px <= 0:
            continue
        length_um, width_um = major_px * UM_PER_PX, minor_px * UM_PER_PX
        aspect_ratio = major_px / minor_px
        if not (LENGTH_UM_RANGE[0] <= length_um <= LENGTH_UM_RANGE[1]):
            continue
        if not (WIDTH_UM_RANGE[0] <= width_um <= WIDTH_UM_RANGE[1]):
            continue
        if not (ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1]):
            continue

        hull_area = ConvexHull(np.column_stack([xs, ys])).volume
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < MIN_SOLIDITY:
            continue

        yield ys, xs, dict(
            area_px=area, length_um=length_um, width_um=width_um,
            aspect_ratio=aspect_ratio, solidity=solidity,
        )


def count_lipid_bodies(bodipy_smooth, ys, xs):
    """Number of distinct BODIPY-bright connected components within a cell mask."""
    cell_mask = np.zeros(bodipy_smooth.shape, dtype=bool)
    cell_mask[ys, xs] = True
    thresh = otsu_threshold(bodipy_smooth[ys, xs])
    bright_mask = cell_mask & (bodipy_smooth > thresh)
    labeled, n = ndi.label(bright_mask, structure=np.ones((3, 3)))
    if n == 0:
        return 0
    sizes = ndi.sum(bright_mask, labeled, index=np.arange(1, n + 1))
    return int(np.sum(sizes >= LIPID_MIN_SIZE_PX))


def group_fovs(directory):
    fovs = defaultdict(dict)
    for path in glob.glob(os.path.join(directory, "*.tiff")):
        match = FOV_FILE_RE.match(os.path.basename(path))
        if match:
            key = (match.group("prefix"), int(match.group("fov")))
            fovs[key][match.group("channel")] = path
    return fovs


def save_qc_overlay(dic_img, cells, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(dic_img, cmap="gray")
    overlay = np.zeros(dic_img.shape, dtype=np.float64)
    for i, (ys, xs, _props) in enumerate(cells, start=1):
        overlay[ys, xs] = i
    ax.imshow(np.ma.masked_where(overlay == 0, overlay), cmap="tab20", alpha=0.5)
    ax.set_title(f"{len(cells)} accepted cell(s)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_categorical_plot(df, value_col, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    rng = np.random.default_rng(0)
    for i, condition in enumerate(CONDITION_ORDER):
        values = df.loc[df["condition"] == condition, value_col].values
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, alpha=0.6, s=18, edgecolor="none")
        if len(values):
            ax.hlines(values.mean(), i - 0.22, i + 0.22, color="black", linewidth=2)
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Directory of renamed <prefix>FOV<n>_<Channel>.tiff files")
    parser.add_argument("output_dir")
    parser.add_argument("--qc-overlays", action="store_true", help="Save a segmentation overlay PNG per FOV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    qc_dir = os.path.join(args.output_dir, "qc_overlays")
    if args.qc_overlays:
        os.makedirs(qc_dir, exist_ok=True)

    rows = []
    fovs = group_fovs(args.input_dir)
    for (prefix, fov_num), channel_paths in sorted(fovs.items()):
        missing = [c for c in CHANNELS if c not in channel_paths]
        if missing:
            print(f"SKIPPING {prefix} FOV{fov_num}: missing {missing}")
            continue

        condition = prefix.split("_Day")[0]
        dic = tifffile.imread(channel_paths["DIC"])
        chl = tifffile.imread(channel_paths["Chlorophyll"]).astype(np.float64)
        bod = tifffile.imread(channel_paths["BODIPY"]).astype(np.float64)

        labeled, n_components = segment_dic(dic)
        cells = list(accepted_cells(labeled, n_components, dic.shape))
        bod_smooth = ndi.gaussian_filter(bod, sigma=LIPID_SMOOTH_SIGMA)

        for cell_id, (ys, xs, props) in enumerate(cells, start=1):
            total_chl, total_bod = chl[ys, xs].sum(), bod[ys, xs].sum()
            rows.append(dict(
                condition=condition, sample=prefix, fov=fov_num, cell_id=cell_id,
                **props,
                total_chlorophyll=total_chl, total_bodipy=total_bod,
                avg_chlorophyll=total_chl / props["area_px"],
                avg_bodipy=total_bod / props["area_px"],
                n_lipid_bodies=count_lipid_bodies(bod_smooth, ys, xs),
            ))

        print(f"{prefix} FOV{fov_num}: {len(cells)} cell(s)")
        if args.qc_overlays:
            save_qc_overlay(dic, cells, os.path.join(qc_dir, f"{prefix}_FOV{fov_num}_qc.png"))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "cell_measurements.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} cell measurements to {csv_path}")
    print(df.groupby("condition").size().reindex(CONDITION_ORDER))

    make_categorical_plot(
        df, "n_lipid_bodies", "Number of lipid bodies per cell",
        os.path.join(args.output_dir, "lipid_bodies_per_cell.png"),
    )
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
