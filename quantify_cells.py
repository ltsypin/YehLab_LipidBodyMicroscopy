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

Before smoothing, each DIC image has a fixed-pattern background subtracted --
a per-pixel median across every DIC image in the input directory, which
isolates stationary optical artifacts (e.g. an out-of-focus dust speck that
shows up at the same pixel location in every FOV) since real cells move from
FOV to FOV and get suppressed by the median. Left uncorrected, such an
artifact's edge ring can get pulled into a real cell's mask during
morphological closing if the cell happens to sit near it, distorting that
cell's fitted shape. This artifact was confirmed DIC-specific (not present in
the Chlorophyll/BODIPY channels at the same pixel location), so the
correction is applied only to the image used for segmentation, not to
fluorescence quantification.

A straight-line convex-hull solidity penalizes genuinely curved (but
otherwise clean) fusiform cells, since bending alone moves area away from the
hull without indicating anything wrong with the cell. A component that fails
the solidity cutoff gets a second chance via has_body_branch: skeletonize its
mask, find its two tips via geodesic distance (robust to curvature and to
both sharp- and blunt-tip skeletonization artifacts), and check whether any
skeleton branch point sits far from both tips. A bent-but-clean cell's
skeleton has no such branch; a genuinely bad component (two cells merged, a
piece of debris fused on) does. This can only ADD acceptances, never remove
one: anything that already passes the solidity cutoff is accepted regardless
of its skeleton, so this cannot regress an already-good detection -- see
project conversation for the calibration data (dataset-wide false accept/
reject rates against known curved and known malformed components).

For each accepted cell, "total" fluorescence is the sum of raw pixel
intensity within the cell mask; "average" is that total divided by the
cell's pixel area (i.e. mean intensity per pixel in the cell).

Lipid bodies are counted by smoothing the BODIPY channel, Otsu-thresholding
it within each cell mask (per cell, not per FOV -- each cell gets its own
bright/dim cutoff from its own pixels), and watershed-splitting the resulting
bright mask at local intensity peaks (skimage.feature.peak_local_max +
skimage.segmentation.watershed) before counting connected components above a
minimum size. Plain intensity thresholding alone can't separate two droplets
that are touching if the valley between their brightness peaks never dips
below the per-cell threshold -- confirmed on this dataset (see project
conversation): a single 1685px "blob" in one cell was visibly 2-3 distinct
ring-shaped droplets in the raw BODIPY. Watershed finds one seed per local
intensity peak and floods outward from each, splitting at the ridge between
them; a single-peak blob watersheds right back to itself, so this only helps
merged cases, it doesn't change anything for already-distinct droplets.
WATERSHED_MIN_DISTANCE_PX controls how close two peaks can be before they're
treated as one (not yet tuned dataset-wide -- see project conversation for
the tradeoff between under- and over-splitting on ring-shaped droplets).

Plastids in the Chlorophyll channel are counted the same way (count_plastids,
sharing count_bright_blobs' logic with count_lipid_bodies). P. tricornutum
normally carries a single plastid that duplicates before the cell divides, so
>1 detected plastid is a candidate marker for a dividing cell -- this hasn't
been validated against any ground-truth dividing-cell annotation yet.

count_bright_blobs' watershed peak search is restricted to the DIC-derived
cell mask, which assumes the fluorescence channel is already registered to
DIC. This script's own main() does NOT apply that registration correction (see
quantify_cells_shifted.py, which does, via correct_fluorescence_registration
below) -- without it, a real peak that sits just past the mask edge (most
likely near a cell's tapered tip, where the mask is narrowest) gets clipped
and reported at the mask boundary instead of its true location. Confirmed on
this dataset: a plastid's true peak sat 11px outside the mask at exactly the
point where the mask tapered, and applying the calibrated registration shift
moved it into a part of the mask over 100px wide, resolving the clip.

Use --days/--reps to restrict the analysis to specific Day/replicate numbers,
parsed from the "<Condition>_Day<N>_rep<M>" filename prefix (e.g. to analyze
only Day 3 replicates 1-2, pass --days 3 --reps 1,2). This changes ONLY which
FOVs are read -- it has no effect on the segmentation or measurement logic,
and condition/mean-computation is always over whatever set of FOVs is passed
in, so filtering to a subset is equivalent to re-running the whole analysis
on that subset (not a post-hoc filter of a full-dataset result). One
exception: the DIC background correction above is always estimated from
every DIC image in input_dir regardless of --days/--reps, since it estimates
a fixed instrument artifact, not anything biological -- narrowing the FOVs
being measured shouldn't narrow the sample used to characterize the camera.

Usage:
    python quantify_cells.py <input_dir> <output_dir> [--days 3] [--reps 1,2]
"""

import argparse
import glob
import os
import re
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import tifffile
import scipy.ndimage as ndi
from scipy.spatial import ConvexHull
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANNELS = ["DIC", "Chlorophyll", "BODIPY"]
FOV_FILE_RE = re.compile(r"^(?P<prefix>.+?)FOV(?P<fov>\d+)_(?P<channel>DIC|Chlorophyll|BODIPY)\.tiff$")
PREFIX_DAY_REP_RE = re.compile(r"_Day(?P<day>\d+)_rep(?P<rep>\d+)$")

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

SKELETON_PRUNE_ITER = 10     # strips spurs shorter than this (px) -- removes sharp/blunt tip artifacts
SKELETON_TIP_MARGIN_PX = 20  # branch points within this geodesic distance of a tip are tip noise, not defects

LIPID_SMOOTH_SIGMA = 1.0
LIPID_MIN_SIZE_PX = 3
LIPID_WATERSHED_MIN_DISTANCE_PX = 3

# Plastid counting (Chlorophyll channel) shares count_bright_blobs with lipid-body
# counting, but has no calibrated precedent of its own yet -- these starting values
# just mirror the lipid-body defaults (see project conversation).
PLASTID_SMOOTH_SIGMA = 1.0
PLASTID_MIN_SIZE_PX = 3
PLASTID_WATERSHED_MIN_DISTANCE_PX = 3

# Fluorescence-to-DIC registration offset -- the outlier-trimmed median of per-cell
# centroid offsets between DIC-derived chloroplast position and Chlorophyll-channel
# autofluorescence centroid, computed across all 188 cells (see
# quantify_cells_shifted.py, the script this was originally calibrated in). Applied
# to both Chlorophyll and BODIPY under the assumption they share the fluorescence
# path's offset -- unverified for BODIPY specifically, since it has no DIC-visible
# structural analog. Without this correction, a peak-finding search restricted to
# the DIC-derived cell mask (e.g. count_bright_blobs' watershed seeding) can clip a
# real fluorescence peak that sits just past the mask edge, particularly near a
# cell's tapered tip where the mask is narrowest -- confirmed on this dataset (see
# project conversation).
FLUORESCENCE_SHIFT_DY_PX = 7
FLUORESCENCE_SHIFT_DX_PX = -1

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


def compute_dic_background(dic_paths):
    """Per-pixel median across every given DIC image. Real cells occupy different
    pixels from FOV to FOV and get suppressed by the median; a stationary optical
    artifact (e.g. a dust speck) appears at the same pixels in every FOV and survives,
    so this map isolates the fixed pattern rather than any single FOV's biology."""
    stack = np.stack([tifffile.imread(p).astype(np.float64) for p in dic_paths])
    return np.median(stack, axis=0)


def correct_dic_background(dic_img, background):
    """Subtract the fixed-pattern deviation of `background` from its own median
    level, flattening stationary artifacts while leaving genuine per-FOV structure
    (real illumination, real cells) untouched."""
    return dic_img.astype(np.float64) - (background - np.median(background))


def correct_fluorescence_registration(channel_img):
    """Translate a fluorescence channel (Chlorophyll or BODIPY) by the calibrated
    offset that aligns it to the DIC-derived cell mask -- see
    FLUORESCENCE_SHIFT_DY_PX/FLUORESCENCE_SHIFT_DX_PX above for how this was
    measured and why it matters for anything that searches within the DIC mask
    (e.g. count_bright_blobs' watershed peak seeding)."""
    return ndi.shift(channel_img.astype(np.float64),
                      shift=(FLUORESCENCE_SHIFT_DY_PX, FLUORESCENCE_SHIFT_DX_PX), order=1)


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


def prune_skeleton(skel, n_iter):
    """Iteratively strip skeleton endpoints n_iter times, removing any spur
    shorter than n_iter px -- discards tiny noise whiskers before topology checks."""
    skel = skel.copy()
    kernel = np.ones((3, 3))
    for _ in range(n_iter):
        neighbor_count = ndi.convolve(skel.astype(int), kernel, mode="constant") - skel.astype(int)
        endpoints = skel & (neighbor_count == 1)
        if not endpoints.any():
            break
        skel = skel & ~endpoints
    return skel


def _geodesic_distances(skel, start):
    """BFS distance (in px, along the skeleton) from `start` to every other skeleton pixel."""
    dist = -np.ones(skel.shape, dtype=int)
    sy, sx = start
    dist[sy, sx] = 0
    queue = deque([(sy, sx)])
    while queue:
        y, x = queue.popleft()
        d = dist[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx] and dist[ny, nx] == -1:
                    dist[ny, nx] = d + 1
                    queue.append((ny, nx))
    return dist


def _find_tips(skel):
    """The skeleton's two tips via the standard double-BFS-sweep diameter heuristic:
    farthest point from an arbitrary start, then farthest point from that. Robust to
    blunt cell ends, where the true tip can have degree >1 in the raster skeleton (a
    fork artifact of the medial-axis transform), so it can't reliably be found by
    looking for degree-1 pixels alone."""
    any_point = tuple(np.argwhere(skel)[0])
    dist_from_any = _geodesic_distances(skel, any_point)
    tip_a = tuple(np.unravel_index(np.argmax(np.where(skel, dist_from_any, -1)), skel.shape))
    dist_from_a = _geodesic_distances(skel, tip_a)
    tip_b = tuple(np.unravel_index(np.argmax(np.where(skel, dist_from_a, -1)), skel.shape))
    dist_from_b = _geodesic_distances(skel, tip_b)
    return dist_from_a, dist_from_b


def has_body_branch(mask):
    """True if `mask`'s skeleton has a branch point far from both of its tips -- a
    real attached defect (e.g. two cells merged, debris fused on), as opposed to a
    skeletonization-noise fork at a sharp or blunt tip. Uses geodesic (along-skeleton)
    distance throughout, so a bent cell's tips are found correctly regardless of how
    much it curves -- this is what lets a merely-curved cell pass while a genuinely
    malformed one still fails."""
    skel = prune_skeleton(skeletonize(mask), SKELETON_PRUNE_ITER)
    kernel = np.ones((3, 3))
    neighbor_count = ndi.convolve(skel.astype(int), kernel, mode="constant") - skel.astype(int)
    neighbor_count = neighbor_count * skel
    branch_points = list(zip(*np.where((neighbor_count >= 3) & skel)))
    if not branch_points:
        return False

    dist_from_a, dist_from_b = _find_tips(skel)
    return any(
        min(dist_from_a[by, bx], dist_from_b[by, bx]) > SKELETON_TIP_MARGIN_PX
        for by, bx in branch_points
    )


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
            local_mask = np.zeros((ys.max() - ys.min() + 3, xs.max() - xs.min() + 3), dtype=bool)
            local_mask[ys - ys.min() + 1, xs - xs.min() + 1] = True
            if has_body_branch(local_mask):
                continue

        yield ys, xs, dict(
            area_px=area, length_um=length_um, width_um=width_um,
            aspect_ratio=aspect_ratio, solidity=solidity,
        )


def count_bright_blobs(channel_smooth, ys, xs, min_size_px, watershed_min_distance):
    """Number of distinct bright blobs within a cell mask, in an already-smoothed
    fluorescence channel. Per-cell Otsu threshold (not a global one -- each cell gets
    its own bright/dim cutoff from its own pixels), then watershed-split at local
    intensity peaks before counting connected components above min_size_px: two
    touching bright regions whose valley never dips below the per-cell threshold
    still get separated instead of merged into one count. A blob with only one local
    peak watersheds right back to itself unchanged."""
    cell_mask = np.zeros(channel_smooth.shape, dtype=bool)
    cell_mask[ys, xs] = True
    thresh = otsu_threshold(channel_smooth[ys, xs])
    bright_mask = cell_mask & (channel_smooth > thresh)

    coords = peak_local_max(channel_smooth, min_distance=watershed_min_distance, labels=bright_mask.astype(int))
    if len(coords) == 0:
        labeled, n = ndi.label(bright_mask, structure=np.ones((3, 3)))
    else:
        markers = np.zeros(channel_smooth.shape, dtype=int)
        markers[coords[:, 0], coords[:, 1]] = np.arange(1, len(coords) + 1)
        labeled = watershed(-channel_smooth, markers=markers, mask=bright_mask, connectivity=2)
        n = int(labeled.max())

    if n == 0:
        return 0
    sizes = ndi.sum(bright_mask, labeled, index=np.arange(1, n + 1))
    return int(np.sum(sizes >= min_size_px))


def count_lipid_bodies(bodipy_smooth, ys, xs):
    """Number of distinct BODIPY-bright lipid bodies within a cell mask."""
    return count_bright_blobs(bodipy_smooth, ys, xs, LIPID_MIN_SIZE_PX, LIPID_WATERSHED_MIN_DISTANCE_PX)


def count_plastids(chlorophyll_smooth, ys, xs):
    """Number of distinct Chlorophyll-bright plastids within a cell mask. P.
    tricornutum normally carries a single plastid that duplicates before the cell
    divides, so >1 here is a candidate marker for a dividing cell (not yet validated
    against ground truth -- see project conversation)."""
    return count_bright_blobs(chlorophyll_smooth, ys, xs, PLASTID_MIN_SIZE_PX, PLASTID_WATERSHED_MIN_DISTANCE_PX)


def group_fovs(directory):
    fovs = defaultdict(dict)
    for path in glob.glob(os.path.join(directory, "*.tiff")):
        match = FOV_FILE_RE.match(os.path.basename(path))
        if match:
            key = (match.group("prefix"), int(match.group("fov")))
            fovs[key][match.group("channel")] = path
    return fovs


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


def filter_fovs_by_day_rep(fovs, days=None, reps=None):
    """Keep only (prefix, fov_num) keys whose parsed day/rep are in the given allow-lists (None = no filter)."""
    if days is None and reps is None:
        return fovs
    filtered = {}
    for key, channel_paths in fovs.items():
        prefix, _fov_num = key
        day, rep = parse_day_rep(prefix)
        if days is not None and day not in days:
            continue
        if reps is not None and rep not in reps:
            continue
        filtered[key] = channel_paths
    return filtered


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
    parser.add_argument("--days", type=str, default=None,
                         help="Comma-separated Day numbers to include, e.g. '3' or '1,3' (default: all days present)")
    parser.add_argument("--reps", type=str, default=None,
                         help="Comma-separated replicate numbers to include, e.g. '1,2' (default: all reps present)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    qc_dir = os.path.join(args.output_dir, "qc_overlays")
    if args.qc_overlays:
        os.makedirs(qc_dir, exist_ok=True)

    all_fovs = group_fovs(args.input_dir)
    if not all_fovs:
        raise SystemExit(f"No FOVs found in {args.input_dir}")
    dic_paths = [paths["DIC"] for paths in all_fovs.values() if "DIC" in paths]
    dic_background = compute_dic_background(dic_paths)
    print(f"Computed DIC background from {len(dic_paths)} FOV(s) in {args.input_dir}")
    plt.figure(figsize=(8, 6))
    plt.imshow(dic_background, cmap="gray")
    plt.title("DIC background (per-pixel median across all FOVs)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "dic_background.png"), dpi=150)
    plt.close()

    days, reps = parse_int_list(args.days), parse_int_list(args.reps)
    rows = []
    fovs = filter_fovs_by_day_rep(all_fovs, days=days, reps=reps)
    if not fovs:
        raise SystemExit(f"No FOVs matched --days={args.days} --reps={args.reps} in {args.input_dir}")
    for (prefix, fov_num), channel_paths in sorted(fovs.items()):
        missing = [c for c in CHANNELS if c not in channel_paths]
        if missing:
            print(f"SKIPPING {prefix} FOV{fov_num}: missing {missing}")
            continue

        condition = prefix.split("_Day")[0]
        dic = tifffile.imread(channel_paths["DIC"])
        dic_corrected = correct_dic_background(dic, dic_background)
        chl = tifffile.imread(channel_paths["Chlorophyll"]).astype(np.float64)
        bod = tifffile.imread(channel_paths["BODIPY"]).astype(np.float64)

        labeled, n_components = segment_dic(dic_corrected)
        cells = list(accepted_cells(labeled, n_components, dic.shape))
        bod_smooth = ndi.gaussian_filter(bod, sigma=LIPID_SMOOTH_SIGMA)
        chl_smooth = ndi.gaussian_filter(chl, sigma=PLASTID_SMOOTH_SIGMA)

        for cell_id, (ys, xs, props) in enumerate(cells, start=1):
            total_chl, total_bod = chl[ys, xs].sum(), bod[ys, xs].sum()
            rows.append(dict(
                condition=condition, sample=prefix, fov=fov_num, cell_id=cell_id,
                **props,
                total_chlorophyll=total_chl, total_bodipy=total_bod,
                avg_chlorophyll=total_chl / props["area_px"],
                avg_bodipy=total_bod / props["area_px"],
                n_lipid_bodies=count_lipid_bodies(bod_smooth, ys, xs),
                n_plastids=count_plastids(chl_smooth, ys, xs),
            ))

        print(f"{prefix} FOV{fov_num}: {len(cells)} cell(s)")
        if args.qc_overlays:
            save_qc_overlay(dic_corrected, cells, os.path.join(qc_dir, f"{prefix}_FOV{fov_num}_qc.png"))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "cell_measurements.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} cell measurements to {csv_path}")
    print(df.groupby("condition").size().reindex(CONDITION_ORDER))

    make_categorical_plot(
        df, "n_lipid_bodies", "Number of lipid bodies per cell",
        os.path.join(args.output_dir, "lipid_bodies_per_cell.png"),
    )
    make_categorical_plot(
        df, "n_plastids", "Number of plastids per cell",
        os.path.join(args.output_dir, "plastids_per_cell.png"),
    )
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
