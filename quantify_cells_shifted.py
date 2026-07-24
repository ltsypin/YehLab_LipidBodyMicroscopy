#!/usr/bin/env python3
"""
Re-run the chlorophyll/BODIPY/lipid-body quantification from quantify_cells.py,
correcting the fluorescence-to-DIC registration shift by translating the
fluorescence images themselves, then summing within the ORIGINAL tight
DIC-derived cell mask (no dilation).

DIC segmentation now matches quantify_cells.py's current pipeline exactly:
the fixed-pattern background is subtracted (compute_dic_background/
correct_dic_background) before segment_dic, and accepted_cells' solidity
check gets its curvature-tolerant second chance (has_body_branch) for free,
since that logic lives inside accepted_cells itself. Previously this script
called segment_dic on the raw, uncorrected DIC image, so its cell population
(and cell_id numbering) diverged from quantify_cells.py's -- they should
match now, modulo whatever --days/--reps filtering each run uses.

The shift (dy=7, dx=-1 px) is the outlier-trimmed median of per-cell centroid
offsets between DIC-derived chloroplast position and Chlorophyll-channel
autofluorescence centroid, computed across all 188 cells (excluding other
cells' pixels from each local window to avoid neighbor contamination). This
superseded an earlier whole-FOV cross-correlation estimate (dy=12, dx=-3):
that value visibly overshot for axis-aligned cells (autofluorescence went
from clipping one edge of the mask to clipping the opposite edge) since it
exceeded a typical cell's ~5 um width, whereas the per-cell centroid estimate
does not. The correction is applied as a single fixed value to every FOV,
since the registration offset should be a systematic property of the optical
path, not something to re-estimate per FOV (per-FOV estimates are noisier due
to cell motion/focal-plane differences between sequential channel captures).
The same correction is applied to BODIPY under the assumption it shares the
fluorescence path's offset; this is unverified for BODIPY specifically, since
it has no DIC-visible structural analog.

This registration correction matters beyond just the raw fluorescence sums:
count_lipid_bodies/count_plastids (quantify_cells.py) search for watershed
seeds only within the DIC-derived cell mask, so a real intensity peak sitting
just past the mask edge -- most likely near a cell's tapered tip, where the
mask is narrowest -- gets clipped and reported at the wrong location if the
fluorescence channel isn't first brought into registration with DIC. This is
why n_plastids/n_lipid_bodies are computed here from the shift-corrected
channels, not quantify_cells.py's uncorrected ones.

Use --days/--reps to restrict to specific Day/replicate numbers (same
behavior as quantify_cells.py's --days/--reps -- see that script's
docstring). As in quantify_cells.py, the DIC background is always estimated
from every DIC image in input_dir regardless of --days/--reps, since it
estimates a fixed instrument artifact, not anything biological.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import tifffile
import scipy.ndimage as ndi

from quantify_cells import (
    CHANNELS, segment_dic, accepted_cells, group_fovs,
    compute_dic_background, correct_dic_background,
    count_lipid_bodies, count_plastids, compute_focus_score,
    LIPID_SMOOTH_SIGMA, PLASTID_SMOOTH_SIGMA, PLASTID_MIN_FOCUS_SCORE,
    FLUORESCENCE_SHIFT_DY_PX, FLUORESCENCE_SHIFT_DX_PX,
    CONDITION_ORDER, make_categorical_plot, parse_int_list, filter_fovs_by_day_rep,
    load_flagged_cell_keys, exclude_flagged_cells,
)

import matplotlib.pyplot as plt  # after quantify_cells import -- that's where matplotlib.use("Agg") happens

SHIFT_DY, SHIFT_DX = FLUORESCENCE_SHIFT_DY_PX, FLUORESCENCE_SHIFT_DX_PX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--shift-dy", type=float, default=SHIFT_DY)
    parser.add_argument("--shift-dx", type=float, default=SHIFT_DX)
    parser.add_argument("--days", type=str, default=None,
                         help="Comma-separated Day numbers to include, e.g. '3' or '1,3' (default: all days present)")
    parser.add_argument("--reps", type=str, default=None,
                         help="Comma-separated replicate numbers to include, e.g. '1,2' (default: all reps present)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
            continue

        condition = prefix.split("_Day")[0]
        dic = tifffile.imread(channel_paths["DIC"])
        dic_corrected = correct_dic_background(dic, dic_background)
        chl = tifffile.imread(channel_paths["Chlorophyll"]).astype(np.float64)
        bod = tifffile.imread(channel_paths["BODIPY"]).astype(np.float64)

        chl_corr = ndi.shift(chl, shift=(args.shift_dy, args.shift_dx), order=1)
        bod_corr = ndi.shift(bod, shift=(args.shift_dy, args.shift_dx), order=1)

        labeled, n_components = segment_dic(dic_corrected)
        cells = list(accepted_cells(labeled, n_components, dic.shape))
        bod_corr_smooth = ndi.gaussian_filter(bod_corr, sigma=LIPID_SMOOTH_SIGMA)
        chl_corr_smooth = ndi.gaussian_filter(chl_corr, sigma=PLASTID_SMOOTH_SIGMA)

        for cell_id, (ys, xs, props) in enumerate(cells, start=1):
            total_chl, total_bod = chl_corr[ys, xs].sum(), bod_corr[ys, xs].sum()
            focus_score = compute_focus_score(chl_corr, ys, xs)
            rows.append(dict(
                condition=condition, sample=prefix, fov=fov_num, cell_id=cell_id,
                **props,
                total_chlorophyll=total_chl, total_bodipy=total_bod,
                avg_chlorophyll=total_chl / props["area_px"],
                avg_bodipy=total_bod / props["area_px"],
                n_lipid_bodies=count_lipid_bodies(bod_corr_smooth, ys, xs),
                n_plastids=count_plastids(chl_corr_smooth, ys, xs),
                chlorophyll_focus_score=focus_score,
                in_focus=focus_score >= PLASTID_MIN_FOCUS_SCORE,
            ))

    df = pd.DataFrame(rows)
    flagged_path = os.path.join(args.output_dir, "flagged_rois.csv")
    df = exclude_flagged_cells(df, load_flagged_cell_keys(flagged_path), flagged_path)

    csv_path = os.path.join(args.output_dir, "cell_measurements_shift_corrected.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} cell measurements (shift=({args.shift_dy},{args.shift_dx})) to {csv_path}")
    print(df.groupby("condition").size().reindex(CONDITION_ORDER))

    make_categorical_plot(df, "avg_chlorophyll", "Mean chlorophyll fluorescence per cell (a.u., shift-corrected)",
                           os.path.join(args.output_dir, "chlorophyll_per_cell_shift_corrected.png"))
    make_categorical_plot(df, "avg_bodipy", "Mean BODIPY fluorescence per cell (a.u., shift-corrected)",
                           os.path.join(args.output_dir, "bodipy_per_cell_shift_corrected.png"))
    make_categorical_plot(df, "n_lipid_bodies", "Number of lipid bodies per cell (shift-corrected)",
                           os.path.join(args.output_dir, "lipid_bodies_per_cell_shift_corrected.png"))
    make_categorical_plot(df, "n_plastids", "Number of plastids per cell (shift-corrected)",
                           os.path.join(args.output_dir, "plastids_per_cell_shift_corrected.png"))
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
