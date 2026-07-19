#!/usr/bin/env python3
"""
Re-run the chlorophyll/BODIPY/lipid-body quantification from quantify_cells.py,
correcting the fluorescence-to-DIC registration shift by translating the
fluorescence images themselves, then summing within the ORIGINAL tight
DIC-derived cell mask (no dilation).

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

Use --days/--reps to restrict to specific Day/replicate numbers (same
behavior as quantify_cells.py's --days/--reps -- see that script's docstring).
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
    count_lipid_bodies, LIPID_SMOOTH_SIGMA, CONDITION_ORDER, make_categorical_plot,
    parse_int_list, filter_fovs_by_day_rep,
)

SHIFT_DY, SHIFT_DX = 7, -1


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

    days, reps = parse_int_list(args.days), parse_int_list(args.reps)
    rows = []
    fovs = filter_fovs_by_day_rep(group_fovs(args.input_dir), days=days, reps=reps)
    if not fovs:
        raise SystemExit(f"No FOVs matched --days={args.days} --reps={args.reps} in {args.input_dir}")
    for (prefix, fov_num), channel_paths in sorted(fovs.items()):
        missing = [c for c in CHANNELS if c not in channel_paths]
        if missing:
            continue

        condition = prefix.split("_Day")[0]
        dic = tifffile.imread(channel_paths["DIC"])
        chl = tifffile.imread(channel_paths["Chlorophyll"]).astype(np.float64)
        bod = tifffile.imread(channel_paths["BODIPY"]).astype(np.float64)

        chl_corr = ndi.shift(chl, shift=(args.shift_dy, args.shift_dx), order=1)
        bod_corr = ndi.shift(bod, shift=(args.shift_dy, args.shift_dx), order=1)

        labeled, _n_components = segment_dic(dic)
        cells = list(accepted_cells(labeled, dic.shape))
        bod_corr_smooth = ndi.gaussian_filter(bod_corr, sigma=LIPID_SMOOTH_SIGMA)

        for cell_id, (ys, xs, props) in enumerate(cells, start=1):
            total_chl, total_bod = chl_corr[ys, xs].sum(), bod_corr[ys, xs].sum()
            rows.append(dict(
                condition=condition, sample=prefix, fov=fov_num, cell_id=cell_id,
                **props,
                total_chlorophyll=total_chl, total_bodipy=total_bod,
                avg_chlorophyll=total_chl / props["area_px"],
                avg_bodipy=total_bod / props["area_px"],
                n_lipid_bodies=count_lipid_bodies(bod_corr_smooth, ys, xs),
            ))

    df = pd.DataFrame(rows)
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
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
