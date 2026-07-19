#!/usr/bin/env python3
"""
Re-run the chlorophyll/BODIPY/lipid-body quantification from quantify_cells.py,
but sum fluorescence within a DILATED version of each DIC-derived cell mask
rather than the tight mask itself.

Why: the fluorescence channels are shifted relative to DIC by a consistent
~12-13 px (see registration diagnostic in conversation), so a tight DIC mask
can clip real signal that has moved just outside the cell outline. Rather
than shifting images (which requires nailing an exact sign/magnitude, and
per-cell motion or focal-plane differences add noise anyway), this expands
the mask by a fixed radius so that shifted signal near the cell is still
captured. Segmentation/morphology filtering is unchanged; only the pixels
summed for fluorescence differ.

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
    CHANNELS, segment_dic, accepted_cells, group_fovs, disk,
    count_lipid_bodies, LIPID_SMOOTH_SIGMA, CONDITION_ORDER, make_categorical_plot,
    parse_int_list, filter_fovs_by_day_rep,
)

DILATE_RADIUS_PX = 15


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--dilate-radius", type=int, default=DILATE_RADIUS_PX)
    parser.add_argument("--days", type=str, default=None,
                         help="Comma-separated Day numbers to include, e.g. '3' or '1,3' (default: all days present)")
    parser.add_argument("--reps", type=str, default=None,
                         help="Comma-separated replicate numbers to include, e.g. '1,2' (default: all reps present)")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    struct = disk(args.dilate_radius)

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

        labeled, _n_components = segment_dic(dic)
        cells = list(accepted_cells(labeled, dic.shape))
        bod_smooth = ndi.gaussian_filter(bod, sigma=LIPID_SMOOTH_SIGMA)

        for cell_id, (ys, xs, props) in enumerate(cells, start=1):
            tight_mask = np.zeros(dic.shape, dtype=bool)
            tight_mask[ys, xs] = True
            dilated_mask = ndi.binary_dilation(tight_mask, structure=struct)
            dys, dxs = np.where(dilated_mask)

            total_chl, total_bod = chl[dys, dxs].sum(), bod[dys, dxs].sum()
            dilated_area = len(dys)
            rows.append(dict(
                condition=condition, sample=prefix, fov=fov_num, cell_id=cell_id,
                area_px=props["area_px"], dilated_area_px=dilated_area,
                total_chlorophyll=total_chl, total_bodipy=total_bod,
                avg_chlorophyll=total_chl / dilated_area,
                avg_bodipy=total_bod / dilated_area,
                n_lipid_bodies=count_lipid_bodies(bod_smooth, dys, dxs),
            ))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "cell_measurements_dilated.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} cell measurements (dilate radius={args.dilate_radius}px) to {csv_path}")
    print(df.groupby("condition").size().reindex(CONDITION_ORDER))

    make_categorical_plot(df, "avg_chlorophyll", "Mean chlorophyll fluorescence per cell (a.u., dilated mask)",
                           os.path.join(args.output_dir, "chlorophyll_per_cell_dilated.png"))
    make_categorical_plot(df, "avg_bodipy", "Mean BODIPY fluorescence per cell (a.u., dilated mask)",
                           os.path.join(args.output_dir, "bodipy_per_cell_dilated.png"))
    make_categorical_plot(df, "n_lipid_bodies", "Number of lipid bodies per cell (dilated mask)",
                           os.path.join(args.output_dir, "lipid_bodies_per_cell_dilated.png"))
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
