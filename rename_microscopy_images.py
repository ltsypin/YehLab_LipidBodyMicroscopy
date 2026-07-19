#!/usr/bin/env python3
"""
Rename raw microscopy images from:
    "<Condition_Day_replicate>captured layer X.tiff"
to:
    "<Condition_Day_replicate>FOV<n>_<Channel>.tiff"

Each sample (Condition_Day_replicate) contains one or more fields of view
(FOVs), each made up of exactly three consecutive layers: DIC, then
Chlorophyll, then BODIPY. Layer numbers start at 3 and increase sequentially,
so sorting a sample's layers and chunking them into groups of three recovers
the FOVs in order.

Use --days/--reps to rename only specific Day/replicate numbers, parsed from
the "<Condition>_Day<N>_rep<M>" prefix, e.g. --days 3 --reps 1,2. Samples
outside the filter are left untouched (not even the divisible-by-3 check
runs on them).

Usage:
    python rename_microscopy_images.py <directory> [--dry-run] [--days 3] [--reps 1,2]
"""

import argparse
import glob
import os
import re
from collections import defaultdict

LAYER_FILE_RE = re.compile(r"^(?P<prefix>.+?)captured layer (?P<layer>\d+)(?P<ext>\.tiff)$")
PREFIX_DAY_REP_RE = re.compile(r"_Day(?P<day>\d+)_rep(?P<rep>\d+)$")
CHANNELS = ["DIC", "Chlorophyll", "BODIPY"]


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


def group_samples(directory):
    """Map each sample prefix in `directory` to its sorted (layer_num, path) files."""
    samples = defaultdict(list)
    for path in glob.glob(os.path.join(directory, "*captured layer *.tiff")):
        match = LAYER_FILE_RE.match(os.path.basename(path))
        if not match:
            continue
        samples[match.group("prefix")].append((int(match.group("layer")), path))
    for prefix in samples:
        samples[prefix].sort(key=lambda layer_and_path: layer_and_path[0])
    return samples


def rename_sample(prefix, files, dry_run):
    if len(files) % 3 != 0:
        print(f"  SKIPPING '{prefix}': {len(files)} files is not divisible by 3")
        return

    for fov_index in range(len(files) // 3):
        triplet = files[fov_index * 3 : fov_index * 3 + 3]
        for channel, (layer_num, path) in zip(CHANNELS, triplet):
            ext = os.path.splitext(path)[1]
            new_name = f"{prefix}FOV{fov_index + 1}_{channel}{ext}"
            new_path = os.path.join(os.path.dirname(path), new_name)
            action = "Would rename" if dry_run else "Renamed"
            print(f"  {action}: {os.path.basename(path)} -> {new_name}")
            if not dry_run:
                os.rename(path, new_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", help="Directory containing the raw .tiff files")
    parser.add_argument("--dry-run", action="store_true", help="Preview renames without changing files")
    parser.add_argument("--days", type=str, default=None,
                         help="Comma-separated Day numbers to rename, e.g. '3' or '1,3' (default: all days present)")
    parser.add_argument("--reps", type=str, default=None,
                         help="Comma-separated replicate numbers to rename, e.g. '1,2' (default: all reps present)")
    args = parser.parse_args()

    days, reps = parse_int_list(args.days), parse_int_list(args.reps)
    samples = group_samples(args.directory)
    if days is not None or reps is not None:
        samples = {
            prefix: files for prefix, files in samples.items()
            if (days is None or parse_day_rep(prefix)[0] in days)
            and (reps is None or parse_day_rep(prefix)[1] in reps)
        }
    if not samples:
        print(f"No 'captured layer' files found in {args.directory} matching --days={args.days} --reps={args.reps}")
        return

    for prefix in sorted(samples):
        files = samples[prefix]
        print(f"Sample '{prefix}': {len(files)} files")
        rename_sample(prefix, files, args.dry_run)


if __name__ == "__main__":
    main()
