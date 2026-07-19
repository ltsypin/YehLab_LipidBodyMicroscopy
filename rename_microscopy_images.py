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

Usage:
    python rename_microscopy_images.py <directory> [--dry-run]
"""

import argparse
import glob
import os
import re
from collections import defaultdict

LAYER_FILE_RE = re.compile(r"^(?P<prefix>.+?)captured layer (?P<layer>\d+)(?P<ext>\.tiff)$")
CHANNELS = ["DIC", "Chlorophyll", "BODIPY"]


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
    args = parser.parse_args()

    samples = group_samples(args.directory)
    if not samples:
        print(f"No 'captured layer' files found in {args.directory}")
        return

    for prefix in sorted(samples):
        files = samples[prefix]
        print(f"Sample '{prefix}': {len(files)} files")
        rename_sample(prefix, files, args.dry_run)


if __name__ == "__main__":
    main()
