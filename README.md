# P. tricornutum nitrogen-source microscopy analysis

Quantifies chlorophyll and BODIPY (lipid droplet) fluorescence, and lipid body
counts, in *Phaeodactylum tricornutum* cells imaged by DIC + two fluorescence
channels, across three nitrogen-source conditions (Nitrate, Arginine, Urea).
This file documents how to reproduce the analysis
end-to-end and, more importantly, *why* each non-obvious choice was made, so
it can be turned into a proper version-controlled repo.

Treat every numeric threshold below as calibrated against *this* dataset by
inspection, not as a biological constant — re-check them against QC overlays
before reusing this pipeline on new data.

**Filtering by Day/replicate**: every script below (`rename_microscopy_images.py`,
`composite_figure.py`, `quantify_cells.py`, `quantify_cells_dilated.py`,
`quantify_cells_shifted.py`) accepts `--days` and `--reps` (comma-separated
integers, e.g. `--days 3 --reps 1,2`), parsed from the
`<Condition>_Day<N>_rep<M>` filename prefix. Omitting a flag means "no
filter, include everything present." This restricts which files/FOVs are
processed — it does not change any segmentation, normalization, or
measurement logic — except in `composite_figure.py`, where the filter also
restricts which images feed the global per-channel min/max used for
normalization (see Step 2), so a filtered composite-figure run is a
self-contained analysis of that subset with its own normalization baseline,
not a partial regeneration against a baseline computed elsewhere.

## Directory layout

```
<YYYYMMDD>_Day<N>/        one folder per acquisition date + Day number
  *.liff                  raw acquisition files for that date+Day (read-only, never modified)
  tiffs/                  raw microscope exports for that date+Day (read-only, never modified)
renamed_composites/       renamed per-FOV TIFFs + composite QC figures (Steps 1-2)
quantification/           per-cell measurements + plots (Steps 3-5)
rename_microscopy_images.py
composite_figure.py
quantify_cells.py
quantify_cells_dilated.py     (rejected approach, kept for reference — see Step 4)
quantify_cells_shifted.py     (adopted approach — see Step 4)
qc_review_app.py             (interactive segmentation QC tool — see "QC review tool" below)
```

Raw `.liff` files and their derived TIFF exports are grouped into a
`<YYYYMMDD>_Day<N>/` folder per acquisition session, where `YYYYMMDD` is the
`.liff` files' creation date and `Day<N>` is parsed from their filenames
(e.g. `20260625_Day3/`, from `.liff` files created 2026-06-25, all named
`..._Day3_...`). This dataset currently has only one such folder since all 9
`.liff` files share one creation date and Day number; expect more as
additional acquisition sessions are added.

Raw data convention within each `tiffs/` folder: each sample prefix is named
`<Condition>_Day<N>_rep<M>`, and contains one TIFF per z-layer named
`<prefix>captured layer <X>.tiff`, plus one unused `<prefix>Original Image.tiff`.
Every 3 consecutive layer numbers are one field of view (FOV): the 1st is
DIC, the 2nd is Chlorophyll, the 3rd is BODIPY (this ordering is a fact about
how the microscope was operated, stated by the experimenter — it is not
detected from the images).

## Environment

This project has a dedicated conda/mamba environment, `microscopy.env`,
pinned in `environment.yml`:

```
mamba env create -f environment.yml
mamba activate microscopy.env
```

Core dependencies (actually imported by the scripts, everything below is
stdlib otherwise): `numpy`, `scipy`, `pandas`, `tifffile`, `matplotlib`,
`bokeh` and `scikit-image` (the last two used only by `qc_review_app.py`,
see below — `scikit-image` there just for `skimage.measure.find_contours`,
to draw cell mask boundaries).

None of the core pipeline scripts (`quantify_cells.py` etc.) use
scikit-image yet. All the classical CV they currently do (Otsu threshold,
connected components, ellipse fitting, convex hull) was hand-implemented on
top of `scipy.ndimage` / `numpy` / `scipy.spatial.ConvexHull` instead, back
when scikit-image wasn't reliably available in the environment this was
originally developed in (see git history) — now that a real scikit-image
install exists, revisiting those hand-rolled implementations in favor of
`skimage.filters.threshold_otsu`, `skimage.measure.regionprops`, etc. would
be a reasonable simplification, though the two aren't guaranteed to produce
numerically identical results, so re-validate against the QC-reviewed FOVs
(see Step 3) before swapping. (This has since been done on the `b-scikit`
branch.)

**Historical note**: earlier development used a repurposed environment
(`tgne.env`) from an unrelated project, because the base conda environment's
matplotlib install was broken (missing `__init__.py`, so `import matplotlib`
silently produced a non-functional namespace package). `microscopy.env` was
created fresh and doesn't have that problem — if you ever see that failure
mode again, it's specific to that one broken environment, not this project.

## Step 0: Fix known raw-data issues

Two replicates (`Arginine_Day3_rep3`, `Urea_Day3_rep3`) originally had 25
`captured layer` files instead of a multiple of 3, each due to one duplicated
capture. This was caught automatically — `rename_microscopy_images.py` skips
and warns on any sample whose file count isn't divisible by 3 — then fixed by
hand: the duplicate file in each sample was renamed with a `duplicate_`
prefix (e.g. `duplicate_Arginine_Day3_rep3captured layer 17.tiff`) rather
than deleted, so it's preserved on disk but excluded from FOV grouping.
Check for this warning on any new raw data before proceeding.

## Step 1: Rename files into channel-labeled FOV groups

```
python rename_microscopy_images.py <directory> [--dry-run] [--days 3] [--reps 1,2]
```

- Globs `*captured layer *.tiff` in `<directory>`, groups by the
  `<Condition>_Day<N>_rep<M>` prefix, sorts each group by layer number, and
  requires the count to be divisible by 3 (see Step 0).
- Chunks each group into consecutive triplets = one FOV each, renaming
  `<prefix>captured layer N.tiff` → `<prefix>FOV<k>_<Channel>.tiff` for
  `Channel` in `{DIC, Chlorophyll, BODIPY}`.
- Always run `--dry-run` first. In this project the script was run against
  *copies* of the raw files inside `renamed_composites/`, never against
  `tiffs/` directly, to keep the raw data immutable and reproducible from
  scratch.

## Step 2: Composite QC/figure generation

```
python composite_figure.py <input_dir> <output_dir> [--panel-size-in 3.0] [--dpi 300] [--days 3] [--reps 1,2]
```

For every FOV, builds a 2×2 panel PNG: **a** = DIC (grayscale), **b** =
Chlorophyll (magenta), **c** = BODIPY (cyan), **d** = Chlorophyll+BODIPY
overlay.

- **Global normalization**: each channel's pixel intensities are rescaled
  using the min/max of *that channel across every image in the input
  directory* (not per-image), so brightness is directly comparable between
  samples/replicates/FOVs. This was an explicit requirement — exposure was
  held fixed per channel across the whole dataset, so per-image
  normalization would have destroyed real signal-level differences between
  conditions.
- **Panel spacing**: exactly 0.5 mm between panels, computed analytically
  from panel size (inches) and the mm→inch conversion via `GridSpec`
  `wspace`/`hspace` — not eyeballed.
- **Scale bar**: white, 65 px = 10 µm (pixel size = 10/65 µm/px, from the
  microscope's stated calibration), with a "10 µm" label, drawn only in
  panel **a**. Font size settled at 8pt after iteration (12pt and 10pt were
  tried and judged too large).
- This step is independent of quantification — `quantify_cells.py` re-derives
  its own segmentation directly from the renamed TIFFs, it does not consume
  these composite images.

## Step 3: Segmentation + fluorescence/lipid-body quantification

```
python quantify_cells.py <input_dir> <output_dir> [--qc-overlays] [--days 3] [--reps 1,2]
```

### Segmentation (classical CV, DIC channel only)

1. Light Gaussian smoothing (σ=1.0) → Sobel gradient magnitude → Otsu
   threshold (self-implemented, vectorized histogram method — no
   scikit-image) × 0.4 → morphological closing (disk radius 10px; bridges
   the two faint, roughly-parallel edge lines of a thin cell into one filled
   body) → hole filling → morphological opening (disk radius 2px, denoise) →
   connected components (`scipy.ndimage.label`).
2. Each component is fit to an equivalent ellipse via image moments
   (matching scikit-image's `regionprops` major/minor axis formula) and kept
   only if:
   - **length 15–40 µm, width 2–7 µm, aspect ratio 5–16.** These bounds were
     calibrated *empirically against this dataset*, not taken from the
     original stated expectation of 40–60 µm long cells. Measured, cleanly
     single-segmented cells consistently came out ~20–37 µm long across
     multiple FOVs/conditions; using the originally-stated 40–60 µm range
     would have rejected essentially every cell. Confirmed with the user
     before narrowing the range — if you get zero or very few detections on
     new data, check this first.
   - **solidity ≥ 0.75** (mask area / convex hull area), to reject
     non-convex/branching shapes.
   - **not touching the image border.**
   - minimum component area 300 px² (drops speckle noise before the
     (relatively expensive) ellipse/hull fitting).

**Known limitation — merged touching cells**: two cells lying side-by-side
and touching along their length can still pass the aspect-ratio + solidity
filters as a single "cell", since a wide touching pair can still look
convex and elongated. Two such cases were found by visual inspection in
`Arginine_Day3_rep3` (FOV4 cell 4, FOV5 cell 3), identifiable by anomalously
high lipid-body counts (7, 8 vs. a dataset median of ~2). Two automatic
detectors were tried and **both rejected**:
  - DIC intensity profile along the cell's *major* axis, looking for two
    separated density peaks (chloroplast clumps) — too noisy even after
    heavy smoothing (both false positives on clean single cells and false
    negatives on known merges), and structurally can't catch side-by-side
    (as opposed to end-to-end) merges, since two side-by-side clumps project
    to the *same* position along the shared long axis.
  - Same idea along the *minor* (width) axis — better (zero false positives
    on 4 clean test cells) but still missed one of the two known merges; the
    minor axis only spans ~2–7 µm, too few bins to reliably resolve two
    side-by-side blobs.
  - **Current recommendation**: manually flag/exclude these two rows; revisit
    automatic detection only if this failure mode turns out to be common in
    future data (e.g. via a mask-area-vs-expected-single-cell-area check, not
    yet implemented).

### Per-cell measurements (`cell_measurements.csv`)

`condition` (parsed from the filename prefix as everything before `_Day`),
`sample`, `fov`, `cell_id`, `area_px`, `length_um`, `width_um`,
`aspect_ratio`, `solidity`, `total_chlorophyll` / `total_bodipy` (sum of raw
pixel intensity in the tight cell mask), `avg_chlorophyll` / `avg_bodipy`
(total ÷ `area_px`, i.e. mean intensity per pixel), `n_lipid_bodies`.

**Lipid body counting**: Gaussian-smooth the BODIPY channel (σ=1.0) →
per-cell Otsu threshold computed only on that cell's own pixels → connected
components of the resulting bright mask, keeping only components ≥3 px →
count = number of surviving components.
**Known limitation**: droplets that touch/overlap enough to merge into one
connected component are undercounted as a single lipid body. A
local-maxima peak-splitting alternative was tried and rejected: a small
peak-detection footprint produced spurious extra peaks from pixel noise
inside a single droplet; a larger footprint just recreated the same
undercounting problem it was meant to fix. The simple, predictable
connected-component count was kept.

`--qc-overlays` writes one PNG per FOV to `<output_dir>/qc_overlays/`
showing which regions were accepted as cells. **Inspect a sample of these
before trusting results on any new data** — this is how both the length-range
mismatch and the merged-cell issue above were originally caught.

Default plot: `lipid_bodies_per_cell.png` (categorical scatter, one point per
cell, x = condition in order `[Nitrate, Arginine, Urea]`, black bar = mean).
Average/total-intensity plots are *not* produced by this script's `main()` by
default — see Step 5.

### QC review tool (`qc_review_app.py`)

`--qc-overlays` PNGs are useful but static and one-per-FOV; `qc_review_app.py`
is an interactive Bokeh server app for browsing the whole dataset's
segmentation FOV-by-FOV and flagging specific cells as poorly segmented, to
build up a concrete list of failure cases (e.g. for choosing/validating
automated-test fixtures, see the open items below).

```
bokeh serve --show qc_review_app.py --args <input_dir> [<output_dir>]
```

(defaults to `renamed_composites`/`quantification` if omitted). The whole
grid is responsive to the browser window width; panels e/f are always
exactly 2x the width of the a-d row, at any window size. Panels a-d are
deliberately small — they share one row, and rely on panels e/f (plus
Bokeh's own pan/zoom tools) for detailed inspection — but are otherwise the
same DIC/Chlorophyll/BODIPY/overlay composite as `composite_figure.py`,
using the same global per-channel normalization (computed once over
`<input_dir>`, reusing `composite_figure.py`'s own functions directly — not
a re-implementation). Panels e, d, and f each carry the same white 65 px =
10 µm scale bar as `composite_figure.py` (`SCALE_BAR_PX`/`SCALE_BAR_UM`,
imported directly from it), bottom-right, with the "10 µm" label centered
over the bar -- except panel d, which shows the bar only, no label
(`add_scale_bar(..., show_label=False)`).

- **Panel e** is DIC with every accepted cell's mask boundary (from
  `skimage.measure.find_contours` on the exact same tight mask
  `quantify_cells.py` measures) drawn as a clickable region: click a cell to
  flag it red (poor segmentation), click again to unflag. Flags persist as
  you navigate between FOVs via the Previous/Next buttons or the FOV
  dropdown. The same panel's **Freehand Draw** tool (toolbar icon) lets you
  lasso a good cell the pipeline missed entirely; tap it and press
  Backspace/Delete to remove it. (An earlier version used the Box Edit tool
  for rectangular ROIs; it didn't reliably respond to drag gestures in
  testing, so it was replaced with Freehand Draw, which also traces the
  actual cell outline instead of a crude bounding box.)
- **Panel f** shows the Step 4 registration-corrected Chlorophyll+BODIPY
  overlay with the same DIC mask outlines on top, so you can visually check
  the correction is actually centering fluorescence signal inside each mask
  rather than clipping an edge, FOV by FOV, instead of trusting the
  aggregate dy/dx estimate alone.
- Any flagged cell or missed-cell ROI gets its own note field ("why did you
  select this?") below the panels, so the reasoning survives into the
  exported CSVs, not just an unlabeled coordinate.
- **Export ROIs** writes every flagged cell (sample, FOV, cell ID, its full
  `quantify_cells.py` measurement row, and its note) to
  `<output_dir>/flagged_rois.csv`, and every missed-cell ROI (sample, FOV,
  ROI index, its pixel bounding box *and* full polygon converted back to
  original image row/col coordinates, and its note) to
  `<output_dir>/missed_cell_rois.csv`.

This tool calls `quantify_cells.segment_dic`/`accepted_cells`/
`count_lipid_bodies` directly, so what it shows is exactly what ends up in
`cell_measurements.csv` — not an approximation of it.

## Step 4: Fluorescence-to-DIC registration correction

**Problem**: the Chlorophyll and BODIPY channels are shifted relative to DIC
(and hence relative to the DIC-derived cell mask from Step 3) by a small but
consistent amount, biasing any fluorescence sum/average taken from the tight
mask.

**Detection**: cross-correlated a DIC-derived "chloroplast proxy"
(`gaussian_blur(DIC, σ=15) - DIC`, so chloroplasts — locally darker than
their surroundings — become positive peaks) against a high-pass-filtered
Chlorophyll image (chlorophyll autofluorescence marks the same chloroplasts),
via FFT, across all 56 FOVs with both channels present. This gave a
consistent (not random) offset clustering around dy=+12, dx=-3 px (row,
col; ~77% of FOVs within dy 10–13, dx -6–0), confirming a systematic
registration issue rather than per-FOV noise. **Note**: some outliers in
this per-FOV distribution are expected and are not evidence against a
systematic shift — cells can move slightly between the sequential DIC/
Chlorophyll captures, and autofluorescence can appear subtly displaced if
that FOV's fluorescence layer was captured at a slightly different focal
plane.

**Sign convention**: `scipy.ndimage.shift(image, shift=(dy,dx))` with `(dy,dx)`
as measured by cross-correlating `(dic_proxy, chlorophyll)` (in that
argument order) is the *correct* direction to move the chlorophyll/BODIPY
image onto the DIC frame. This was confirmed **empirically**, not just
derived: apply the candidate correction, then re-measure the residual shift
between the corrected image and the DIC proxy — the correct sign/magnitude
drives that residual to exactly (0,0), the wrong one doubles the error. Do
not trust a hand-derived sign convention for FFT cross-correlation without
this kind of check; it's easy to get backwards.

**Refinement — the first estimate overshot**: the whole-FOV cross-correlation
estimate (dy=12, dx=-3) visibly *overshot* for axis-aligned cells — a
cell's chlorophyll blob went from clipping the tight mask's top edge
(uncorrected) to clipping its *bottom* edge (dy=12-corrected), i.e. it
passed straight through the center. Since cells are only ~2–7 µm (13–45 px)
wide, a correction that large in the direction perpendicular to a cell's
long axis can push signal past the opposite edge rather than centering it.
A second, independent estimate was computed as the median of *per-cell*
centroid offsets — intensity-weighted centroid of the DIC chloroplast proxy
within each tight mask, vs. intensity-weighted centroid of Chlorophyll in a
local window around that cell with all *other* cells' pixels explicitly
excluded (to avoid a neighboring cell's much brighter blob skewing the
window) — across all 188 cells. This gave a smaller, outlier-trimmed median
of dy≈7.2, dx≈-1.15 px, which was visually confirmed (on 3 FOVs across all
3 conditions) to center the signal without the dy=12 overshoot.
**Adopted correction: dy=7, dx=-1 px.**

Two correction *strategies* were tried; only the second was adopted:

1. **Rejected**: dilate the tight DIC mask by a fixed radius (15 px, chosen
   so <3% of multi-cell FOVs get any mask-to-mask overlap) instead of moving
   the images, then sum fluorescence in the larger mask
   (`quantify_cells_dilated.py`, kept for reference only). This "covers" the
   shift, but increases mean cell area ~2.8× with mostly background pixels,
   which *mechanically dilutes* real per-condition differences in average
   intensity regardless of whether registration is actually the issue —
   confirmed directly: it flattened an apparent BODIPY condition difference
   that, once retested with proper shift correction, turned out to be real
   and to hold up. **Do not use this approach to draw biological
   conclusions** — it was a useful diagnostic for *how much* dilution
   matters, not a fix.
2. **Adopted**: shift the Chlorophyll and BODIPY images themselves by the
   fixed correction (dy=7, dx=-1 px, via `scipy.ndimage.shift(order=1)`),
   then re-run the *exact same* tight-mask extraction as Step 3.

```
python quantify_cells_shifted.py <input_dir> <output_dir> [--shift-dy 7] [--shift-dx -1] [--days 3] [--reps 1,2]
```

(`quantify_cells_dilated.py` takes the same `--days`/`--reps` flags, plus its own `--dilate-radius`.)

Outputs `cell_measurements_shift_corrected.csv` and
`{chlorophyll,bodipy,lipid_bodies}_per_cell_shift_corrected.png`. Lipid body
counts are essentially unaffected by this correction (the per-cell Otsu
threshold used there is fairly insensitive to a small, correctly-centered
shift) — it mainly matters for the intensity metrics.

**Caveat carried forward**: the same (dy=7, dx=-1) correction, measured from
Chlorophyll, is also applied to BODIPY, since BODIPY has no DIC-visible
structural analog to independently register against. This assumes both
fluorescence channels share the same optical-path offset relative to DIC
(plausible — both are captured through what should be a similar path
immediately after DIC — but **not independently verified**).

**This is the recommended pipeline for fluorescence quantification going
forward** — i.e. Step 3 for segmentation + lipid body counts, Step 4 in
place of Step 3's raw `avg_chlorophyll`/`avg_bodipy` for anything involving
fluorescence intensity.

## Step 5: Ad hoc comparison / subset analyses

These were done as one-off snippets against the CSVs above rather than as
standalone scripts — worth formalizing before this becomes a "real" repo:

- **Total (not average) fluorescence per cell**: same categorical-scatter
  style, using the `total_chlorophyll`/`total_bodipy` columns already present
  in the CSVs (`chlorophyll_total_per_cell.png`, `bodipy_total_per_cell.png`).
  Total scales with cell size and is noisier/less discriminating between
  conditions than the average — prefer `avg_*` unless total signal per cell
  is specifically what you want.
- **Replicate subsetting**: originally done by filtering `cell_measurements*.csv`
  post hoc on `sample.str.extract(r'rep(\d+)')`; now built into every script
  as `--days`/`--reps` (see the "Filtering by Day/replicate" note near the
  top of this file) — prefer re-running with `--reps 1,2` etc. over filtering
  an already-generated CSV, since it also lets `composite_figure.py`
  recompute its normalization baseline over just that subset. Restricting to
  reps 1–2 conveniently also excludes the two known merged-cell artifacts
  from Step 3, since both are in rep 3. Suffix such outputs `_rep<N>-<M>`.
- **Visual mask/registration comparisons** (`quantification/mask_comparison/`):
  tight-vs-dilated mask overlays, and before/after registration-correction
  overlays, generated by small scripts reusing
  `quantify_cells.segment_dic`/`accepted_cells`. Should be consolidated into
  one `make_qc_figures.py` in the real repo.
- **Example lipid-body segmentation overlays**
  (`quantification/lipid_body_examples/`): per-cell crops showing the
  detected BODIPY-bright regions overlaid on DIC/BODIPY, used to visually
  audit the lipid-body counting method against a range of low/high counts.

## Key numeric constants (defined at the top of each script; repeated here for convenience)

| Constant | Value | Where |
|---|---|---|
| Pixel size | 10 µm = 65 px (0.1538 µm/px) | `composite_figure.py`, `quantify_cells.py` |
| Panel gap | 0.5 mm | `composite_figure.py` |
| DIC smoothing σ | 1.0 px | `quantify_cells.py` |
| Gradient threshold multiplier | 0.4 × Otsu | `quantify_cells.py` |
| Morphological closing / opening radius | 10 px / 2 px | `quantify_cells.py` |
| Min component area | 300 px² | `quantify_cells.py` |
| Cell length / width / aspect ratio bounds | 15–40 µm / 2–7 µm / 5–16 | `quantify_cells.py` |
| Min solidity | 0.75 | `quantify_cells.py` |
| Lipid body smoothing σ / min size | 1.0 px / 3 px | `quantify_cells.py` |
| Dilation radius (rejected approach) | 15 px | `quantify_cells_dilated.py` |
| Registration correction (dy, dx) | (7, -1) px | `quantify_cells_shifted.py` |

## Open items

1. Resolve the two known merged-cell rows (`Arginine_Day3_rep3` FOV4 cell 4,
   FOV5 cell 3) — manually exclude, or design/validate a better geometric
   merged-cell detector (a mask-area-vs-expected-single-cell-area check was
   proposed but not implemented).
2. Independently verify the BODIPY registration-shift assumption if
   possible — e.g. a fiducial/bead calibration slide imaged in all 3
   channels would settle this properly, rather than relying on the DIC ↔
   chlorophyll structural correspondence as a proxy.
3. `quantify_cells.py`'s own plotting only produces the lipid-body-count
   plot; the average/total intensity plots currently live in
   `quantify_cells_shifted.py` and ad hoc snippets (Step 5). Consolidate into
   one script with a `--metric` flag.
4. No automated tests exist. At minimum, snapshot-test
   `segment_dic`/`accepted_cells`/`count_lipid_bodies` against a couple of
   the QC-reviewed FOVs so future refactors don't silently change accepted
   cell counts or measurements.
5. `renamed_composites/` currently holds both the renamed per-FOV TIFFs and
   the composite QC PNGs — consider splitting these into separate
   directories for clarity.

## Quickstart: full pipeline from raw data

```bash
# 0. Copy raw captured-layer files into a working directory (never modify <date>_Day<N>/tiffs/ directly)
mkdir renamed_composites
cp 20260625_Day3/tiffs/*"captured layer"*.tiff renamed_composites/

# 1. Rename into FOV-grouped, channel-labeled files (check for divisible-by-3 warnings!)
python rename_microscopy_images.py renamed_composites

# 2. Composite QC figures (optional, for visual review / presentation)
python composite_figure.py renamed_composites renamed_composites

# 3. Segmentation + lipid body counts (add --qc-overlays and inspect a sample)
mkdir quantification
python quantify_cells.py renamed_composites quantification --qc-overlays

# 4. Registration-corrected fluorescence quantification (use this, not Step 3's
#    avg_chlorophyll/avg_bodipy, for any fluorescence-intensity conclusions)
python quantify_cells_shifted.py renamed_composites quantification

# Optional: restrict any step to specific Day(s)/replicate(s), e.g. Day 3 reps 1-2
python quantify_cells_shifted.py renamed_composites quantification --days 3 --reps 1,2
```
