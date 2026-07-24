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
segmentation_params_explorer_app.py    (interactive Step 3 segmentation tuning — see "Interactive parameter-exploration tools" below)
blob_counting_params_explorer_app.py   (interactive lipid-body/plastid counting tuning — see "Interactive parameter-exploration tools" below)
cell_params_review_app.py    (per-cell manual parameter estimation + notes — see "Per-cell parameter review tool" below)
plot_by_replicate.py         (replicate-colored lipid-body/BODIPY/chlorophyll plots — see Step 5)
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

0. **Fixed-pattern background correction** (`compute_dic_background` /
   `correct_dic_background`), applied before anything else: a per-pixel
   *median* across every DIC image in `input_dir` (always the full
   directory, regardless of any `--days`/`--reps` filter — this estimates a
   stationary instrument artifact, not biology). Real cells occupy different
   pixels from FOV to FOV and get suppressed by the median, while a
   stationary optical artifact (an out-of-focus dust speck, in this dataset)
   survives at the same pixel location in every FOV and gets isolated.
   Confirmed **DIC-specific** — not present in Chlorophyll/BODIPY at the same
   pixel location — so the correction is applied only to the DIC image fed
   into segmentation, never to the fluorescence channels used for
   quantification. Left uncorrected, the artifact's edge ring could get
   pulled into a real cell's mask during the closing step below if the cell
   happened to sit near it, distorting that cell's fitted shape. This fix
   alone changed the accepted-cell count from 188 to 194 on this dataset.
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
     non-convex/branching shapes — **or**, if solidity fails, a second chance
     via `has_body_branch`: skeletonize the component's mask, find its two
     tips by geodesic (along-skeleton) distance (robust to curvature, and to
     both sharp- and blunt-tip skeletonization artifacts), and check whether
     any skeleton branch point sits far (> `SKELETON_TIP_MARGIN_PX` = 20px)
     from both tips. A bent-but-otherwise-clean fusiform cell's skeleton has
     no such branch — straight-line solidity penalizes it purely for
     curving, not for anything actually wrong — while a genuinely malformed
     component (two cells merged, debris fused on) does have one. This can
     only *add* acceptances, never remove one: anything that already passes
     the plain solidity cutoff is accepted regardless of its skeleton. This
     rescue changed the accepted-cell count from 194 to 209 on this dataset
     (i.e. it, not the background fix above, accounts for most of the total
     188→209 change since the two fixes were calibrated).
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
(total ÷ `area_px`, i.e. mean intensity per pixel), `n_lipid_bodies`,
`n_plastids`, `chlorophyll_focus_score`, `in_focus`.

**Lipid body / plastid counting (watershed)**: both use the same function,
`count_bright_blobs` — Gaussian-smooth the channel (σ=1.0) → per-cell Otsu
threshold computed only on that cell's own pixels → **watershed-split** the
resulting bright mask at local intensity peaks
(`skimage.feature.peak_local_max` + `skimage.segmentation.watershed`) →
count connected components ≥3 px in the *split* result. `count_lipid_bodies`
(BODIPY) and `count_plastids` (Chlorophyll) are thin wrappers over it with
their own constants.

This supersedes plain connected-component counting, which undercounts
touching blobs whenever the valley between their brightness peaks never dips
below the per-cell threshold — confirmed on this dataset: a single 1685px
"blob" in one cell was visibly 2–3 distinct ring-shaped droplets in the raw
BODIPY. Watershed seeds one marker per local intensity peak and floods
outward from each, splitting at the ridge between them; a single-peak blob
watersheds right back to itself, so this can only help merged cases, never
change an already-distinct count. (An earlier local-maxima peak-splitting
attempt, tried before proper watershed flooding, was rejected: a small
peak-detection footprint alone produced spurious extra peaks from pixel
noise inside one droplet, and a larger footprint just recreated the
undercounting it was meant to fix — the missing piece was watershed's
ridge-flooding step, not peak detection by itself.)

Two independent watershed knobs, both exposed per-channel:
- `watershed_min_distance` (px) — purely spatial non-max-suppression radius;
  two peaks closer than this always collapse to one. `LIPID_WATERSHED_MIN_DISTANCE_PX = 3`
  is a confirmed-reasonable default for BODIPY, as is `PLASTID_WATERSHED_MIN_DISTANCE_PX = 3`
  for Chlorophyll.
- `watershed_min_prominence` (intensity units) — a complementary, purely
  topographic criterion via `skimage.morphology.h_maxima`: a peak only
  survives if there's no path to an equal-or-higher peak along which the
  intensity drop is smaller than this value. This is what distinguishes two
  real neighboring blobs (each has its own full-height peak with a
  genuinely deep valley between them) from one blob with two lobes at
  slightly different focus/brightness (the dip between them never drops
  back to background). Calibrated: `LIPID_WATERSHED_MIN_PROMINENCE = 60`,
  `PLASTID_WATERSHED_MIN_PROMINENCE = 100`.

**Plastid counting now uses the skeleton-clustered method** (promoted from
prototype, see below), calibrated against two known validation cells:
`Arginine_Day3_rep2FOV2` cell 2 (two real close-together plastids) and
`Arginine_Day3_rep3FOV4` cell 4 (a dividing cell near the end of cytokinesis
whose two plastids should be symmetric about the cell's long axis). At the
original defaults (plain watershed, `min_distance=3`, `min_prominence=0`,
mirroring BODIPY), >1 plastid was detected in ~91% of cells (190 of 209) —
not biologically plausible for *P. tricornutum*, which normally carries one
plastid that duplicates only near the end of division. A systematic sweep of
`sigma`/`min_distance` and of `threshold_mult`+`min_prominence` together
showed this wasn't a matter of finding one missed value: no single
global-parameter combination with *plain* watershed satisfied both
validation cells at once (the first needs `min_distance` around 8; the
second breaks under every value that fixes the first). The skeleton-clustered
method (below) resolves both simultaneously.

**`watershed_min_prominence` and skeleton-clustering interact — keep
prominence low enough that both peaks survive to be clustered.** `h_maxima`
prominence filtering runs *before* peak clustering, so if it's set too high
it can prune a real peak down to nothing before clustering ever sees it,
and clustering cannot recover a peak that no longer exists. Confirmed on
`Arginine_Day3_rep3FOV4` cell 4: at `min_prominence` 0–100 both of its real
peaks survive and correctly cluster into `n_plastids=2`; at 200+, `h_maxima`
prunes one of the two peaks first and the cell reads `n_plastids=1`
regardless of clustering. `PLASTID_WATERSHED_MIN_PROMINENCE = 100` was
chosen with this margin in mind — don't raise it without re-checking both
validation cells.

**Chlorophyll focus score** (`compute_focus_score`, `chlorophyll_focus_score`
column): variance of the Laplacian of the *raw* (unsmoothed) Chlorophyll
channel within a cell's mask — a standard microscopy autofocus metric
(in-focus structure has high-frequency detail that blur suppresses).
Motivation: cells with 0 detected plastids turned out, on inspection, to be
genuinely out-of-focus cells rather than a counting bug (e.g.
`Arginine_Day3_rep1FOV3` cell 3) — sharpness of the plastid signal is itself
a useful QC signal, not just noise to work around. Cross-checked against
every previously manually-flagged cell in this dataset: all 3 cells flagged
"out of focus" scored in the 5th–7th percentile dataset-wide, while cells
flagged for unrelated reasons (incomplete/ragged segmentation) scored
49th–99th — a clean separation. `in_focus` is just
`chlorophyll_focus_score >= PLASTID_MIN_FOCUS_SCORE` (580, an informed
~10th-percentile starting cutoff, not exhaustively validated at the
boundary); nothing is dropped from the output automatically — filtering on
`in_focus` is an explicit downstream choice.

**Registration order matters for watershed seeding, not just intensity
sums.** `count_bright_blobs`' peak search is restricted to the DIC-derived
cell mask, which assumes the fluorescence channel is already registered to
DIC (see Step 4). `quantify_cells.py`'s own `main()` does **not** apply that
correction before counting blobs — it computes `n_lipid_bodies`/`n_plastids`
from the raw, unregistered channels, so a real peak sitting just past the
mask edge (most likely near a cell's tapered tip, where the mask is
narrowest) can get clipped and reported at the wrong location. This was
caught directly on this dataset: in `Arginine_Day3_rep2FOV2` cell 2, a
plastid's true peak sat 11px outside the mask, at exactly the point where
the mask tapered; applying the registration shift moved it into a part of
the mask over 100px wide, resolving the clip. `quantify_cells_shifted.py`
(Step 4) applies the registration correction *before* calling
`count_lipid_bodies`/`count_plastids`, so its blob counts are the
order-correct ones — prefer those over `quantify_cells.py`'s own
uncorrected `n_lipid_bodies`/`n_plastids` for any cell near a mask edge.

`--qc-overlays` writes one PNG per FOV to `<output_dir>/qc_overlays/`
showing which regions were accepted as cells. **Inspect a sample of these
before trusting results on any new data** — this is how both the length-range
mismatch and the merged-cell issue above were originally caught.

Default plots: `lipid_bodies_per_cell.png` and `plastids_per_cell.png`
(categorical scatter, one point per cell, x = condition in order
`[Nitrate, Arginine, Urea]`, black bar = mean), plus `dic_background.png`
(the fixed-pattern background map itself). Average/total-intensity plots are
*not* produced by this script's `main()` by default — see Step 5.

**Hand-flagged cell exclusion**: if `<output_dir>/flagged_rois.csv` exists
(produced by `qc_review_app.py`, see below), both this script and
`quantify_cells_shifted.py` automatically drop every `(sample, fov, cell_id)`
it lists — poor segmentation, out of focus, cell doublets, etc, per each
row's own note — from `cell_measurements.csv`/`cell_measurements_shift_corrected.csv`
and all derived plots, before anything is saved (`load_flagged_cell_keys`/
`exclude_flagged_cells` in `quantify_cells.py`). Excluded cell_ids are never
renumbered — a gap just means that cell was hand-excluded, keeping the
mapping back to `flagged_rois.csv`/the QC overlays traceable. This is
automatic and silent about *why* a given cell was flagged (see the note
column in `flagged_rois.csv` itself for that) but prints how many cells it
excluded. If the file doesn't exist, nothing is excluded — this only
activates once a reviewer has actually produced one.

### Interactive parameter-exploration tools

Two Bokeh apps let you tune segmentation/counting parameters live against
real FOVs and individual cells, instead of editing constants and re-running
the full pipeline:

```
bokeh serve --show segmentation_params_explorer_app.py --args <input_dir>
bokeh serve --show blob_counting_params_explorer_app.py --args <input_dir>
```

Also registered in `~/ClaudeCowork/.claude/launch.json` as
`bokeh-segmentation-params` (port 5008) and `bokeh-blob-counting-params`
(port 5010).

- **`segmentation_params_explorer_app.py`**: sliders for the Step 3
  DIC-segmentation constants (smoothing σ, threshold multiplier, morphology
  radii, length/width/aspect/solidity bounds), a DIC-background-correction
  on/off toggle, and status text that reports when a cell was accepted only
  via the curvature-tolerant solidity rescue (`has_body_branch`).
- **`blob_counting_params_explorer_app.py`**: BODIPY/Chlorophyll blob
  counting, calling `quantify_cells.count_bright_blobs` directly (so it
  can't drift from the production math). Channel toggle (each with its own
  remembered defaults), registration-correction toggle (on by default,
  matching `quantify_cells_shifted.py`), a connectivity toggle, and a 3-way
  method toggle — plain per-cell threshold, production watershed, or a
  skeleton-clustered prototype (below) — with sliders for smoothing σ,
  threshold multiplier, `watershed_min_distance`, `watershed_min_prominence`,
  and (skeleton method only) cluster gap. Cell crops auto-rotate (cosmetic
  only, `np.rot90`) so each cell's longer side always displays vertically.

**Skeleton-clustered plastid splitting — promoted to production.** Neither
`min_distance` nor `min_prominence` alone could satisfy both known
plastid-counting validation cases above. This method (originally prototyped
in `blob_counting_params_explorer_app.py`, method toggle "Watershed,
skeleton-clustered") reuses the cell's own DIC-derived skeleton (identical
construction to `has_body_branch`'s) to project every raw intensity peak
onto a curvature-tolerant coordinate — arc-length along the cell's
centerline, and which side of it — then clusters peaks by (side, arc-length
gap ≤ `cluster_gap_px`) into one watershed marker per cluster instead of one
per raw peak. This succeeds on both known validation cells where every
global-parameter combination with plain watershed failed, because unlike a
straight-line long-axis split it tolerates cell curvature.

Promoted into `quantify_cells.py`'s own `count_bright_blobs`/`count_plastids`
(`method="skeleton"` — `_project_peaks_onto_skeleton`/
`_cluster_peaks_by_skeleton`, ported verbatim from the explorer-app
prototype) as of this calibration. `count_lipid_bodies`/BODIPY still uses
plain watershed (`method="watershed"`, the default) — only Chlorophyll/plastid
counting uses skeleton-clustering. `PLASTID_SKELETON_CLUSTER_GAP_PX = 20` is
inherited from the prototype's own default and has not been independently
re-tuned. The originally-flagged open risk — whether a single non-dividing
plastid's own internal texture noise could land peaks on both skeleton sides
and cause a false split — has not been exhaustively checked against a large
sample of known single-plastid cells, only the two validation cells above;
watch `n_plastids` for implausible results as more data comes in.

### QC review tool (`qc_review_app.py`)

`--qc-overlays` PNGs are useful but static and one-per-FOV; `qc_review_app.py`
is an interactive Bokeh server app for browsing the whole dataset's
segmentation FOV-by-FOV and flagging specific cells as poorly segmented, to
build up a concrete list of failure cases (e.g. for choosing/validating
automated-test fixtures, see the open items below). Its `flagged_rois.csv`
export feeds directly back into `quantify_cells.py`/`quantify_cells_shifted.py`
(see "Hand-flagged cell exclusion" in Step 3) — flagging a cell here excludes
it from the actual analysis, not just from this review UI.

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

Panels a-d share one pair of `Range1d` instances, and panels e/f share a
separate pair, so panning or zooming any one of a-d applies to all four
(and likewise for e/f) -- panels aren't independently zoomable within
their own group, by design, since you're always comparing the same FOV
across channels.

- **Panel e** is DIC with every accepted cell's mask boundary (from
  `skimage.measure.find_contours` on the exact same tight mask
  `quantify_cells.py` measures) drawn as a clickable region: click a cell to
  flag it red (poor segmentation), click again to unflag. Flags persist as
  you navigate between FOVs via the Previous/Next buttons or the FOV
  dropdown. The same panel's **Freehand Draw** tool (toolbar icon) lets you
  lasso a good cell the pipeline missed entirely. (An earlier version used
  the Box Edit tool for rectangular ROIs; it didn't reliably respond to
  drag gestures in testing, so it was replaced with Freehand Draw, which
  also traces the actual cell outline instead of a crude bounding box.)
  Drew one poorly? Don't rely on the tool's own tap-to-select-then-Backspace
  gesture -- that wasn't reliable either. Instead use the **Remove ROI**
  button next to that ROI's note field, below the panels.
- **Panel f** shows the Step 4 registration-corrected Chlorophyll+BODIPY
  overlay with the same DIC mask outlines on top, so you can visually check
  the correction is actually centering fluorescence signal inside each mask
  rather than clipping an edge, FOV by FOV, instead of trusting the
  aggregate dy/dx estimate alone.
- Any flagged cell or missed-cell ROI gets its own note field ("why did you
  select this?") below the panels, so the reasoning survives into the
  exported CSVs, not just an unlabeled coordinate.
- **Export ROIs** writes every flagged cell (sample, FOV, cell ID, its full
  `quantify_cells.py` measurement row, and its note) and every missed-cell
  ROI (sample, FOV, ROI index, its pixel bounding box *and* full polygon
  converted back to original image row/col coordinates, and its note) to
  two CSVs whose filenames are set directly in the app (two text boxes at
  the top, defaulting to `flagged_rois.csv`/`missed_cell_rois.csv`, joined
  with `<output_dir>`). Both filenames are checked for an existing file at
  startup and whenever changed; if found, its contents are merged into the
  dashboard's in-memory state (not wiping out whatever's already flagged in
  the current session) -- so closing and reopening the app, or pointing it
  at a colleague's export, picks up right where that file left off.

This tool calls `quantify_cells.segment_dic`/`accepted_cells`/
`count_lipid_bodies`/`compute_dic_background`/`correct_dic_background`
directly, so what it shows is exactly what ends up in `cell_measurements.csv`
— not an approximation of it. (This app used to call `segment_dic` on the
raw, uncorrected DIC image, so its cell population could silently disagree
with `quantify_cells.py`'s own `main()` wherever the fixed-pattern background
correction mattered — fixed to apply the same correction first, same as
`quantify_cells_shifted.py`.)

### Per-cell parameter review tool (`cell_params_review_app.py`)

A third interactive tool, merging `qc_review_app.py`'s per-cell note/CSV-export
workflow with the two parameter-explorer apps above, for walking the whole
dataset cell by cell and manually estimating segmentation/counting/focus
parameters per cell — e.g. to compare a human's per-cell parameter estimate
against an automated optimization, or to check how consistent manual
estimates are across FOVs/cells.

```
bokeh serve --show cell_params_review_app.py --args <input_dir> [<output_dir>]
```

(defaults to `renamed_composites`/`quantification` if omitted; also registered
in `~/ClaudeCowork/.claude/launch.json` as `bokeh-cell-params-review`, port
5011.)

Three tabs per cell, all built from `quantify_cells.py`'s own functions (never
reimplemented):

1. **DIC segmentation** — the same 3-panel view as
   `segmentation_params_explorer_app.py` (raw DIC / binary mask / accept-reject
   with hover), at full-FOV scale, with every slider that app exposes. The
   current cell's outline is drawn in blue on the accept/reject panel for
   orientation. DIC segmentation is a property of the whole FOV, not one cell,
   so — unlike the other two tabs — its parameters are saved **per FOV**, not
   per cell (see "Parameter keying" below).
2. **Chlorophyll (plastids)** — the same 5-panel view as
   `blob_counting_params_explorer_app.py` (DIC+outline / raw / smoothed /
   bright mask / counted-or-too-small), fixed to the Chlorophyll channel, with
   every slider that app exposes (including the skeleton-clustered prototype
   method), plus a focus-score testing section: a pre-Laplacian
   smoothing-sigma slider (0 = matches production's raw/unsmoothed
   `compute_focus_score`) and a focus-score threshold slider (defaults to
   `PLASTID_MIN_FOCUS_SCORE`), showing the live focus score and `in_focus`
   flag for the current cell. **This tool's own starting defaults for this tab
   are `method="skeleton"` (the skeleton-clustered prototype) and
   `watershed_min_prominence=600`** — set at the user's explicit request for
   manual-estimation sessions in this tool specifically; they intentionally
   differ from `quantify_cells.py`'s own production constant
   (`PLASTID_WATERSHED_MIN_PROMINENCE=0`, disabled) and from
   `blob_counting_params_explorer_app.py`'s own defaults, neither of which
   were changed.
3. **BODIPY (lipid bodies)** — identical 5-panel view fixed to the BODIPY
   channel; no focus section, since `compute_focus_score` is not used for
   BODIPY in production. This tab's own starting default keeps
   `method="watershed"` (unchanged) but sets `watershed_min_prominence=100`,
   again a starting point for this tool only.

As in `blob_counting_params_explorer_app.py`, DIC segmentation itself (which
cells exist and how many) is a **fixed** upstream input for navigation and for
tabs 2/3's cell list — tab 1's sliders are an independent exploration surface
layered on top and never change the cell list, so it can't shift under you
while exploring.

**Parameter keying**: Chlorophyll/BODIPY parameters and the note are saved
**per cell** (production already computes a per-cell Otsu threshold for blob
counting). DIC parameters are saved **per FOV** (DIC segmentation runs once
per FOV, so it wouldn't make sense for the same segmentation to have different
recorded parameters depending on which of its cells you happened to be
viewing when you tuned it). The exported CSV still has one row per cell,
matching `qc_review_app.py`'s convention — every cell sharing a FOV gets that
FOV's current DIC parameters copied into its row at export time.

**Save behavior**: navigating to another cell (Prev/Next/dropdowns) commits
the current sliders into an in-memory registry; nothing is written to disk
until **Export CSV** is clicked. Revisiting an already-committed FOV/cell
restores its last-committed values into the sliders instead of resetting to
calibrated defaults, so a prior estimate can be reviewed or revised. The CSV
filename (default `cell_parameter_review.csv`, joined with `<output_dir>`) is
checked for an existing file both at startup and whenever changed, and its
contents are merged into the in-memory registry — mirroring
`qc_review_app.py`'s flagged-cell CSVs — so closing and reopening the app
picks up where a previous session left off.

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

Outputs `cell_measurements_shift_corrected.csv`,
`{chlorophyll,bodipy,lipid_bodies,plastids}_per_cell_shift_corrected.png`,
and `dic_background.png`. Lipid body counts are essentially unaffected by
this correction (the per-cell Otsu threshold used there is fairly
insensitive to a small, correctly-centered shift) — it mainly matters for
the intensity metrics **and** for plastid/lipid-body watershed seeding (see
below).

**Caveat carried forward**: the same (dy=7, dx=-1) correction, measured from
Chlorophyll, is also applied to BODIPY, since BODIPY has no DIC-visible
structural analog to independently register against. This assumes both
fluorescence channels share the same optical-path offset relative to DIC
(plausible — both are captured through what should be a similar path
immediately after DIC — but **not independently verified**).

**Beyond intensity sums — this correction also matters for watershed peak
seeding.** `count_lipid_bodies`/`count_plastids` (Step 3) restrict their
watershed peak search to the DIC-derived cell mask, which assumes the
fluorescence channel is already in registration with DIC. `quantify_cells.py`
computes `n_lipid_bodies`/`n_plastids` from the *raw, uncorrected* channels;
`quantify_cells_shifted.py` computes them from the shift-corrected channels,
so its blob counts (as well as its intensity sums) are the order-correct
ones — see the per-cell measurements section of Step 3 for the specific bug
this fixes.

**`quantify_cells_shifted.py`'s DIC segmentation is now identical to
`quantify_cells.py`'s** — both apply `compute_dic_background`/
`correct_dic_background` before `segment_dic`, and `accepted_cells`' shared
curvature-tolerant solidity rescue applies to both for free. Previously
`quantify_cells_shifted.py` segmented the *raw* DIC image, so its cell
population (204 cells) and `cell_id` numbering diverged from
`quantify_cells.py`'s (209 cells at the time); they now match exactly,
modulo whatever `--days`/`--reps` filter each run uses.

**This is the recommended pipeline for fluorescence quantification going
forward** — i.e. Step 3 for segmentation, Step 4 in place of Step 3's raw
`avg_chlorophyll`/`avg_bodipy`/`n_lipid_bodies`/`n_plastids` for anything
involving fluorescence intensity or blob counts.

## Step 5: Ad hoc comparison / subset analyses

These were done as one-off snippets against the CSVs above rather than as
standalone scripts — worth formalizing before this becomes a "real" repo:

- **Replicate-colored plots** (`plot_by_replicate.py`, a real standalone
  script, not an ad hoc snippet): `n_lipid_bodies`, `avg_bodipy`, and
  `avg_chlorophyll` per cell, split by condition (same categorical-scatter
  style as `make_categorical_plot`) with each point additionally colored by
  replicate number (`make_categorical_plot_by_replicate`, `quantify_cells.py`).
  Reads `cell_measurements_shift_corrected.csv` — the recommended pipeline
  for both fluorescence intensity and blob counts (see Step 4) — so
  hand-flagged cells are already excluded. Run:
  `python plot_by_replicate.py [<csv_path>] [<output_dir>]` (defaults to
  `quantification/cell_measurements_shift_corrected.csv` and `quantification/`).
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
- **`quantification_reps1-2/`** (2026-07-24): rep 3 was identified as an
  outlier relative to reps 1–2 across all three nitrogen conditions (visible
  in the replicate-colored plots above), so this directory holds a
  self-contained reps-1,2-only analysis — `composite_figure.py`,
  `quantify_cells.py`, and `quantify_cells_shifted.py` all re-run with
  `--reps 1,2` (own normalization baseline, own DIC segmentation/measurements),
  plus `plot_by_replicate.py`'s three plots recomputed from that subset. The
  full-dataset (reps 1–3) results in `quantification/` are untouched.
  `flagged_rois.csv` was copied (not moved) into this directory first, since
  `exclude_flagged_cells` looks for it in `output_dir` and the same
  hand-reviewed exclusions apply regardless of which reps subset is analyzed.
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
| Min solidity | 0.75 (rescuable via `has_body_branch` — see Step 3) | `quantify_cells.py` |
| Skeleton prune iterations / tip margin | 10 px / 20 px | `quantify_cells.py` |
| Lipid body smoothing σ / min size | 1.0 px / 3 px | `quantify_cells.py` |
| Lipid body watershed min distance / min prominence | 3 px / 60 | `quantify_cells.py` |
| Plastid smoothing σ / min size | 1.0 px / 3 px | `quantify_cells.py` |
| Plastid watershed min distance / min prominence | 3 px / 100 (see prominence/clustering interaction note, Step 3) | `quantify_cells.py` |
| Plastid counting method | skeleton-clustered (`count_plastids`, promoted from prototype) | `quantify_cells.py` |
| Plastid min focus score | 580 | `quantify_cells.py` |
| Dilation radius (rejected approach) | 15 px | `quantify_cells_dilated.py` |
| Registration correction (dy, dx) | (7, -1) px | `quantify_cells.py` (also used by `quantify_cells_shifted.py`) |
| Skeleton-clustering cluster gap | 20 px (inherited from prototype default, **not independently re-tuned**) | `quantify_cells.py` |

## Open items

1. Resolve the two known merged-cell rows (`Arginine_Day3_rep3` FOV4 cell 4,
   FOV5 cell 3) — manually exclude, or design/validate a better geometric
   merged-cell detector (a mask-area-vs-expected-single-cell-area check was
   proposed but not implemented).
2. Independently verify the BODIPY registration-shift assumption if
   possible — e.g. a fiducial/bead calibration slide imaged in all 3
   channels would settle this properly, rather than relying on the DIC ↔
   chlorophyll structural correspondence as a proxy.
3. `quantify_cells.py`'s own plotting only produces the lipid-body-count and
   plastid-count plots; the average/total intensity plots currently live in
   `quantify_cells_shifted.py` and ad hoc snippets (Step 5). Consolidate into
   one script with a `--metric` flag.
4. No automated tests exist. At minimum, snapshot-test
   `segment_dic`/`accepted_cells`/`count_lipid_bodies`/`count_plastids`
   against a couple of the QC-reviewed FOVs so future refactors don't
   silently change accepted cell counts or measurements.
5. `renamed_composites/` currently holds both the renamed per-FOV TIFFs and
   the composite QC PNGs — consider splitting these into separate
   directories for clarity.
6. ~~Calibrate the Chlorophyll watershed parameters~~ — **done** as of
   2026-07-23: `PLASTID_WATERSHED_MIN_PROMINENCE=100` with the
   skeleton-clustered method satisfies both known validation cells (see
   Step 3). Still open: this hasn't been checked against a large sample of
   known single-plastid (non-dividing) cells for false splits, only the two
   validation cells — watch `n_plastids` for implausible dataset-wide rates.
7. ~~Decide whether to promote the skeleton-clustered plastid-splitting
   prototype to production~~ — **done** as of 2026-07-23, see Step 3.
   `cluster_gap_px=20` is still just inherited from the prototype default,
   not independently re-tuned.
8. `quantify_cells.py`'s own `main()` computes `n_lipid_bodies`/`n_plastids`
   from raw, unregistered fluorescence channels (see Step 3/4) — consider
   whether it should apply `correct_fluorescence_registration` itself, or
   whether `quantify_cells_shifted.py` should simply become the one script
   that computes blob counts at all (avoiding two slightly different
   `n_plastids`/`n_lipid_bodies` values per cell across the two CSVs).

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
