# lwa-healpix

HEALPix reprojection and coadding tools for OVRO-LWA images.
Reproject snapshot FITS images into HEALPix, coadd overlapping observations,
and publish the results as HiPS tile sets for interactive viewing in Aladin Lite.

## Installation

```bash
pip install .
```

## Functions

### `coadd_fits`

Reproject a list of FITS images onto either a HEALPix grid (`nside`) or a
2-D target image (`target_header`) and coadd them with footprint-based
weighting. Supports an optional minimum-elevation mask to exclude noisy
data near the horizon. Returns the combined map and total weight arrays.

**Quality screening (optional):** If you set `quality_max_rms` and/or
`quality_outlier_sigma`, each file is checked *before* reprojection using
only a **central patch** of the spatial plane (memory-mapped slice; default
25% of each axis, capped at 512 pixels per axis). The metric is either the
standard deviation (`quality_metric="std"`) or a robust scale
`1.4826 × MAD` (`quality_metric="mad_sigma"`). Use `quality_max_rms` for an
absolute ceiling in the same units as the image data (`BUNIT`). Use
`quality_outlier_sigma` to drop images whose metric exceeds
`median + σ × 1.4826 × MAD` over the batch. If `min_elevation` is set, the
same horizon blanking is applied to that patch before the metric. If every
file fails screening, `coadd_fits` raises `ValueError`.

### `center_patch_rms_from_fits`

Lower-level helper: compute the same center-patch dispersion statistic for a
single FITS path (useful for inspection or custom pipelines).

### `healpix_to_hips`

Convert a 1-D HEALPix map into a HiPS tile set via an intermediate
Plate Carree (CAR) grid. The CAR grid pixel scale is automatically matched
to the HEALPix NSIDE so that resolution is preserved. An `index.html`
viewer (Aladin Lite) is copied into the output directory.

### `combine_fits_to_spectral_cube`

Combine single-frequency FITS images into a 3-D spectral cube. Each input
file is expected to have 4 axes (RA, Dec, Freq, Stokes) with length-1
frequency and Stokes dimensions. The Stokes axis is dropped and the
frequency planes are stacked, sorted by frequency, into a cube suitable
for HiPS3D generation.

When several files share a frequency, they are coadded with `coadd_fits`;
the same quality arguments as above (`quality_max_rms`, `quality_outlier_sigma`,
`quality_metric`, `quality_center_fraction`, `quality_center_max_pixels`)
are forwarded for those coadds. **Note:** channels that have only a single
input file are stacked without running quality screening (there is nothing
to coadd).

### `fits_to_hips_cube`

End-to-end pipeline from per-frequency FITS images to a HiPS3D tile set.
Wraps `combine_fits_to_spectral_cube` and `reproject_to_hips` with useful
defaults for tile size, spectral tile depth, and coordinate frame. Accepts
the same optional quality-screening keyword arguments as
`combine_fits_to_spectral_cube` and passes them through.

## Future directions

- **All-sky maps**: Generate all-sky maps for OVRO-LWA by coadding deep
  images spanning a range of LST.
- **Multi-frequency all-sky maps**: Use HiPS3D to create spectral-cube
  versions of the all-sky map, combining data across many frequencies.
- **HEALPix-domain coadding**: Explore direct coadding of data in HEALPix
  projections rather than intermediate flat-sky grids.
