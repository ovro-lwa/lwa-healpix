# lwa-healpix

HEALPix reprojection and coadding tools for OVRO-LWA images.
Reproject snapshot FITS images into HEALPix, coadd overlapping observations,
and publish the results as HiPS tile sets for interactive viewing in Aladin Lite.

## Installation

```bash
pip install .
```

## Functions

### `coadd_fits_to_healpix`

Reproject a list of FITS images to HEALPix and coadd them with
footprint-based weighting. Supports an optional minimum-elevation mask to
exclude noisy data near the horizon. Returns the combined map and total
weight arrays.

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

### `fits_to_hips_cube`

End-to-end pipeline from per-frequency FITS images to a HiPS3D tile set.
Wraps `combine_fits_to_spectral_cube` and `reproject_to_hips` with useful
defaults for tile size, spectral tile depth, and coordinate frame.

## Future directions

- **All-sky maps**: Generate all-sky maps for OVRO-LWA by coadding deep
  images spanning a range of LST.
- **Multi-frequency all-sky maps**: Use HiPS3D to create spectral-cube
  versions of the all-sky map, combining data across many frequencies.
- **HEALPix-domain coadding**: Explore direct coadding of data in HEALPix
  projections rather than intermediate flat-sky grids.
