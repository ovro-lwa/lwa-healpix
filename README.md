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
data near the horizon:

- **Circular:** `min_elevation=10` blanks pixels with elevation below 10°.
- **Elliptical:** `min_elevation_ns=15, min_elevation_ew=40` keeps an
  ellipse in the zenith-angle plane (stricter toward east/west). Elevation
  is `90° − separation(pixel, CRVAL)` with CRVAL treated as local zenith;
  azimuth is the position angle from zenith (N→E).

Returns the combined map and total weight arrays.

**Quality screening (optional):** If you set `quality_max_rms` and/or
`quality_outlier_sigma`, each file is checked *before* reprojection using
only a **central patch** of the spatial plane (memory-mapped slice; default
25% of each axis, capped at 512 pixels per axis). The metric is either the
standard deviation (`quality_metric="std"`) or a robust scale
`1.4826 × MAD` (`quality_metric="mad_sigma"`). Use `quality_max_rms` for an
absolute ceiling in the same units as the image data (`BUNIT`). Use
`quality_outlier_sigma` to drop images whose metric exceeds
`median + σ × 1.4826 × MAD` over the batch. If elevation blanking is set
(circular or elliptical), the same horizon mask is applied to that patch
before the metric. If every file fails screening, `coadd_fits` raises
`ValueError`.

### `parallactic_delta_q_edge_summary`

Diagnostic table: for eight cardinal directions on the elliptical elevation
mask edge, compute the range of 1-hour parallactic-angle change (`Δq`) over
24 LST bins. Useful as a projection/stacking-error check when coadding
hours with elongated PSFs.

### `center_patch_rms_from_fits`

Lower-level helper: compute the same center-patch dispersion statistic for a
single FITS path (useful for inspection or custom pipelines).

### `reproject_healpix_to_wcs`

Reproject a 1-D HEALPix map onto an arbitrary 2-D celestial WCS
(`target_header`). Returns `(data, footprint)`.

### `nested_tile_header` / `iter_nested_tile_headers`

Build local TAN (or SIN) WCS headers centered on nested HEALPix pixels at
`nside_tile`, with pixel scale from `nside_map` and optional FOV `overlap`
(default `0.2`). Typical detect defaults: `nside_map=2048`, `nside_tile=4`.

### `healpix_to_hdu`

Reproject a HEALPix map (and optional weight) onto a target header, blank
low-weight / zero-footprint pixels to NaN, and return a `PrimaryHDU`. Does
**not** set beam keywords (`BMAJ`/`BMIN`).

### `write_healpix_fits` / `read_healpix_fits`

Persist a HEALPix imaging product as a multi-extension FITS file:

- HDU 0 (Primary): `NSIDE`, `ORDERING`, `COORDSYS`, `PIXTYPE=HEALPIX`
- HDU 1 (`MAP`): 1-D float32 map
- HDU 2 (`WEIGHT`): 1-D float32 weight

Use `read_healpix_fits` to reload. This is an **image** coadd product, not a
catalog HiPS map.

### `healpix_to_hips`

Convert a 1-D HEALPix map into a HiPS tile set via an intermediate
Plate Carree (CAR) grid. The CAR grid pixel scale is automatically matched
to the HEALPix NSIDE so that resolution is preserved. An `index.html`
viewer (Aladin Lite) is copied into the output directory.

### Percentile-based viewer cuts

`healpix_to_hips`, `fits_to_hips`, and `fits_to_hips_cube` calculate the
default display cuts from the 1st and 99th percentiles of finite, unmasked
input pixels. The resulting absolute values are written to
`hips_pixel_cut` in the HiPS `properties` file. The bundled viewer loads
these defaults into its min/max controls; users can still edit either value.

Choose different percentiles with `cut_percentiles=(low, high)`, disable
percentile calculation with `cut_percentiles=None`, or provide an explicit
`properties={"hips_pixel_cut": "min max"}` override.

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
Wraps `combine_fits_to_spectral_cube` and `reproject_to_hips` with defaults
aligned to the [CDS HiPS3D specification](https://aladin.cds.unistra.fr/java/DocTechHiPS3Den.pdf)
(`tile_size=256`, `tile_depth=16`). After tile generation the function also:

- sets `dataproduct_type = spectral-cube`, `obs_restfreq`, tight `em_min`/
  `em_max` (metres), and `obs_regime = Radio`
- leaves `hips_version = 1.4` and `hips_builder = astropy/reproject via
  lwa-healpix` (honest provenance; not Hipsgen output)
- writes `obs_description` explaining FMOC indexing vs generator
- writes `hips_initial_freq` from the centre channel of the input cube
- writes `Moc.fits` with cube∩tile FMOC frequency coverage (`mocpy`)
- copies a HiPS3D Aladin Lite v3.8.1 `index.html` viewer (`newImageSurvey`)

To patch an existing HiPS3D directory without regenerating tiles, use
`upgrade_hips3d(output_directory, freq_min_hz=..., freq_max_hz=...)`.
By default only missing metadata is filled in; pass `overwrite=True` to
replace `properties` keys, `Moc.fits`, and `index.html`.

Accepts the same optional quality-screening keyword arguments as
`combine_fits_to_spectral_cube` and passes them through.

**Viewing in Aladin Lite:** browsers cannot load local tile files directly.
Serve the output directory over HTTP and open `index.html`:

```bash
python -m http.server 8000 --directory output/hips_cube
# then browse to http://localhost:8000/
```

Use Aladin Lite **v3.8+** for HiPS3D spectrogram navigation. Reference HiPS3D
examples are listed in the
[CDS HiPS3D tutorial](https://aladin.cds.unistra.fr/java/TutoHiPS3Den.pdf).

## Future directions

- **Multi-frequency all-sky maps**: Use HiPS3D to create spectral-cube
  versions of the all-sky map, combining data across many frequencies.
- **HEALPix-domain coadding**: Explore direct coadding of data in HEALPix
  projections rather than intermediate flat-sky grids.

All-sky LST coadds via `coadd_fits(..., nside=...)` plus nested-tile reverse
projection (`nested_tile_header` / `healpix_to_hdu`) are available for
downstream PyBDSF (see `lwa-catalog` notebook `ovro_lwa_healpix_tile_detect.ipynb`).
