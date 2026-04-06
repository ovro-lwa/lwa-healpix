"""Reproject FITS images, coadd, and build spectral cubes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import wcs as astropy_wcs
from astropy.io import fits
from reproject import reproject_interp, reproject_to_healpix

from .utils import (
    _extract_2d,
    _find_spectral_axis,
    _pixel_elevations,
    center_patch_rms_from_fits,
)

__all__ = [
    "coadd_fits",
    "combine_fits_to_spectral_cube",
]

logger = logging.getLogger(__name__)


def _screen_paths_by_quality(
    file_paths: list[str | Path],
    *,
    quality_max_rms: float | None,
    quality_outlier_sigma: float | None,
    quality_metric: Literal["std", "mad_sigma"],
    quality_center_fraction: float,
    quality_center_max_pixels: int | None,
    min_elevation: float | None,
) -> list[Path]:
    """Return paths that pass center-patch quality checks."""
    rows: list[tuple[Path, float]] = []
    for fp in file_paths:
        p = Path(fp)
        rms = center_patch_rms_from_fits(
            p,
            center_fraction=quality_center_fraction,
            center_max_pixels=quality_center_max_pixels,
            metric=quality_metric,
            min_elevation=min_elevation,
        )
        rows.append((p, rms))
        if np.isnan(rms):
            logger.warning(
                "Quality screen: no finite pixels in center patch, rejecting %s",
                p,
            )

    kept = [(p, r) for p, r in rows if not np.isnan(r)]

    if quality_max_rms is not None:
        nxt: list[tuple[Path, float]] = []
        for p, r in kept:
            if r > quality_max_rms:
                logger.warning(
                    "Quality screen: center-patch %s=%.6g exceeds quality_max_rms=%.6g, rejecting %s",
                    quality_metric, r, quality_max_rms, p,
                )
            else:
                nxt.append((p, r))
        kept = nxt

    if quality_outlier_sigma is not None and len(kept) > 0:
        rvals = np.array([r for _, r in kept], dtype=np.float64)
        med = float(np.median(rvals))
        mad = float(np.median(np.abs(rvals - med)))
        thresh = med + quality_outlier_sigma * 1.4826 * mad
        nxt = []
        for p, r in kept:
            if r > thresh:
                logger.warning(
                    "Quality screen: center-patch %s=%.6g above outlier threshold %.6g "
                    "(median=%.6g), rejecting %s",
                    quality_metric, r, thresh, med, p,
                )
            else:
                nxt.append((p, r))
        kept = nxt

    return [p for p, _ in kept]


def coadd_fits(
    file_paths: list[str | Path],
    *,
    nside: int | None = None,
    target_header: fits.Header | None = None,
    coord_frame: str = "galactic",
    nested: bool = False,
    min_elevation: float | None = None,
    quality_max_rms: float | None = None,
    quality_outlier_sigma: float | None = None,
    quality_metric: Literal["std", "mad_sigma"] = "std",
    quality_center_fraction: float = 0.25,
    quality_center_max_pixels: int | None = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject FITS images onto a common grid and coadd them.

    Each input file is expected to have 4 axes ``(RA, Dec, Freq,
    Stokes)`` with length-1 frequency and Stokes dimensions.  The 2-D
    spatial plane is extracted, optionally masked by elevation, then
    reprojected and accumulated with footprint-based weighting.

    Exactly one of *nside* or *target_header* must be given:

    * **nside** — reproject onto a HEALPix grid.  Returns a 1-D map.
    * **target_header** — reproject onto a 2-D image grid described by
      the FITS header.  Returns a 2-D array.

    Optional **quality screening** (when ``quality_max_rms`` and/or
    ``quality_outlier_sigma`` is set) runs *before* reprojection: it
    opens each file with memory mapping, computes dispersion (std or
    MAD-based scale) on a **central patch** only, and drops files that
    fail the thresholds.  This avoids expensive full reprojection of
    clearly bad images.  Units for ``quality_max_rms`` match the image
    data (see ``BUNIT``).  When ``min_elevation`` is set, the same
    blanking is applied to the central patch before the metric.

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to FITS files to reproject and coadd.
    nside : int or None, optional
        HEALPix NSIDE parameter.  Mutually exclusive with
        *target_header*.
    target_header : `~astropy.io.fits.Header` or None, optional
        WCS header for a 2-D target grid.  Must include ``NAXIS1`` and
        ``NAXIS2``.  Mutually exclusive with *nside*.
    coord_frame : str, optional
        Coordinate frame for the output (e.g. ``"galactic"``).  Used
        when *nside* is given.  Default is ``"galactic"``.
    nested : bool, optional
        HEALPix NESTED ordering.  Only relevant when *nside* is given.
    min_elevation : float or None, optional
        Minimum elevation above the horizon in degrees.  Pixels below
        this elevation are blanked before reprojection.  Elevation is
        measured as the angular distance from the image reference point
        (assumed to be the local zenith).
    quality_max_rms : float or None, optional
        If set, reject images whose center-patch dispersion exceeds this
        value.
    quality_outlier_sigma : float or None, optional
        If set, reject images whose center-patch dispersion exceeds
        ``median + sigma * 1.4826 * MAD`` over the surviving batch.
    quality_metric : {\"std\", \"mad_sigma\"}, optional
        Dispersion statistic on the center patch (default ``\"std\"``).
    quality_center_fraction : float, optional
        Linear fraction of each spatial axis used for the center patch
        (default ``0.25``).
    quality_center_max_pixels : int or None, optional
        Maximum patch size in pixels per axis (default ``512``).

    Returns
    -------
    combined : numpy.ndarray
        Weighted-average coadded map (1-D for HEALPix, 2-D for image).
    total_weight : numpy.ndarray
        Sum of footprint weights per pixel.
    """
    if (nside is None) == (target_header is None):
        msg = "Exactly one of 'nside' or 'target_header' must be given"
        raise ValueError(msg)

    if not file_paths:
        msg = "At least one FITS file path is required"
        raise ValueError(msg)

    paths = list(file_paths)
    if quality_max_rms is not None or quality_outlier_sigma is not None:
        paths = _screen_paths_by_quality(
            paths,
            quality_max_rms=quality_max_rms,
            quality_outlier_sigma=quality_outlier_sigma,
            quality_metric=quality_metric,
            quality_center_fraction=quality_center_fraction,
            quality_center_max_pixels=quality_center_max_pixels,
            min_elevation=min_elevation,
        )
        if not paths:
            msg = "All input images were rejected by quality screening"
            raise ValueError(msg)

    if target_header is not None:
        shape_out = (
            int(target_header["NAXIS2"]),
            int(target_header["NAXIS1"]),
        )

    sum_data: np.ndarray | None = None
    sum_weight: np.ndarray | None = None

    for fpath in paths:
        hdu = fits.open(fpath)[0]
        data_2d, wcs_2d = _extract_2d(hdu)

        if min_elevation is not None:
            elevation = _pixel_elevations(wcs_2d, data_2d.shape)
            data_2d = data_2d.copy()
            data_2d[elevation < min_elevation] = np.nan

        if nside is not None:
            reprojected, footprint = reproject_to_healpix(
                (data_2d, wcs_2d), coord_frame,
                nside=nside, nested=nested,
            )
        else:
            # Use the 2-D celestial WCS for the output grid.  Passing a Header
            # with NAXIS>2 makes ``reproject`` build a multi-dimensional output
            # WCS, which does not match the 2-D input from ``_extract_2d``.
            wcs_out = astropy_wcs.WCS(target_header).celestial
            reprojected, footprint = reproject_interp(
                (data_2d, wcs_2d), wcs_out,
                shape_out=shape_out,
            )

        reprojected = np.nan_to_num(reprojected, nan=0.0)

        if sum_data is None:
            sum_data = reprojected * footprint
            sum_weight = footprint.copy()
        else:
            sum_data += reprojected * footprint
            sum_weight += footprint

    total_weight = sum_weight
    combined = np.zeros_like(sum_data)
    np.divide(sum_data, total_weight, out=combined, where=total_weight > 0)
    return combined, total_weight


def combine_fits_to_spectral_cube(
    file_paths: list[str | Path] | dict[float, list[str | Path]],
    output_path: str | Path,
    *,
    freq_values: list[float] | np.ndarray | None = None,
    min_elevation: float | None = None,
    quality_max_rms: float | None = None,
    quality_outlier_sigma: float | None = None,
    quality_metric: Literal["std", "mad_sigma"] = "std",
    quality_center_fraction: float = 0.25,
    quality_center_max_pixels: int | None = 512,
) -> fits.HDUList:
    """Combine FITS images into a spectral cube, with optional coadding.

    Each input file is expected to contain a 4-D FITS image with axes
    ``(RA, Dec, Freq, Stokes)`` where the frequency and Stokes axes are
    both length 1.

    *file_paths* may be given in two forms:

    * **flat list** -- one file per frequency channel.  All files must
      share the same spatial pixel grid.
    * **dict** -- keys are frequency values and values are lists of
      files at that frequency.  Files within each group are reprojected
      onto a common spatial grid (from the first file of the lowest-
      frequency group) and coadded via :func:`coadd_fits` before being
      stacked into the cube.

    When a flat list contains duplicate frequencies, files at the same
    frequency are automatically grouped and coadded.

    Parameters
    ----------
    file_paths : list or dict
        Paths to the FITS files.  A flat list gives one file per channel
        (or auto-groups by frequency).  A ``dict[float, list[path]]``
        maps each frequency to files to be coadded.
    output_path : str or Path
        Path for the output spectral-cube FITS file.
    freq_values : array-like or None, optional
        Explicit frequency values for each file in the *flat-list* form.
        Ignored when *file_paths* is a dict (keys are the frequencies).
    min_elevation : float or None, optional
        Minimum elevation in degrees.  Passed through to
        :func:`coadd_fits` for per-channel coadding.
    quality_max_rms : float or None, optional
        Passed to :func:`coadd_fits` when coadding multiple files per
        frequency.
    quality_outlier_sigma : float or None, optional
        Passed to :func:`coadd_fits`.
    quality_metric : {\"std\", \"mad_sigma\"}, optional
        Passed to :func:`coadd_fits`.
    quality_center_fraction : float, optional
        Passed to :func:`coadd_fits`.
    quality_center_max_pixels : int or None, optional
        Passed to :func:`coadd_fits`.

    Returns
    -------
    hdul : `~astropy.io.fits.HDUList`
        The written spectral-cube HDU list.

    Raises
    ------
    ValueError
        If fewer than two distinct frequency channels result, spatial
        dimensions are inconsistent, or frequency information cannot be
        determined.
    """
    # ------------------------------------------------------------------
    # Normalise input to  freq_groups: dict[float, list[Path]]
    # ------------------------------------------------------------------
    if isinstance(file_paths, dict):
        freq_groups: dict[float, list[str | Path]] = {
            float(k): list(v) for k, v in file_paths.items()
        }
    else:
        if len(file_paths) < 2:
            msg = "At least two FITS files are required to build a spectral cube"
            raise ValueError(msg)

        freqs_raw: list[float] = []
        for idx, fpath in enumerate(file_paths):
            if freq_values is not None:
                freqs_raw.append(float(freq_values[idx]))
            else:
                hdr = fits.getheader(fpath)
                ax = _find_spectral_axis(hdr)
                freqs_raw.append(float(hdr[f"CRVAL{ax}"]))

        freq_groups = {}
        for freq, fpath in zip(freqs_raw, file_paths):
            freq_groups.setdefault(freq, []).append(fpath)

    sorted_freqs = sorted(freq_groups)
    if len(sorted_freqs) < 2:
        msg = "At least two distinct frequency channels are required"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Reference spatial grid from the first file at the lowest frequency.
    # ------------------------------------------------------------------
    first_hdu = fits.open(freq_groups[sorted_freqs[0]][0])[0]
    ref_data, ref_wcs_2d = _extract_2d(first_hdu)
    if ref_data.ndim != 2:
        msg = (
            f"Expected 2-D spatial image after removing Stokes and Freq "
            f"axes, got {ref_data.ndim}-D"
        )
        raise ValueError(msg)
    ny, nx = ref_data.shape
    ref_header_full = first_hdu.header

    ref_target = ref_wcs_2d.to_header()
    ref_target["NAXIS"] = 2
    ref_target["NAXIS1"] = nx
    ref_target["NAXIS2"] = ny

    # ------------------------------------------------------------------
    # Build each frequency plane
    # ------------------------------------------------------------------
    nfreq = len(sorted_freqs)
    cube = np.empty((nfreq, ny, nx), dtype=np.float32)

    for i, freq in enumerate(sorted_freqs):
        group = freq_groups[freq]
        if len(group) == 1:
            hdu = fits.open(group[0])[0]
            plane, _ = _extract_2d(hdu)
            if plane.shape != (ny, nx):
                msg = (
                    f"Spatial shape mismatch: expected ({ny}, {nx}), "
                    f"got {plane.shape} from {group[0]}"
                )
                raise ValueError(msg)
        else:
            plane, _ = coadd_fits(
                group,
                target_header=ref_target,
                min_elevation=min_elevation,
                quality_max_rms=quality_max_rms,
                quality_outlier_sigma=quality_outlier_sigma,
                quality_metric=quality_metric,
                quality_center_fraction=quality_center_fraction,
                quality_center_max_pixels=quality_center_max_pixels,
            )
        cube[i] = plane

    # ------------------------------------------------------------------
    # Assemble the 3-D WCS header
    # ------------------------------------------------------------------
    header_2d = ref_wcs_2d.to_header()
    freqs_sorted = np.array(sorted_freqs)

    header_3d = fits.Header()
    header_3d["NAXIS"] = 3
    header_3d["NAXIS1"] = nx
    header_3d["NAXIS2"] = ny
    header_3d["NAXIS3"] = nfreq

    for card in header_2d.cards:
        if card.keyword == "WCSAXES":
            header_3d["WCSAXES"] = 3
        elif card.keyword:
            header_3d[card.keyword] = (card.value, card.comment)

    freq_ax = _find_spectral_axis(ref_header_full)
    freq_suffix = str(freq_ax)
    header_3d["CTYPE3"] = ref_header_full.get(f"CTYPE{freq_suffix}", "FREQ")
    header_3d["CRPIX3"] = 1.0
    header_3d["CRVAL3"] = freqs_sorted[0]
    header_3d["CUNIT3"] = ref_header_full.get(f"CUNIT{freq_suffix}", "Hz")

    if nfreq > 1:
        cdelt3 = float(freqs_sorted[1] - freqs_sorted[0])
        if cdelt3 == 0:
            msg = "First two frequencies are identical; cannot determine channel spacing"
            raise ValueError(msg)
        header_3d["CDELT3"] = cdelt3

        expected = freqs_sorted[0] + np.arange(nfreq) * cdelt3
        max_deviation = np.max(np.abs(freqs_sorted - expected))
        if max_deviation > 0.01 * abs(cdelt3):
            logger.warning(
                "Frequencies are not uniformly spaced (max deviation: "
                "%.3g %s). The FITS WCS frequency axis assumes uniform "
                "spacing; downstream tools may misinterpret channel "
                "frequencies.",
                max_deviation,
                header_3d["CUNIT3"],
            )

    for key in ("TELESCOP", "INSTRUME", "OBSERVER", "OBJECT",
                "DATE-OBS", "BUNIT", "EQUINOX", "RADESYS"):
        val = ref_header_full.get(key)
        if val is not None:
            header_3d[key] = val

    primary = fits.PrimaryHDU(data=cube, header=header_3d)
    hdul = fits.HDUList([primary])

    output_path = Path(output_path)
    hdul.writeto(output_path, overwrite=True)
    logger.info(
        "Wrote spectral cube (%d channels, %d x %d) to %s",
        nfreq, nx, ny, output_path,
    )
    return hdul
