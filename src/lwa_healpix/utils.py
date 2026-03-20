"""Shared helpers and OVRO-LWA pipeline utilities."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits

__all__ = [
    "group_pipeline_files",
]

_FREQ_DIR_RE = re.compile(r"(\d+)\s*MHz", re.IGNORECASE)


def _pixel_elevations(wcs_2d: wcs.WCS, shape: tuple[int, int]) -> np.ndarray:
    """Return the elevation in degrees of every pixel in a 2-D image.

    Elevation is defined as 90 degrees minus the angular separation from
    the image reference point (``CRVAL``), which is assumed to be the
    local zenith.
    """
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    sky = wcs_2d.pixel_to_world(x, y)
    center = SkyCoord(
        wcs_2d.wcs.crval[0], wcs_2d.wcs.crval[1],
        unit="deg", frame=sky.frame.name,
    )
    return 90.0 - sky.separation(center).deg


def _find_spectral_axis(header: fits.Header) -> int:
    """Return the 1-based FITS axis number of the frequency axis.

    Looks at ``CTYPE{i}`` for ``i = 1 .. NAXIS``.  Some OVRO-LWA pipeline
    products are stored as **2-D images** (``NAXIS = 2``) but retain
    spectral keywords on axis 3 (e.g. ``CRVAL3`` without ``NAXIS3`` or
    ``CTYPE3``).  In that case this function returns ``3`` so callers
    can read ``CRVAL3`` / ``CTYPE3`` / ``CUNIT3`` consistently with
    :func:`combine_fits_to_spectral_cube` and HiPS3D workflows.

    Raises ``ValueError`` if no frequency axis can be found.
    """
    naxis = header.get("NAXIS", 0)
    for i in range(1, naxis + 1):
        ctype = header.get(f"CTYPE{i}", "")
        if ctype.upper().startswith("FREQ"):
            return i

    # Vestigial axis-3 spectral metadata on a 2-D (or 3-D) image.
    if "CRVAL3" in header:
        ctype3 = (header.get("CTYPE3") or "").strip()
        if not ctype3 or ctype3.upper().startswith("FREQ"):
            return 3

    msg = "Cannot find a FREQ axis in the FITS header"
    raise ValueError(msg)


def _find_stokes_axis(header: fits.Header) -> int:
    """Return the 1-based FITS axis number of the Stokes axis.

    Raises ``ValueError`` if no Stokes axis can be found.
    """
    for i in range(1, header.get("NAXIS", 0) + 1):
        ctype = header.get(f"CTYPE{i}", "")
        if ctype.upper().startswith("STOKES"):
            return i

    msg = "Cannot find a STOKES axis in the FITS header"
    raise ValueError(msg)


def _extract_2d(hdu: fits.PrimaryHDU) -> tuple[np.ndarray, wcs.WCS]:
    """Return the 2-D spatial data and WCS from an image HDU.

    If the HDU is already 2-D (``NAXIS = 2`` and data rank 2), returns
    the array and celestial WCS as-is.  This covers pipeline images that
    keep only vestigial frequency keywords (e.g. ``CRVAL3``) on axis 3.

    Otherwise expects a 4-D image with length-1 Freq and Stokes axes,
    drops those axes, and returns the spatial plane.
    """
    header = hdu.header
    data = hdu.data
    naxis = header.get("NAXIS", 0)

    if data is not None and data.ndim == 2 and naxis == 2:
        return np.asarray(data), wcs.WCS(header).celestial

    freq_ax = _find_spectral_axis(header)
    stokes_ax = _find_stokes_axis(header)

    naxis_hdr = header["NAXIS"]
    stokes_numpy = naxis_hdr - stokes_ax
    freq_numpy = naxis_hdr - freq_ax

    ax_hi, ax_lo = sorted([stokes_numpy, freq_numpy], reverse=True)
    data = np.take(np.take(hdu.data, 0, axis=ax_hi), 0, axis=ax_lo)
    wcs_2d = wcs.WCS(header).celestial
    return data, wcs_2d


def group_pipeline_files(
    file_paths: list[str | Path],
) -> dict[float, list[Path]]:
    """Group OVRO-LWA pipeline FITS files by frequency.

    The frequency for each file is determined by trying two methods in
    order:

    1. **Path parsing** — look for a ``{freq}MHz`` directory component
       in the file path (e.g.
       ``/lustre/pipeline/images/10h/.../41MHz/I/deep/image.fits``).
    2. **FITS header** — read ``CRVAL`` on the spectral (``FREQ``) axis
       from the file header.

    Files at the same frequency (but different LSTs, dates, or runs)
    are grouped together so they can be coadded by
    :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube` or
    :func:`~lwa_healpix.coadd.coadd_fits`.

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to FITS files produced by the pipeline.

    Returns
    -------
    groups : dict[float, list[Path]]
        Mapping from frequency in Hz to the list of files at that
        frequency, sorted by ascending frequency.

    Raises
    ------
    ValueError
        If the frequency cannot be determined from either the path or
        the FITS header.
    """
    groups: dict[float, list[Path]] = {}
    for fpath in file_paths:
        p = Path(fpath)
        freq_hz: float | None = None
        for part in p.parts:
            m = _FREQ_DIR_RE.fullmatch(part)
            if m:
                freq_hz = float(m.group(1)) * 1e6
                break

        if freq_hz is None:
            try:
                hdr = fits.getheader(p)
                ax = _find_spectral_axis(hdr)
                freq_hz = float(hdr[f"CRVAL{ax}"])
            except (OSError, KeyError, ValueError):
                msg = f"Cannot determine frequency from path or header: {fpath}"
                raise ValueError(msg)

        groups.setdefault(freq_hz, []).append(p)

    return dict(sorted(groups.items()))
