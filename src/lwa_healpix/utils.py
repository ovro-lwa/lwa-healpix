"""Shared helpers and OVRO-LWA pipeline utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits

__all__ = [
    "OVRO_LATITUDE_DEG",
    "altaz_to_ha_dec",
    "center_patch_rms_from_fits",
    "elliptical_mask_edge_zenith_angle_deg",
    "group_pipeline_files",
    "lst_hour_from_path",
    "parallactic_angle_deg",
    "parallactic_delta_q_edge_summary",
]

_FREQ_DIR_RE = re.compile(r"(\d+)\s*MHz", re.IGNORECASE)
_LST_DIR_RE = re.compile(r"^(\d+)h$", re.IGNORECASE)

# Owens Valley Radio Observatory (same value as lwa_catalog.constants)
OVRO_LATITUDE_DEG: float = 37.239777

_CARDINAL_AZ_DEG: tuple[float, ...] = tuple(float(a) for a in range(0, 360, 45))
_CARDINAL_LABELS: tuple[str, ...] = (
    "N", "NE", "E", "SE", "S", "SW", "W", "NW",
)


def _resolve_elevation_cut(
    *,
    min_elevation: float | None = None,
    min_elevation_ns: float | None = None,
    min_elevation_ew: float | None = None,
) -> tuple[str, float] | tuple[str, float, float] | None:
    """Validate elevation kwargs.

    Returns
    -------
    None
        No blanking.
    ``(\"circular\", elev)``
        Scalar circular minimum elevation (degrees).
    ``(\"ellipse\", elev_ns, elev_ew)``
        Elliptical cut with N/S and E/W minimum elevations (degrees).
    """
    has_scalar = min_elevation is not None
    has_ns = min_elevation_ns is not None
    has_ew = min_elevation_ew is not None
    if has_ns != has_ew:
        msg = "min_elevation_ns and min_elevation_ew must be set together"
        raise ValueError(msg)
    if has_scalar and (has_ns or has_ew):
        msg = (
            "Use either min_elevation (circular) or "
            "min_elevation_ns/min_elevation_ew (elliptical), not both"
        )
        raise ValueError(msg)
    if has_scalar:
        return ("circular", float(min_elevation))
    if has_ns and has_ew:
        return ("ellipse", float(min_elevation_ns), float(min_elevation_ew))
    return None


def elliptical_mask_edge_zenith_angle_deg(
    az_deg: float | np.ndarray,
    *,
    min_elevation_ns: float,
    min_elevation_ew: float,
) -> float | np.ndarray:
    """Zenith angle (deg) of the elliptical elevation-mask edge at azimuth *az_deg*.

    Azimuth is measured from north through east (same as astronomical PA from
    zenith). Semi-axes are ``z_ns = 90 − elev_ns`` and ``z_ew = 90 − elev_ew``.
    """
    z_ns = 90.0 - float(min_elevation_ns)
    z_ew = 90.0 - float(min_elevation_ew)
    if z_ns <= 0.0 or z_ew <= 0.0:
        msg = "min_elevation_ns/ew must be < 90 deg"
        raise ValueError(msg)
    az = np.deg2rad(np.asarray(az_deg, dtype=float))
    # z such that (z sin / z_ew)^2 + (z cos / z_ns)^2 = 1
    denom = np.sqrt((np.sin(az) / z_ew) ** 2 + (np.cos(az) / z_ns) ** 2)
    z = 1.0 / denom
    if np.ndim(az_deg) == 0:
        return float(z)
    return z


def _elevation_outside_mask(
    z_deg: np.ndarray,
    pa_rad: np.ndarray,
    *,
    min_elevation: float | None = None,
    min_elevation_ns: float | None = None,
    min_elevation_ew: float | None = None,
) -> np.ndarray:
    """Return True where pixels lie outside the elevation keep-mask (should blank)."""
    cut = _resolve_elevation_cut(
        min_elevation=min_elevation,
        min_elevation_ns=min_elevation_ns,
        min_elevation_ew=min_elevation_ew,
    )
    if cut is None:
        return np.zeros(np.shape(z_deg), dtype=bool)
    z = np.asarray(z_deg, dtype=float)
    if cut[0] == "circular":
        elev_min = cut[1]
        return z > (90.0 - elev_min)
    _, elev_ns, elev_ew = cut
    z_ns = 90.0 - elev_ns
    z_ew = 90.0 - elev_ew
    if z_ns <= 0.0 or z_ew <= 0.0:
        msg = "min_elevation_ns/ew must be < 90 deg"
        raise ValueError(msg)
    pa = np.asarray(pa_rad, dtype=float)
    r2 = (z * np.sin(pa) / z_ew) ** 2 + (z * np.cos(pa) / z_ns) ** 2
    return r2 > 1.0


def _pixel_zenith_angle_pa(
    wcs_2d: wcs.WCS,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(zenith_angle_deg, pa_rad)`` for every pixel (CRVAL = zenith)."""
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    sky = wcs_2d.pixel_to_world(x, y)
    center = SkyCoord(
        wcs_2d.wcs.crval[0], wcs_2d.wcs.crval[1],
        unit="deg", frame=sky.frame.name,
    )
    z_deg = sky.separation(center).deg
    pa_rad = center.position_angle(sky).rad
    return np.asarray(z_deg, dtype=float), np.asarray(pa_rad, dtype=float)


def _pixel_zenith_angle_pa_window(
    wcs_2d: wcs.WCS,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Zenith angle / PA for a pixel window; same model as `_pixel_zenith_angle_pa`."""
    y, x = np.mgrid[y0:y1, x0:x1]
    sky = wcs_2d.pixel_to_world(x, y)
    center = SkyCoord(
        wcs_2d.wcs.crval[0], wcs_2d.wcs.crval[1],
        unit="deg", frame=sky.frame.name,
    )
    z_deg = sky.separation(center).deg
    pa_rad = center.position_angle(sky).rad
    return np.asarray(z_deg, dtype=float), np.asarray(pa_rad, dtype=float)


def _pixel_elevations(wcs_2d: wcs.WCS, shape: tuple[int, int]) -> np.ndarray:
    """Return the elevation in degrees of every pixel in a 2-D image.

    Elevation is defined as 90 degrees minus the angular separation from
    the image reference point (``CRVAL``), which is assumed to be the
    local zenith.
    """
    z_deg, _ = _pixel_zenith_angle_pa(wcs_2d, shape)
    return 90.0 - z_deg


def _blank_data_outside_elevation(
    data_2d: np.ndarray,
    wcs_2d: wcs.WCS,
    *,
    min_elevation: float | None = None,
    min_elevation_ns: float | None = None,
    min_elevation_ew: float | None = None,
) -> np.ndarray:
    """Return a copy of *data_2d* with outside-mask pixels set to NaN."""
    cut = _resolve_elevation_cut(
        min_elevation=min_elevation,
        min_elevation_ns=min_elevation_ns,
        min_elevation_ew=min_elevation_ew,
    )
    if cut is None:
        return data_2d
    z_deg, pa_rad = _pixel_zenith_angle_pa(wcs_2d, data_2d.shape)
    outside = _elevation_outside_mask(
        z_deg,
        pa_rad,
        min_elevation=min_elevation,
        min_elevation_ns=min_elevation_ns,
        min_elevation_ew=min_elevation_ew,
    )
    out = np.array(data_2d, copy=True, dtype=np.float64)
    out[outside] = np.nan
    return out


def _wrap180(deg: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to ``(-180, 180]`` degrees."""
    x = np.asarray(deg, dtype=float)
    wrapped = (x + 180.0) % 360.0 - 180.0
    if np.ndim(deg) == 0:
        return float(wrapped)
    return wrapped


def altaz_to_ha_dec(
    alt_deg: float,
    az_deg: float,
    *,
    latitude_deg: float = OVRO_LATITUDE_DEG,
) -> tuple[float, float]:
    """Convert altitude/azimuth (deg; az N→E) to hour angle and declination (deg)."""
    alt = np.deg2rad(float(alt_deg))
    az = np.deg2rad(float(az_deg))
    lat = np.deg2rad(float(latitude_deg))
    sin_dec = np.sin(alt) * np.sin(lat) + np.cos(alt) * np.cos(lat) * np.cos(az)
    sin_dec = float(np.clip(sin_dec, -1.0, 1.0))
    dec = float(np.arcsin(sin_dec))
    cos_dec = np.cos(dec)
    if abs(cos_dec) < 1e-12:
        ha = 0.0
    else:
        sin_ha = -np.cos(alt) * np.sin(az) / cos_dec
        cos_ha = (
            np.sin(alt) * np.cos(lat) - np.cos(alt) * np.sin(lat) * np.cos(az)
        ) / cos_dec
        ha = float(np.arctan2(sin_ha, cos_ha))
    return float(np.rad2deg(ha)), float(np.rad2deg(dec))


def parallactic_angle_deg(
    ha_deg: float | np.ndarray,
    dec_deg: float,
    *,
    latitude_deg: float = OVRO_LATITUDE_DEG,
) -> float | np.ndarray:
    """Parallactic angle (deg) for hour angle / declination at *latitude_deg*."""
    ha = np.deg2rad(np.asarray(ha_deg, dtype=float))
    dec = np.deg2rad(float(dec_deg))
    lat = np.deg2rad(float(latitude_deg))
    q = np.arctan2(
        np.sin(ha),
        np.cos(dec) * np.tan(lat) - np.sin(dec) * np.cos(ha),
    )
    q_deg = np.rad2deg(q)
    if np.ndim(ha_deg) == 0:
        return float(q_deg)
    return q_deg


def parallactic_delta_q_edge_summary(
    *,
    min_elevation_ns: float,
    min_elevation_ew: float,
    latitude_deg: float = OVRO_LATITUDE_DEG,
) -> list[dict[str, float | str]]:
    """1-hour parallactic-angle change at 8 cardinal elliptical-mask edges.

    For each azimuth ``0, 45, …, 315°``, place a point on the elevation-mask
    edge, convert to ``(HA₀, Dec)``, then for LST bins ``h = 0…23`` compute
    ``Δq_h = wrap180(q(HA₀ + (h+1)·15°) − q(HA₀ + h·15°))``.  Returns one
    dict per direction with range / min / max / median of ``|Δq|``.
    """
    rows: list[dict[str, float | str]] = []
    for az, label in zip(_CARDINAL_AZ_DEG, _CARDINAL_LABELS, strict=True):
        z_edge = float(
            elliptical_mask_edge_zenith_angle_deg(
                az,
                min_elevation_ns=min_elevation_ns,
                min_elevation_ew=min_elevation_ew,
            )
        )
        elev = 90.0 - z_edge
        ha0, dec = altaz_to_ha_dec(elev, az, latitude_deg=latitude_deg)
        delta: list[float] = []
        for h in range(24):
            ha_a = ha0 + h * 15.0
            ha_b = ha0 + (h + 1) * 15.0
            dq = _wrap180(
                parallactic_angle_deg(ha_b, dec, latitude_deg=latitude_deg)
                - parallactic_angle_deg(ha_a, dec, latitude_deg=latitude_deg)
            )
            delta.append(float(dq))
        darr = np.asarray(delta, dtype=float)
        abs_d = np.abs(darr)
        rows.append(
            {
                "direction": label,
                "az_deg": float(az),
                "elev_edge_deg": elev,
                "zenith_angle_deg": z_edge,
                "dec_deg": dec,
                "ha0_deg": ha0,
                "dq_range_deg": float(np.nanmax(darr) - np.nanmin(darr)),
                "dq_min_deg": float(np.nanmin(darr)),
                "dq_max_deg": float(np.nanmax(darr)),
                "abs_dq_min_deg": float(np.nanmin(abs_d)),
                "abs_dq_max_deg": float(np.nanmax(abs_d)),
                "abs_dq_median_deg": float(np.nanmedian(abs_d)),
            }
        )
    return rows


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


def _axis_center_slice(n: int, fraction: float, max_pixels: int | None) -> tuple[int, int]:
    """Return ``start, stop`` for a centered slice along one axis of length *n*."""
    w = max(1, int(n * fraction))
    if max_pixels is not None:
        w = min(w, max_pixels)
    w = min(w, n)
    w = max(1, w)
    start = (n - w) // 2
    return start, start + w


def _rms_from_finite_values(
    patch: np.ndarray,
    metric: str,
) -> float:
    """Std or robust MAD×1.4826 on finite values."""
    v = patch[np.isfinite(patch)].astype(np.float64, copy=False).ravel()
    if v.size == 0:
        return float("nan")
    if metric == "std":
        return float(np.std(v))
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return 1.4826 * mad


def _pixel_elevations_window(
    wcs_2d: wcs.WCS,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Elevation (deg) for a pixel window; same zenith model as `_pixel_elevations`."""
    z_deg, _ = _pixel_zenith_angle_pa_window(wcs_2d, y0, y1, x0, x1)
    return 90.0 - z_deg


def center_patch_rms_from_fits(
    path: str | Path,
    *,
    center_fraction: float = 0.25,
    center_max_pixels: int | None = 512,
    metric: str = "std",
    min_elevation: float | None = None,
    min_elevation_ns: float | None = None,
    min_elevation_ew: float | None = None,
) -> float:
    """Cheap quality metric: dispersion on a central patch only (memmap slice).

    Reads only the central region of the spatial plane.  Supports 2-D images
    and 4-D LWA-style arrays whose last two axes are spatial.

    Elevation blanking uses the same circular / elliptical rules as
    :func:`~lwa_healpix.coadd.coadd_fits` (``min_elevation`` or
    ``min_elevation_ns`` / ``min_elevation_ew``).
    """
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        hdu = hdul[0]
        hdr = hdu.header
        data = hdu.data
        if data is None:
            return float("nan")

        naxis1 = int(hdr.get("NAXIS1", 1))
        naxis2 = int(hdr.get("NAXIS2", 1))

        if data.ndim == 2:
            ny, nx = data.shape
            sy, ey = _axis_center_slice(ny, center_fraction, center_max_pixels)
            sx, ex = _axis_center_slice(nx, center_fraction, center_max_pixels)
            patch = np.asarray(data[sy:ey, sx:ex])
            wcs_2d = wcs.WCS(hdr).celestial
        elif data.ndim == 4:
            ny, nx = naxis2, naxis1
            sy, ey = _axis_center_slice(ny, center_fraction, center_max_pixels)
            sx, ex = _axis_center_slice(nx, center_fraction, center_max_pixels)
            patch = np.asarray(data[0, 0, sy:ey, sx:ex])
            wcs_2d = wcs.WCS(hdr).celestial
        elif data.ndim == 3:
            ny, nx = data.shape[-2], data.shape[-1]
            sy, ey = _axis_center_slice(ny, center_fraction, center_max_pixels)
            sx, ex = _axis_center_slice(nx, center_fraction, center_max_pixels)
            patch = np.asarray(data[0, sy:ey, sx:ex])
            wcs_2d = wcs.WCS(hdr).celestial
        else:
            return float("nan")

        cut = _resolve_elevation_cut(
            min_elevation=min_elevation,
            min_elevation_ns=min_elevation_ns,
            min_elevation_ew=min_elevation_ew,
        )
        if cut is not None:
            z_deg, pa_rad = _pixel_zenith_angle_pa_window(wcs_2d, sy, ey, sx, ex)
            outside = _elevation_outside_mask(
                z_deg,
                pa_rad,
                min_elevation=min_elevation,
                min_elevation_ns=min_elevation_ns,
                min_elevation_ew=min_elevation_ew,
            )
            patch = patch.copy()
            patch[outside] = np.nan

        return _rms_from_finite_values(patch, metric)



def lst_hour_from_path(path: str | Path) -> int:
    """Return the integer LST hour from an OVRO-LWA pipeline path.

    Looks for a path component of the form ``{hour}h`` (e.g. ``10h``,
    ``14h``), as in
    ``/lustre/pipeline/images/10h/.../41MHz/I/deep/image.fits``.

    Parameters
    ----------
    path : str or Path
        Path to a pipeline FITS file.

    Returns
    -------
    hour : int
        Local sidereal time hour, 0–23.

    Raises
    ------
    ValueError
        If no LST hour component can be found in the path.
    """
    for part in Path(path).parts:
        m = _LST_DIR_RE.fullmatch(part)
        if m is not None:
            hour = int(m.group(1))
            if 0 <= hour <= 23:
                return hour
    msg = f"Cannot determine LST hour from path: {path}"
    raise ValueError(msg)


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
