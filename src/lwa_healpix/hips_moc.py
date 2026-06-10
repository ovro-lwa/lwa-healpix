"""Write space–frequency MOC coverage for HiPS3D tile sets."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import astropy.units as u
import numpy as np
from mocpy import MOC, SFMOC
from reproject.hips.utils import spectral_index_to_coord

__all__ = [
    "C_LIGHT_M_S",
    "coverage_freq_range_hz",
    "effective_freq_range_hz",
    "fmoc_freq_range_from_tiles",
    "freq_range_from_cube_header",
    "wavelength_range_from_freq",
    "write_hips3d_moc",
]

logger = logging.getLogger(__name__)

C_LIGHT_M_S = 2.99792458e8

_NORDER3D_RE = re.compile(r"^Norder(\d+)_(\d+)$")
_NPIX3D_RE = re.compile(r"^Npix(\d+)_(\d+)\.")


def freq_range_from_cube_header(header) -> tuple[float, float, float]:
    """Return ``(freq_min_hz, freq_max_hz, initial_freq_hz)`` from a 3-D FITS header.

    The initial frequency is taken at the centre channel (``NAXIS3 // 2``).
    """
    crval3 = float(header["CRVAL3"])
    naxis3 = int(header["NAXIS3"])
    cdelt3 = float(header.get("CDELT3", 0.0))

    if naxis3 < 1:
        msg = "NAXIS3 must be at least 1"
        raise ValueError(msg)

    mid = naxis3 // 2
    initial = crval3 + mid * cdelt3

    if naxis3 == 1 or cdelt3 == 0.0:
        return crval3, crval3, initial

    channel_edges = crval3 + np.arange(naxis3 + 1) * cdelt3
    freq_min = float(min(channel_edges))
    freq_max = float(max(channel_edges))
    return freq_min, freq_max, initial


def wavelength_range_from_freq(
    freq_min_hz: float,
    freq_max_hz: float,
) -> tuple[float, float]:
    """Return ``(em_min, em_max)`` in metres for HiPS ``properties``.

    ``em_min`` corresponds to the highest frequency (shortest wavelength).
    """
    if freq_min_hz <= 0 or freq_max_hz <= 0:
        msg = "Frequency limits must be positive"
        raise ValueError(msg)
    lo = min(freq_min_hz, freq_max_hz)
    hi = max(freq_min_hz, freq_max_hz)
    return C_LIGHT_M_S / hi, C_LIGHT_M_S / lo


def _scan_max_level_tiles(
    output_directory: Path,
) -> tuple[int, int, set[int], set[int]]:
    """Return spatial/spectral orders and npix sets at the deepest tile level."""
    spatial_order = -1
    spectral_order = -1
    spatial_npix: set[int] = set()
    spectral_fpix: set[int] = set()

    for entry in output_directory.iterdir():
        if not entry.is_dir():
            continue
        match = _NORDER3D_RE.match(entry.name)
        if match is None:
            continue
        k, ell = int(match.group(1)), int(match.group(2))
        if (k, ell) < (spatial_order, spectral_order):
            continue
        if (k, ell) > (spatial_order, spectral_order):
            spatial_order = k
            spectral_order = ell
            spatial_npix = set()
            spectral_fpix = set()

        for tile_path in entry.rglob("Npix*_*.*"):
            npix_match = _NPIX3D_RE.match(tile_path.name)
            if npix_match is None:
                continue
            spatial_npix.add(int(npix_match.group(1)))
            spectral_fpix.add(int(npix_match.group(2)))

    if spatial_order < 0:
        msg = f"No HiPS3D tile directories found under {output_directory}"
        raise ValueError(msg)

    return spatial_order, spectral_order, spatial_npix, spectral_fpix


def fmoc_freq_range_from_tiles(
    spectral_order: int,
    spectral_fpix: set[int],
) -> tuple[float, float]:
    """Infer the FMOC frequency span covered by spectral tile indices."""
    if not spectral_fpix:
        msg = "No spectral tile indices found"
        raise ValueError(msg)

    freq_lo = min(
        spectral_index_to_coord(spectral_order, fpix).to_value(u.Hz)
        for fpix in spectral_fpix
    )
    freq_hi = max(
        spectral_index_to_coord(spectral_order, fpix + 1).to_value(u.Hz)
        for fpix in spectral_fpix
    )
    return float(freq_lo), float(freq_hi)


def effective_freq_range_hz(
    cube_freq_min_hz: float,
    cube_freq_max_hz: float,
    tile_freq_min_hz: float,
    tile_freq_max_hz: float,
) -> tuple[float, float]:
    """Intersect cube and tile FMOC ranges for tight coverage metadata."""
    cube_lo = min(cube_freq_min_hz, cube_freq_max_hz)
    cube_hi = max(cube_freq_min_hz, cube_freq_max_hz)
    tile_lo = min(tile_freq_min_hz, tile_freq_max_hz)
    tile_hi = max(tile_freq_min_hz, tile_freq_max_hz)
    eff_lo = max(cube_lo, tile_lo)
    eff_hi = min(cube_hi, tile_hi)
    if eff_lo > eff_hi:
        logger.warning(
            "Cube frequency range [%.6g, %.6g] Hz does not overlap tile FMOC "
            "range [%.6g, %.6g] Hz; using cube range for MOC",
            cube_lo,
            cube_hi,
            tile_lo,
            tile_hi,
        )
        return cube_lo, cube_hi
    return eff_lo, eff_hi


def coverage_freq_range_hz(
    output_directory: str | Path,
    cube_freq_min_hz: float,
    cube_freq_max_hz: float,
) -> tuple[float, float]:
    """Return cube∩tile FMOC frequency limits for metadata and MOC."""
    _, spectral_order, _, spectral_fpix = _scan_max_level_tiles(
        Path(output_directory),
    )
    tile_fmin, tile_fmax = fmoc_freq_range_from_tiles(spectral_order, spectral_fpix)
    return effective_freq_range_hz(
        cube_freq_min_hz, cube_freq_max_hz, tile_fmin, tile_fmax,
    )


def write_hips3d_moc(
    output_directory: str | Path,
    *,
    freq_min_hz: float,
    freq_max_hz: float,
    overwrite: bool = True,
) -> Path:
    """Write ``Moc.fits`` (SFMOC) describing HiPS3D spatial and frequency coverage.

    Frequency limits are intersected with the FMOC span of tiles present at
    the deepest spectral order so the MOC does not cover the full 1e-18–1e38 Hz
    FMOC universe.

    Parameters
    ----------
    output_directory : str or Path
        Root directory of the HiPS3D tile set.
    freq_min_hz, freq_max_hz : float
        Frequency coverage in hertz from the input spectral cube.
    overwrite : bool, optional
        Replace an existing ``Moc.fits`` (passed through to ``mocpy``).
        Default is ``True``.

    Returns
    -------
    path : Path
        Path to the written ``Moc.fits`` file.
    """
    output_directory = Path(output_directory)
    spatial_order, spectral_order, spatial_npix, spectral_fpix = (
        _scan_max_level_tiles(output_directory)
    )

    if not spatial_npix:
        msg = "No HiPS3D tiles found to build MOC coverage"
        raise ValueError(msg)

    tile_fmin, tile_fmax = fmoc_freq_range_from_tiles(spectral_order, spectral_fpix)
    eff_fmin, eff_fmax = effective_freq_range_hz(
        freq_min_hz, freq_max_hz, tile_fmin, tile_fmax,
    )

    spatial_moc = MOC.from_healpix_cells(
        sorted(spatial_npix),
        spatial_order,
        spatial_order,
    )
    sfmoc = SFMOC.from_spatial_coverages(
        eff_fmin * u.Hz,
        eff_fmax * u.Hz,
        [spatial_moc],
        max_order_frequency=spectral_order,
    )

    moc_path = output_directory / "Moc.fits"
    sfmoc.save(moc_path, format="fits", overwrite=overwrite)
    logger.info(
        "Wrote HiPS3D Moc.fits (%d spatial cells at order %d, "
        "freq order %d, %.6g–%.6g Hz from cube∩tiles)",
        len(spatial_npix),
        spatial_order,
        spectral_order,
        eff_fmin,
        eff_fmax,
    )
    return moc_path
