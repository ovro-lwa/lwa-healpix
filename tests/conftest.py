"""Shared fixtures for lwa_healpix tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def _make_lwa_fits(
    path: Path,
    freq_hz: float,
    *,
    nx: int = 64,
    ny: int = 64,
    pixel_scale: float = 0.01,
    fill_value: float | None = None,
    use_cd_matrix: bool = False,
) -> Path:
    """Write a minimal 4-axis FITS file mimicking OVRO-LWA output.

    Axes are (RA, Dec, Freq, Stokes) with length-1 Freq and Stokes.
    """
    rng = np.random.default_rng(int(freq_hz))
    data = rng.standard_normal((1, 1, ny, nx)).astype(np.float32)
    if fill_value is not None:
        data[:] = fill_value

    header = fits.Header()
    header["NAXIS"] = 4
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["NAXIS3"] = 1
    header["NAXIS4"] = 1

    header["CTYPE1"] = "RA---SIN"
    header["CRPIX1"] = (nx + 1) / 2.0
    header["CRVAL1"] = 180.0
    header["CUNIT1"] = "deg"

    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX2"] = (ny + 1) / 2.0
    header["CRVAL2"] = 34.0
    header["CUNIT2"] = "deg"

    if use_cd_matrix:
        header["CD1_1"] = -pixel_scale
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = pixel_scale
    else:
        header["CDELT1"] = -pixel_scale
        header["CDELT2"] = pixel_scale

    header["CTYPE3"] = "FREQ"
    header["CRPIX3"] = 1.0
    header["CRVAL3"] = freq_hz
    header["CDELT3"] = 1e6
    header["CUNIT3"] = "Hz"

    header["CTYPE4"] = "STOKES"
    header["CRPIX4"] = 1.0
    header["CRVAL4"] = 1.0
    header["CDELT4"] = 1.0

    header["TELESCOP"] = "OVRO-LWA"
    header["BUNIT"] = "Jy/beam"

    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(path, overwrite=True)
    return path


@pytest.fixture()
def fits_files(tmp_path: Path) -> list[Path]:
    """Three FITS files at 30, 40, 50 MHz (unsorted on purpose)."""
    freqs = [40e6, 30e6, 50e6]
    return [
        _make_lwa_fits(tmp_path / f"obs_{int(f / 1e6)}MHz.fits", f)
        for f in freqs
    ]


@pytest.fixture()
def fits_files_cd_matrix(tmp_path: Path) -> list[Path]:
    """Three FITS files using a CD matrix instead of CDELT."""
    freqs = [40e6, 30e6, 50e6]
    return [
        _make_lwa_fits(
            tmp_path / f"cd_{int(f / 1e6)}MHz.fits", f, use_cd_matrix=True,
        )
        for f in freqs
    ]


@pytest.fixture()
def constant_fits_files(tmp_path: Path) -> list[Path]:
    """Two FITS files with constant pixel values (1.0 and 3.0)."""
    paths = []
    for val, freq in [(1.0, 30e6), (3.0, 40e6)]:
        p = _make_lwa_fits(
            tmp_path / f"const_{int(freq / 1e6)}MHz.fits",
            freq,
            fill_value=val,
        )
        paths.append(p)
    return paths


@pytest.fixture()
def wide_fits_files(tmp_path: Path) -> list[Path]:
    """FITS files with a wide FOV for HEALPix reprojection tests.

    Uses 0.5 deg/pixel × 64 px = 32° FOV so that even low-NSIDE HEALPix
    grids have covered pixels.
    """
    freqs = [40e6, 30e6, 50e6]
    return [
        _make_lwa_fits(
            tmp_path / f"wide_{int(f / 1e6)}MHz.fits",
            f,
            nx=64,
            ny=64,
            pixel_scale=0.5,
        )
        for f in freqs
    ]


@pytest.fixture()
def wide_constant_fits_files(tmp_path: Path) -> list[Path]:
    """Wide-FOV FITS files with constant pixel values (1.0 and 3.0)."""
    paths = []
    for val, freq in [(1.0, 30e6), (3.0, 40e6)]:
        p = _make_lwa_fits(
            tmp_path / f"wconst_{int(freq / 1e6)}MHz.fits",
            freq,
            fill_value=val,
            nx=64,
            ny=64,
            pixel_scale=0.5,
        )
        paths.append(p)
    return paths
