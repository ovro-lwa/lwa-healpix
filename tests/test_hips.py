"""Tests for lwa_healpix.hips."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from conftest import _make_lwa_fits

from lwa_healpix.hips import (
    _car_header_for_nside,
    fits_to_hips,
    healpix_to_hips,
)


# ---------------------------------------------------------------------------
# _car_header_for_nside
# ---------------------------------------------------------------------------


class TestCarHeaderForNside:
    def test_pixel_scale_matches_nside(self):
        nside = 512
        header = _car_header_for_nside(nside)
        expected_scale = np.degrees(np.sqrt(4 * np.pi / (12 * nside**2)))
        assert abs(header["CDELT2"] - expected_scale) < 0.01 * expected_scale

    def test_galactic_ctypes(self):
        header = _car_header_for_nside(64, coord_frame="galactic")
        assert header["CTYPE1"] == "GLON-CAR"
        assert header["CTYPE2"] == "GLAT-CAR"

    def test_equatorial_ctypes(self):
        header = _car_header_for_nside(64, coord_frame="equatorial")
        assert header["CTYPE1"] == "RA---CAR"
        assert header["CTYPE2"] == "DEC--CAR"

    def test_full_sky_coverage(self):
        header = _car_header_for_nside(128)
        nx, ny = header["NAXIS1"], header["NAXIS2"]
        cdelt = abs(header["CDELT1"])
        assert nx * cdelt >= 359.9
        assert ny * cdelt >= 179.9


# ---------------------------------------------------------------------------
# healpix_to_hips
# ---------------------------------------------------------------------------


class TestHealpixToHips:
    def test_output_directory_created(self, tmp_path):
        nside = 8
        npix = 12 * nside**2
        healpix_map = np.ones(npix, dtype=np.float32)

        out_dir = tmp_path / "hips_test"
        healpix_to_hips(healpix_map, output_directory=out_dir, threads=False)

        assert out_dir.is_dir()
        assert (out_dir / "properties").exists()
        assert (out_dir / "index.html").exists()

    def test_norder_directories_exist(self, tmp_path):
        nside = 8
        npix = 12 * nside**2
        healpix_map = np.ones(npix, dtype=np.float32)

        out_dir = tmp_path / "hips_norder"
        healpix_to_hips(healpix_map, output_directory=out_dir, threads=False)

        norder_dirs = sorted(out_dir.glob("Norder*"))
        assert len(norder_dirs) >= 1


# ---------------------------------------------------------------------------
# fits_to_hips
# ---------------------------------------------------------------------------


class TestFitsToHips:
    def test_output_from_array_header(self, tmp_path):
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 64
        header["NAXIS2"] = 64
        header["CTYPE1"] = "RA---SIN"
        header["CRPIX1"] = 32.5
        header["CRVAL1"] = 180.0
        header["CDELT1"] = -0.5
        header["CUNIT1"] = "deg"
        header["CTYPE2"] = "DEC--SIN"
        header["CRPIX2"] = 32.5
        header["CRVAL2"] = 34.0
        header["CDELT2"] = 0.5
        header["CUNIT2"] = "deg"

        data = np.ones((64, 64), dtype=np.float32)

        out_dir = tmp_path / "hips_fits"
        fits_to_hips(
            (data, header),
            output_directory=out_dir,
            threads=False,
        )

        assert out_dir.is_dir()
        assert (out_dir / "properties").exists()
        assert (out_dir / "index.html").exists()

    def test_output_from_fits_file(self, tmp_path):
        fpath = _make_lwa_fits(
            tmp_path / "img.fits", 30e6, nx=64, ny=64, pixel_scale=0.5,
        )
        hdu = fits.open(fpath)[0]
        data_2d = hdu.data[0, 0]
        header_2d = fits.Header()
        for key in ("NAXIS1", "NAXIS2", "CTYPE1", "CRPIX1", "CRVAL1",
                     "CDELT1", "CUNIT1", "CTYPE2", "CRPIX2", "CRVAL2",
                     "CDELT2", "CUNIT2"):
            if key in hdu.header:
                header_2d[key] = hdu.header[key]
        header_2d["NAXIS"] = 2

        out_dir = tmp_path / "hips_from_file"
        fits_to_hips(
            (data_2d, header_2d),
            output_directory=out_dir,
            threads=False,
        )

        assert out_dir.is_dir()
        norder_dirs = sorted(out_dir.glob("Norder*"))
        assert len(norder_dirs) >= 1
