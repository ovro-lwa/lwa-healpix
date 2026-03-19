"""Tests for lwa_healpix.coadd."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy import wcs
from astropy.io import fits

from conftest import _make_lwa_fits

from lwa_healpix.coadd import (
    _car_header_for_nside,
    _find_spectral_axis,
    _find_stokes_axis,
    coadd_fits,
    combine_fits_to_spectral_cube,
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
# _find_spectral_axis / _find_stokes_axis
# ---------------------------------------------------------------------------


class TestFindAxes:
    def test_find_spectral_axis(self):
        header = fits.Header()
        header["NAXIS"] = 4
        header["CTYPE1"] = "RA---SIN"
        header["CTYPE2"] = "DEC--SIN"
        header["CTYPE3"] = "FREQ"
        header["CTYPE4"] = "STOKES"
        assert _find_spectral_axis(header) == 3

    def test_find_stokes_axis(self):
        header = fits.Header()
        header["NAXIS"] = 4
        header["CTYPE1"] = "RA---SIN"
        header["CTYPE2"] = "DEC--SIN"
        header["CTYPE3"] = "FREQ"
        header["CTYPE4"] = "STOKES"
        assert _find_stokes_axis(header) == 4

    def test_missing_spectral_axis_raises(self):
        header = fits.Header()
        header["NAXIS"] = 2
        header["CTYPE1"] = "RA---SIN"
        header["CTYPE2"] = "DEC--SIN"
        with pytest.raises(ValueError, match="FREQ"):
            _find_spectral_axis(header)

    def test_missing_stokes_axis_raises(self):
        header = fits.Header()
        header["NAXIS"] = 2
        header["CTYPE1"] = "RA---SIN"
        header["CTYPE2"] = "DEC--SIN"
        with pytest.raises(ValueError, match="STOKES"):
            _find_stokes_axis(header)


# ---------------------------------------------------------------------------
# combine_fits_to_spectral_cube
# ---------------------------------------------------------------------------


class TestCombineFitsToSpectralCube:
    def test_output_shape(self, fits_files, tmp_path):
        out = tmp_path / "cube.fits"
        hdul = combine_fits_to_spectral_cube(fits_files, out)
        assert hdul[0].data.shape == (3, 64, 64)

    def test_frequencies_sorted(self, fits_files, tmp_path):
        out = tmp_path / "cube.fits"
        hdul = combine_fits_to_spectral_cube(fits_files, out)
        h = hdul[0].header
        assert h["CRVAL3"] == 30e6
        assert h["CDELT3"] == 10e6

    def test_stokes_axis_removed(self, fits_files, tmp_path):
        out = tmp_path / "cube.fits"
        hdul = combine_fits_to_spectral_cube(fits_files, out)
        h = hdul[0].header
        assert h["NAXIS"] == 3
        ctypes = [h[f"CTYPE{i}"] for i in range(1, 4)]
        assert "STOKES" not in ctypes
        assert "FREQ" in ctypes

    def test_spectral_axis_metadata_preserved(self, fits_files, tmp_path):
        out = tmp_path / "cube.fits"
        hdul = combine_fits_to_spectral_cube(fits_files, out)
        h = hdul[0].header
        assert h["CUNIT3"] == "Hz"
        assert h["CTYPE3"] == "FREQ"

    def test_cd_matrix_pixel_scale_preserved(self, fits_files_cd_matrix, tmp_path):
        out = tmp_path / "cube.fits"
        hdul = combine_fits_to_spectral_cube(fits_files_cd_matrix, out)
        w = wcs.WCS(hdul[0].header)
        scales = w.proj_plane_pixel_scales()
        assert abs(scales[0].value - 0.01) < 1e-6
        assert abs(scales[1].value - 0.01) < 1e-6

    def test_metadata_propagated(self, fits_files, tmp_path):
        out = tmp_path / "cube.fits"
        hdul = combine_fits_to_spectral_cube(fits_files, out)
        h = hdul[0].header
        assert h["TELESCOP"] == "OVRO-LWA"
        assert h["BUNIT"] == "Jy/beam"

    def test_file_written_to_disk(self, fits_files, tmp_path):
        out = tmp_path / "cube.fits"
        combine_fits_to_spectral_cube(fits_files, out)
        assert out.exists()
        with fits.open(out) as hdul:
            assert hdul[0].data.shape == (3, 64, 64)

    def test_explicit_freq_values(self, fits_files, tmp_path):
        out = tmp_path / "cube.fits"
        hdul = combine_fits_to_spectral_cube(
            fits_files, out, freq_values=[100e6, 200e6, 300e6],
        )
        h = hdul[0].header
        assert h["CRVAL3"] == 100e6
        assert h["CDELT3"] == 100e6

    def test_too_few_files_raises(self, tmp_path):
        p = _make_lwa_fits(tmp_path / "single.fits", 30e6)
        with pytest.raises(ValueError, match="At least two"):
            combine_fits_to_spectral_cube([p], tmp_path / "out.fits")

    def test_spatial_mismatch_raises(self, tmp_path):
        p1 = _make_lwa_fits(tmp_path / "a.fits", 30e6, nx=64, ny=64)
        p2 = _make_lwa_fits(tmp_path / "b.fits", 40e6, nx=32, ny=32)
        with pytest.raises(ValueError, match="Spatial shape mismatch"):
            combine_fits_to_spectral_cube([p1, p2], tmp_path / "out.fits")

    def test_nonuniform_spacing_warns(self, tmp_path, caplog):
        files = [
            _make_lwa_fits(tmp_path / "f1.fits", 30e6),
            _make_lwa_fits(tmp_path / "f2.fits", 40e6),
            _make_lwa_fits(tmp_path / "f3.fits", 55e6),
        ]
        import logging

        with caplog.at_level(logging.WARNING, logger="lwa_healpix.coadd"):
            combine_fits_to_spectral_cube(files, tmp_path / "out.fits")
        assert "not uniformly spaced" in caplog.text

    def test_dict_input_single_file_per_freq(self, tmp_path):
        f1 = _make_lwa_fits(tmp_path / "a.fits", 30e6)
        f2 = _make_lwa_fits(tmp_path / "b.fits", 40e6)
        groups = {30e6: [f1], 40e6: [f2]}
        hdul = combine_fits_to_spectral_cube(groups, tmp_path / "cube.fits")
        assert hdul[0].data.shape == (2, 64, 64)
        assert hdul[0].header["CRVAL3"] == 30e6

    def test_dict_input_coadds_multiple_files(self, tmp_path):
        f1a = _make_lwa_fits(tmp_path / "a1.fits", 30e6, fill_value=2.0)
        f1b = _make_lwa_fits(tmp_path / "a2.fits", 30e6, fill_value=4.0)
        f2 = _make_lwa_fits(tmp_path / "b.fits", 40e6, fill_value=10.0)
        groups = {30e6: [f1a, f1b], 40e6: [f2]}
        hdul = combine_fits_to_spectral_cube(groups, tmp_path / "cube.fits")
        assert hdul[0].data.shape == (2, 64, 64)
        plane_30 = hdul[0].data[0]
        assert np.allclose(plane_30, 3.0, atol=0.5)

    def test_flat_list_auto_groups_duplicate_freqs(self, tmp_path):
        f1 = _make_lwa_fits(tmp_path / "a.fits", 30e6, fill_value=1.0)
        f2 = _make_lwa_fits(tmp_path / "b.fits", 30e6, fill_value=3.0)
        f3 = _make_lwa_fits(tmp_path / "c.fits", 40e6, fill_value=5.0)
        hdul = combine_fits_to_spectral_cube(
            [f1, f2, f3], tmp_path / "cube.fits",
        )
        assert hdul[0].data.shape == (2, 64, 64)
        assert hdul[0].header["CRVAL3"] == 30e6


# ---------------------------------------------------------------------------
# coadd_fits  (unified reprojection + coadd)
# ---------------------------------------------------------------------------


class TestCoaddFits:
    def test_healpix_target(self, wide_fits_files):
        combined, weights = coadd_fits(
            wide_fits_files, nside=4, coord_frame="galactic",
        )
        npix = 12 * 4**2
        assert combined.shape == (npix,)
        assert np.any(weights > 0)

    def test_image_target(self, fits_files, tmp_path):
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 64
        header["NAXIS2"] = 64
        header["CTYPE1"] = "RA---SIN"
        header["CRPIX1"] = 32.5
        header["CRVAL1"] = 180.0
        header["CDELT1"] = -0.01
        header["CUNIT1"] = "deg"
        header["CTYPE2"] = "DEC--SIN"
        header["CRPIX2"] = 32.5
        header["CRVAL2"] = 34.0
        header["CDELT2"] = 0.01
        header["CUNIT2"] = "deg"
        combined, weights = coadd_fits(
            fits_files, target_header=header,
        )
        assert combined.shape == (64, 64)
        assert np.any(weights > 0)

    def test_must_specify_one_target(self, fits_files):
        with pytest.raises(ValueError, match="Exactly one"):
            coadd_fits(fits_files)

    def test_cannot_specify_both_targets(self, fits_files):
        header = fits.Header()
        header["NAXIS1"] = 64
        header["NAXIS2"] = 64
        with pytest.raises(ValueError, match="Exactly one"):
            coadd_fits(fits_files, nside=4, target_header=header)

    def test_weighted_average_healpix(self, wide_constant_fits_files):
        combined, weights = coadd_fits(
            wide_constant_fits_files, nside=4,
        )
        covered = weights > 0
        assert np.any(covered)
        values = combined[covered]
        assert np.all(values >= 0.9)
        assert np.all(values <= 3.1)


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
