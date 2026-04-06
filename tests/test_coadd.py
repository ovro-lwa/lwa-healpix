"""Tests for lwa_healpix.coadd."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from astropy import wcs
from astropy.io import fits

from conftest import _make_lwa_fits

from lwa_healpix.coadd import (
    coadd_fits,
    combine_fits_to_spectral_cube,
)


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
# coadd_fits
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

    def test_target_header_may_include_extra_freq_stokes_axes(self, tmp_path):
        """NAXIS=4 target headers must not break reproject (2-D celestial grid)."""
        f1 = _make_lwa_fits(tmp_path / "a.fits", 30e6, fill_value=1.0)
        f2 = _make_lwa_fits(tmp_path / "b.fits", 40e6, fill_value=1.0)
        h4 = fits.Header()
        h4["NAXIS"] = 4
        h4["NAXIS1"] = h4["NAXIS2"] = 64
        h4["NAXIS3"] = h4["NAXIS4"] = 1
        h4["CTYPE1"] = "RA---SIN"
        h4["CTYPE2"] = "DEC--SIN"
        h4["CTYPE3"] = "FREQ"
        h4["CTYPE4"] = "STOKES"
        h4["CRPIX1"] = h4["CRPIX2"] = 32.5
        h4["CRPIX3"] = h4["CRPIX4"] = 1.0
        h4["CRVAL1"] = 180.0
        h4["CRVAL2"] = 34.0
        h4["CRVAL3"] = 30e6
        h4["CRVAL4"] = 1.0
        h4["CDELT1"] = -0.01
        h4["CDELT2"] = 0.01
        h4["CDELT3"] = 1e6
        h4["CDELT4"] = 1.0
        combined, weights = coadd_fits([f1, f2], target_header=h4)
        assert combined.shape == (64, 64)
        assert np.allclose(combined[weights > 0], 1.0, atol=0.01)

    def test_must_specify_one_target(self, fits_files):
        with pytest.raises(ValueError, match="Exactly one"):
            coadd_fits(fits_files)

    def test_cannot_specify_both_targets(self, fits_files):
        header = fits.Header()
        header["NAXIS1"] = 64
        header["NAXIS2"] = 64
        with pytest.raises(ValueError, match="Exactly one"):
            coadd_fits(fits_files, nside=4, target_header=header)

    def test_empty_file_list_raises(self):
        with pytest.raises(ValueError, match="At least one FITS"):
            coadd_fits([], nside=4)

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
# Quality screening (coadd_fits)
# ---------------------------------------------------------------------------


def _image_target_header(nx: int = 64, ny: int = 64) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CTYPE1"] = "RA---SIN"
    header["CRPIX1"] = (nx + 1) / 2.0
    header["CRVAL1"] = 180.0
    header["CDELT1"] = -0.01
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = "DEC--SIN"
    header["CRPIX2"] = (ny + 1) / 2.0
    header["CRVAL2"] = 34.0
    header["CDELT2"] = 0.01
    header["CUNIT2"] = "deg"
    return header


class TestCoaddFitsQuality:
    def test_quality_max_rms_excludes_noisy_file(self, tmp_path):
        quiet = _make_lwa_fits(tmp_path / "q.fits", 30e6, fill_value=1.0)
        noisy = _make_lwa_fits(tmp_path / "n.fits", 30e6, noise_scale=1.0)
        hdr = _image_target_header()
        combined, weights = coadd_fits(
            [quiet, noisy],
            target_header=hdr,
            quality_max_rms=0.05,
        )
        assert np.all(weights > 0)
        assert np.allclose(combined[weights > 0], 1.0, atol=0.05)

    def test_quality_all_rejected_raises(self, tmp_path):
        a = _make_lwa_fits(tmp_path / "a.fits", 30e6, noise_scale=1.0)
        b = _make_lwa_fits(tmp_path / "b.fits", 40e6, noise_scale=1.0)
        hdr = _image_target_header()
        with pytest.raises(ValueError, match="All input images were rejected"):
            coadd_fits(
                [a, b],
                target_header=hdr,
                quality_max_rms=1e-12,
            )

    def test_quality_outlier_sigma_rejects_spike(self, tmp_path):
        files = [
            _make_lwa_fits(tmp_path / f"f{i}.fits", 30e6 + i, fill_value=0.0)
            for i in range(4)
        ]
        files.append(
            _make_lwa_fits(tmp_path / "out.fits", 99e6, noise_scale=80.0),
        )
        hdr = _image_target_header()
        combined, weights = coadd_fits(
            files,
            target_header=hdr,
            quality_outlier_sigma=2.5,
        )
        assert np.all(weights > 0)
        assert np.allclose(combined[weights > 0], 0.0, atol=0.05)


class TestCombineFitsQuality:
    def test_dict_groups_respect_quality_max_rms(self, tmp_path):
        g30 = [
            _make_lwa_fits(tmp_path / "30a.fits", 30e6, fill_value=2.0),
            _make_lwa_fits(tmp_path / "30b.fits", 30e6, noise_scale=1.0),
        ]
        g40 = [
            _make_lwa_fits(tmp_path / "40a.fits", 40e6, fill_value=3.0),
            _make_lwa_fits(tmp_path / "40b.fits", 40e6, noise_scale=1.0),
        ]
        groups = {30e6: g30, 40e6: g40}
        hdul = combine_fits_to_spectral_cube(
            groups,
            tmp_path / "cube.fits",
            quality_max_rms=0.1,
        )
        d = hdul[0].data
        assert np.allclose(d[0], 2.0, atol=0.05)
        assert np.allclose(d[1], 3.0, atol=0.05)

