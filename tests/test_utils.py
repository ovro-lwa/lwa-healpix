"""Tests for lwa_healpix.utils."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from conftest import _make_lwa_fits

from lwa_healpix.utils import (
    _extract_2d,
    _find_spectral_axis,
    _find_stokes_axis,
    group_pipeline_files,
)


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

    def test_find_spectral_axis_2d_vestigial_crval3(self):
        """NAXIS=2 images may keep CRVAL3 without CTYPE3/NAXIS3."""
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = 16
        header["NAXIS2"] = 16
        header["CTYPE1"] = "RA---SIN"
        header["CTYPE2"] = "DEC--SIN"
        header["CRVAL3"] = 66e6
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


class TestExtract2d:
    def test_pure_2d_with_vestigial_crval3(self, tmp_path):
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = 8
        hdr["NAXIS2"] = 8
        hdr["CTYPE1"] = "RA---SIN"
        hdr["CRPIX1"] = 4.5
        hdr["CRVAL1"] = 180.0
        hdr["CDELT1"] = -0.01
        hdr["CTYPE2"] = "DEC--SIN"
        hdr["CRPIX2"] = 4.5
        hdr["CRVAL2"] = 34.0
        hdr["CDELT2"] = 0.01
        hdr["CRVAL3"] = 41e6
        data = np.zeros((8, 8), dtype=np.float32)
        path = tmp_path / "flat2d.fits"
        fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
        with fits.open(path) as hdul:
            plane, w2 = _extract_2d(hdul[0])
        assert plane.shape == (8, 8)
        assert w2.naxis == 2


# ---------------------------------------------------------------------------
# group_pipeline_files
# ---------------------------------------------------------------------------


class TestGroupPipelineFiles:
    BASE = "/lustre/pipeline/images"

    def _pipeline_path(self, lst: str, freq_mhz: int, date: str = "2024-12-18") -> str:
        return (
            f"{self.BASE}/{lst}/{date}/Run_20260227_014418/{freq_mhz}MHz/I/deep/"
            f"{freq_mhz}MHz-I-Deep-Taper-Robust-0-image-{date.replace('-', '')}_123232.pbcorr.fits"
        )

    def test_groups_by_frequency(self):
        paths = [
            self._pipeline_path("10h", 41),
            self._pipeline_path("10h", 55),
            self._pipeline_path("10h", 73),
        ]
        groups = group_pipeline_files(paths)
        assert list(groups.keys()) == [41e6, 55e6, 73e6]
        assert len(groups[41e6]) == 1

    def test_multiple_lst_same_freq(self):
        paths = [
            self._pipeline_path("10h", 41),
            self._pipeline_path("14h", 41),
            self._pipeline_path("18h", 41),
            self._pipeline_path("10h", 55),
        ]
        groups = group_pipeline_files(paths)
        assert len(groups[41e6]) == 3
        assert len(groups[55e6]) == 1

    def test_sorted_by_frequency(self):
        paths = [
            self._pipeline_path("10h", 73),
            self._pipeline_path("10h", 41),
            self._pipeline_path("10h", 55),
        ]
        groups = group_pipeline_files(paths)
        assert list(groups.keys()) == [41e6, 55e6, 73e6]

    def test_missing_freq_raises(self):
        with pytest.raises(ValueError, match="Cannot determine frequency"):
            group_pipeline_files(["/data/images/no_freq_here/image.fits"])

    def test_returns_path_objects(self):
        paths = [
            self._pipeline_path("10h", 41),
            self._pipeline_path("14h", 55),
        ]
        groups = group_pipeline_files(paths)
        for file_list in groups.values():
            for p in file_list:
                assert isinstance(p, Path)

    def test_header_fallback(self, tmp_path):
        """Files without {freq}MHz in the path use the FITS header."""
        paths = [
            _make_lwa_fits(tmp_path / "a.fits", 41e6),
            _make_lwa_fits(tmp_path / "b.fits", 55e6),
            _make_lwa_fits(tmp_path / "c.fits", 41e6),
        ]
        groups = group_pipeline_files(paths)
        assert list(groups.keys()) == [41e6, 55e6]
        assert len(groups[41e6]) == 2
        assert len(groups[55e6]) == 1

    def test_header_fallback_2d_with_crval3(self, tmp_path):
        """2D FITS files with a bare CRVAL3 (no CTYPE3/NAXIS3)."""
        for name, freq_hz in [("x.fits", 41e6), ("y.fits", 66e6)]:
            hdr = fits.Header()
            hdr["NAXIS"] = 2
            hdr["NAXIS1"] = 16
            hdr["NAXIS2"] = 16
            hdr["CTYPE1"] = "RA---SIN"
            hdr["CRPIX1"] = 8.5
            hdr["CRVAL1"] = 180.0
            hdr["CDELT1"] = -0.01
            hdr["CTYPE2"] = "DEC--SIN"
            hdr["CRPIX2"] = 8.5
            hdr["CRVAL2"] = 34.0
            hdr["CDELT2"] = 0.01
            hdr["CRVAL3"] = freq_hz
            data = np.zeros((16, 16), dtype=np.float32)
            fits.PrimaryHDU(data=data, header=hdr).writeto(
                tmp_path / name, overwrite=True,
            )

        groups = group_pipeline_files(
            [tmp_path / "x.fits", tmp_path / "y.fits"],
        )
        assert list(groups.keys()) == [41e6, 66e6]
