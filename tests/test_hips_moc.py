"""Tests for HiPS3D MOC helpers."""

from __future__ import annotations

from astropy.io import fits

from lwa_healpix.hips_moc import (
    C_LIGHT_M_S,
    freq_range_from_cube_header,
    wavelength_range_from_freq,
)


class TestFreqRangeFromCubeHeader:
    def test_uniform_cube(self):
        header = fits.Header()
        header["NAXIS3"] = 3
        header["CRVAL3"] = 30e6
        header["CDELT3"] = 10e6
        fmin, fmax, initial = freq_range_from_cube_header(header)
        assert fmin == 30e6
        assert fmax == 60e6
        assert initial == 40e6

    def test_single_channel(self):
        header = fits.Header()
        header["NAXIS3"] = 1
        header["CRVAL3"] = 41e6
        fmin, fmax, initial = freq_range_from_cube_header(header)
        assert fmin == fmax == initial == 41e6

    def test_wavelength_range_from_freq(self):
        em_min, em_max = wavelength_range_from_freq(30e6, 50e6)
        assert em_min == C_LIGHT_M_S / 50e6
        assert em_max == C_LIGHT_M_S / 30e6
        assert em_min < em_max
