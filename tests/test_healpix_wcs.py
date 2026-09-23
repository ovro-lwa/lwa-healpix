"""Tests for lwa_healpix.healpix_wcs."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from lwa_healpix.healpix_wcs import (
    healpix_to_hdu,
    iter_nested_tile_headers,
    nested_tile_header,
    pixel_scale_deg_for_nside,
    reproject_healpix_to_wcs,
)


def _tan_header(
    *,
    crval1: float = 180.0,
    crval2: float = 30.0,
    naxis: int = 32,
    cdelt: float = 0.5,
) -> fits.Header:
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = naxis
    hdr["NAXIS2"] = naxis
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = crval1
    hdr["CRVAL2"] = crval2
    hdr["CRPIX1"] = (naxis + 1) / 2.0
    hdr["CRPIX2"] = (naxis + 1) / 2.0
    hdr["CDELT1"] = -cdelt
    hdr["CDELT2"] = cdelt
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    return hdr


def _sin_header(**kwargs) -> fits.Header:
    hdr = _tan_header(**kwargs)
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    return hdr


class TestReprojectHealpixToWcs:
    def test_constant_map_tan_finite_where_footprint(self):
        nside = 16
        healpix_map = np.full(12 * nside**2, 3.5, dtype=np.float64)
        hdr = _tan_header()
        data, footprint = reproject_healpix_to_wcs(
            healpix_map, hdr, coord_frame="icrs", nested=False,
        )
        assert data.shape == (32, 32)
        assert footprint.shape == (32, 32)
        covered = footprint > 0
        assert covered.any()
        assert np.allclose(data[covered], 3.5, rtol=1e-5, atol=1e-5)

    def test_sin_patch(self):
        nside = 16
        healpix_map = np.ones(12 * nside**2, dtype=np.float64)
        data, footprint = reproject_healpix_to_wcs(
            healpix_map,
            _sin_header(crval2=45.0),
            coord_frame="icrs",
        )
        assert data.shape == (32, 32)
        assert (footprint > 0).any()

    def test_nested_true(self):
        nside = 8
        healpix_map = np.arange(12 * nside**2, dtype=np.float64)
        data, footprint = reproject_healpix_to_wcs(
            healpix_map,
            _tan_header(naxis=16, cdelt=1.0),
            coord_frame="icrs",
            nested=True,
        )
        assert data.shape == (16, 16)
        assert (footprint > 0).sum() > 0

    def test_rejects_2d_map(self):
        hdr = _tan_header()
        with pytest.raises(ValueError, match="1-D"):
            reproject_healpix_to_wcs(np.ones((4, 4)), hdr)

    def test_rejects_header_without_naxis(self):
        nside = 8
        healpix_map = np.ones(12 * nside**2)
        hdr = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN"})
        with pytest.raises(ValueError, match="NAXIS1"):
            reproject_healpix_to_wcs(healpix_map, hdr, coord_frame="icrs")


class TestNestedTileHeader:
    def test_crval_matches_pix2ang(self):
        nside_tile = 4
        ipix = 17
        hdr = nested_tile_header(
            nside_tile, ipix, nside_map=16, overlap=0.2, ctype="TAN",
        )
        theta, phi = hp.pix2ang(nside_tile, ipix, nest=True)
        assert hdr["CRVAL1"] == pytest.approx(np.degrees(phi))
        assert hdr["CRVAL2"] == pytest.approx(90.0 - np.degrees(theta))
        assert hdr["CTYPE1"] == "RA---TAN"
        assert abs(hdr["CDELT1"]) == pytest.approx(pixel_scale_deg_for_nside(16))

    def test_naxis_from_fov(self):
        hdr = nested_tile_header(4, 0, nside_map=32, overlap=0.0)
        tile_scale = pixel_scale_deg_for_nside(4)
        map_scale = pixel_scale_deg_for_nside(32)
        expected = int(np.ceil(tile_scale / map_scale))
        assert hdr["NAXIS1"] == expected
        assert hdr["NAXIS2"] == expected

    def test_invalid_ipix(self):
        with pytest.raises(ValueError, match="ipix"):
            nested_tile_header(4, 9999, nside_map=16)

    def test_nside_not_divisible(self):
        with pytest.raises(ValueError, match="divisible"):
            nested_tile_header(16, 0, nside_map=8)

    def test_nside_not_power_of_two(self):
        with pytest.raises(ValueError, match="power of 2"):
            nested_tile_header(3, 0, nside_map=12)


class TestHealpixToHdu:
    def test_all_nan_when_weight_zero(self):
        nside = 16
        npix = 12 * nside**2
        hdr = nested_tile_header(4, 0, nside_map=nside, overlap=0.2)
        hdu = healpix_to_hdu(
            np.ones(npix),
            hdr,
            weight=np.zeros(npix),
            coord_frame="icrs",
            nested=True,
        )
        assert hdu.data.dtype == np.float32
        assert np.isnan(hdu.data).all()
        assert "BMAJ" not in hdu.header

    def test_finite_when_weight_positive(self):
        nside = 16
        npix = 12 * nside**2
        hdr = nested_tile_header(4, 0, nside_map=nside, overlap=0.2)
        hdu = healpix_to_hdu(
            np.full(npix, 2.5),
            hdr,
            weight=np.ones(npix),
            coord_frame="icrs",
            nested=True,
        )
        assert np.isfinite(hdu.data).any()
        assert np.nanmax(hdu.data) == pytest.approx(2.5, rel=1e-4)

    def test_header_updates(self):
        nside = 8
        hdr = nested_tile_header(2, 0, nside_map=nside)
        hdu = healpix_to_hdu(
            np.ones(12 * nside**2),
            hdr,
            nested=True,
            header_updates={"BUNIT": "Jy/beam"},
        )
        assert hdu.header["BUNIT"] == "Jy/beam"
        wcs = WCS(hdu.header)
        sky = wcs.pixel_to_world(hdu.header["CRPIX1"] - 1, hdu.header["CRPIX2"] - 1)
        assert sky.ra.deg == pytest.approx(hdu.header["CRVAL1"], abs=0.5)
        assert sky.dec.deg == pytest.approx(hdu.header["CRVAL2"], abs=0.5)


class TestIterNestedTileHeaders:
    def test_count_all_tiles(self):
        headers = list(iter_nested_tile_headers(4, 16, overlap=0.0))
        assert len(headers) == 12 * 4**2

    def test_skip_empty_weight(self):
        nside_map = 8
        nside_tile = 2
        weight = np.zeros(12 * nside_map**2, dtype=np.float64)
        ratio = (nside_map // nside_tile) ** 2
        weight[0:ratio] = 1.0
        tiles = list(
            iter_nested_tile_headers(
                nside_tile, nside_map, weight=weight, min_weight_sum=0.0,
            )
        )
        assert len(tiles) == 1
        assert tiles[0][0] == 0
