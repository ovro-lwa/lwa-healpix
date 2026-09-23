"""Tests for HEALPix FITS I/O."""

from __future__ import annotations

import numpy as np
import pytest

from lwa_healpix.healpix_io import read_healpix_fits, write_healpix_fits


def test_write_read_roundtrip(tmp_path):
    nside = 8
    npix = 12 * nside**2
    healpix_map = np.arange(npix, dtype=np.float32)
    weight = np.linspace(0.0, 1.0, npix, dtype=np.float32)
    path = tmp_path / "sky.fits"
    write_healpix_fits(
        path,
        healpix_map,
        weight,
        nside=nside,
        nested=True,
        coord_frame="equatorial",
    )
    m2, w2, meta = read_healpix_fits(path)
    assert meta["nside"] == nside
    assert meta["nested"] is True
    assert meta["coord_frame"] == "equatorial"
    np.testing.assert_array_equal(m2, healpix_map)
    np.testing.assert_allclose(w2, weight)


def test_galactic_coordsys(tmp_path):
    nside = 4
    npix = 12 * nside**2
    path = tmp_path / "g.fits"
    write_healpix_fits(
        path,
        np.ones(npix, dtype=np.float32),
        np.ones(npix, dtype=np.float32),
        nside=nside,
        nested=False,
        coord_frame="galactic",
    )
    _, _, meta = read_healpix_fits(path)
    assert meta["nested"] is False
    assert meta["coord_frame"] == "galactic"


def test_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="length"):
        write_healpix_fits(
            tmp_path / "bad.fits",
            np.ones(10),
            np.ones(10),
            nside=4,
        )
