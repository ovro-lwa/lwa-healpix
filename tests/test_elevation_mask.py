"""Tests for circular / elliptical elevation blanking and parallactic edge check."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from lwa_healpix.coadd import coadd_fits
from lwa_healpix.utils import (
    OVRO_LATITUDE_DEG,
    _elevation_outside_mask,
    _resolve_elevation_cut,
    altaz_to_ha_dec,
    elliptical_mask_edge_zenith_angle_deg,
    parallactic_angle_deg,
    parallactic_delta_q_edge_summary,
)


def test_resolve_elevation_cut_rules() -> None:
    assert _resolve_elevation_cut() is None
    assert _resolve_elevation_cut(min_elevation=10.0) == ("circular", 10.0)
    assert _resolve_elevation_cut(min_elevation_ns=15.0, min_elevation_ew=40.0) == (
        "ellipse",
        15.0,
        40.0,
    )
    with pytest.raises(ValueError, match="together"):
        _resolve_elevation_cut(min_elevation_ns=15.0)
    with pytest.raises(ValueError, match="not both"):
        _resolve_elevation_cut(min_elevation=10.0, min_elevation_ns=15.0, min_elevation_ew=40.0)


def test_elliptical_edge_matches_ns_ew_axes() -> None:
    z_n = elliptical_mask_edge_zenith_angle_deg(
        0.0, min_elevation_ns=15.0, min_elevation_ew=40.0
    )
    z_e = elliptical_mask_edge_zenith_angle_deg(
        90.0, min_elevation_ns=15.0, min_elevation_ew=40.0
    )
    assert z_n == pytest.approx(75.0)
    assert z_e == pytest.approx(50.0)
    # Circular when equal thresholds
    z_c = elliptical_mask_edge_zenith_angle_deg(
        45.0, min_elevation_ns=20.0, min_elevation_ew=20.0
    )
    assert z_c == pytest.approx(70.0)


def test_elevation_outside_mask_axes() -> None:
    # Just inside / outside N and E edges
    z = np.array([74.0, 76.0, 49.0, 51.0])
    pa = np.deg2rad(np.array([0.0, 0.0, 90.0, 90.0]))
    out = _elevation_outside_mask(
        z, pa, min_elevation_ns=15.0, min_elevation_ew=40.0
    )
    assert list(out) == [False, True, False, True]

    # Circular equivalence
    z2 = np.array([69.0, 71.0])
    pa2 = np.deg2rad(np.array([0.0, 90.0]))
    out_ell = _elevation_outside_mask(
        z2, pa2, min_elevation_ns=20.0, min_elevation_ew=20.0
    )
    out_circ = _elevation_outside_mask(z2, pa2, min_elevation=20.0)
    assert np.array_equal(out_ell, out_circ)


def _zenith_sin_fits(path, *, size: int = 61, cdelt_deg: float = 2.0) -> None:
    """Write a flat image with CRVAL at zenith Dec=0 for simple PA geometry."""
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = size
    hdr["NAXIS2"] = size
    hdr["CTYPE1"] = "RA---SIN"
    hdr["CTYPE2"] = "DEC--SIN"
    hdr["CRVAL1"] = 0.0
    hdr["CRVAL2"] = 0.0  # zenith at equator for test geometry
    hdr["CRPIX1"] = (size + 1) / 2.0
    hdr["CRPIX2"] = (size + 1) / 2.0
    hdr["CDELT1"] = -cdelt_deg
    hdr["CDELT2"] = cdelt_deg
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    data = np.ones((size, size), dtype=np.float32)
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def test_coadd_elliptical_blanks_ew_sooner(tmp_path) -> None:
    path = tmp_path / "zenith.fits"
    _zenith_sin_fits(path)
    combined, weight = coadd_fits(
        [path],
        nside=16,
        coord_frame="icrs",
        nested=True,
        min_elevation_ns=15.0,
        min_elevation_ew=40.0,
    )
    assert combined.shape == (12 * 16**2,)
    assert (weight > 0).any()
    # With only elevation masking before reproject, finite footprint should exist
    assert np.isfinite(combined[weight > 0]).all()


def test_parallactic_delta_q_edge_summary_shape() -> None:
    rows = parallactic_delta_q_edge_summary(
        min_elevation_ns=15.0, min_elevation_ew=40.0
    )
    assert len(rows) == 8
    assert [r["direction"] for r in rows] == [
        "N", "NE", "E", "SE", "S", "SW", "W", "NW",
    ]
    # E/W edges are higher elevation → smaller zenith angle than N/S
    by_dir = {r["direction"]: r for r in rows}
    assert by_dir["E"]["elev_edge_deg"] == pytest.approx(40.0)
    assert by_dir["N"]["elev_edge_deg"] == pytest.approx(15.0)
    assert by_dir["E"]["zenith_angle_deg"] < by_dir["N"]["zenith_angle_deg"]
    # Dec at E vs N edges should differ under an elliptical mask
    assert by_dir["E"]["dec_deg"] != pytest.approx(by_dir["N"]["dec_deg"], abs=0.5)
    for r in rows:
        assert r["dq_range_deg"] >= 0.0
        assert r["abs_dq_max_deg"] >= r["abs_dq_median_deg"] >= r["abs_dq_min_deg"]


def test_parallactic_circular_mask_consistent_dec_distance() -> None:
    """Equal ns/ew → same zenith distance on all axes; Dec depends on az via altaz."""
    rows = parallactic_delta_q_edge_summary(
        min_elevation_ns=30.0, min_elevation_ew=30.0
    )
    zs = [float(r["zenith_angle_deg"]) for r in rows]
    assert max(zs) - min(zs) < 1e-6
    # Spot-check: altaz→HA/Dec then q is finite
    ha, dec = altaz_to_ha_dec(60.0, 90.0, latitude_deg=OVRO_LATITUDE_DEG)
    q = parallactic_angle_deg(ha, dec, latitude_deg=OVRO_LATITUDE_DEG)
    assert np.isfinite(q)
