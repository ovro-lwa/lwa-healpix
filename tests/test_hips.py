"""Tests for lwa_healpix.hips."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from conftest import _make_lwa_fits

from lwa_healpix.hips import (
    _car_header_for_nside,
    fits_to_hips,
    fits_to_hips_cube,
    healpix_to_hips,
    upgrade_hips3d,
)
from lwa_healpix.hips_moc import C_LIGHT_M_S, wavelength_range_from_freq


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


# ---------------------------------------------------------------------------
# fits_to_hips_cube (HiPS3D)
# ---------------------------------------------------------------------------


class TestFitsToHipsCube:
    def test_hips3d_output_layout(self, tmp_path):
        files = [
            _make_lwa_fits(tmp_path / "a.fits", 30e6, nx=32, ny=32, pixel_scale=0.5, fill_value=1.0),
            _make_lwa_fits(tmp_path / "b.fits", 40e6, nx=32, ny=32, pixel_scale=0.5, fill_value=2.0),
            _make_lwa_fits(tmp_path / "c.fits", 50e6, nx=32, ny=32, pixel_scale=0.5, fill_value=3.0),
        ]
        out_dir = tmp_path / "hips3d"
        fits_to_hips_cube(
            files,
            out_dir,
            tile_size=16,
            tile_depth=4,
            threads=False,
        )

        props_text = (out_dir / "properties").read_text()
        assert "dataproduct_type     = spectral-cube" in props_text
        assert "hips_order_freq" in props_text
        assert "hips_tile_depth      = 4" in props_text
        assert "hips_tile_width      = 16" in props_text
        assert "hips_initial_freq" in props_text
        assert "hips_version         = 1.4" in props_text
        assert "hips_builder         = astropy/reproject via lwa-healpix" in props_text
        assert "obs_description" in props_text
        assert "not built by Hipsgen" in props_text
        assert "obs_restfreq" in props_text
        assert "obs_regime           = Radio" in props_text
        assert "em_min" in props_text
        assert "em_max" in props_text
        em_min = float(props_text.split("em_min")[1].split("\n")[0].split("=")[1])
        em_max = float(props_text.split("em_max")[1].split("\n")[0].split("=")[1])
        assert em_min < em_max
        assert em_min > 1.0  # tens of MHz band → metres scale

        norder3d = list(out_dir.glob("Norder*_*"))
        assert len(norder3d) >= 1
        assert list(out_dir.rglob("Npix*_*.*")) != []
        assert (out_dir / "Moc.fits").exists()
        assert (out_dir / "index.html").exists()
        index_html = (out_dir / "index.html").read_text()
        assert "HiPS3D" in index_html
        assert "newImageSurvey" in index_html
        assert "3.8.1/aladin.js" in index_html

    def test_default_tile_dimensions(self):
        import inspect

        sig = inspect.signature(fits_to_hips_cube)
        assert sig.parameters["tile_size"].default == 256
        assert sig.parameters["tile_depth"].default == 16

    def test_user_properties_override_initial_freq(self, tmp_path):
        files = [
            _make_lwa_fits(tmp_path / "a.fits", 30e6, nx=32, ny=32, pixel_scale=0.5),
            _make_lwa_fits(tmp_path / "b.fits", 40e6, nx=32, ny=32, pixel_scale=0.5),
        ]
        out_dir = tmp_path / "hips3d_custom"
        fits_to_hips_cube(
            files,
            out_dir,
            tile_size=16,
            tile_depth=4,
            threads=False,
            properties={"hips_initial_freq": "99e6", "obs_title": "Test cube"},
        )
        props = (out_dir / "properties").read_text()
        assert "hips_initial_freq    = 99e6" in props
        assert "obs_title            = Test cube" in props

    def test_user_obs_description_not_overwritten(self, tmp_path):
        files = [
            _make_lwa_fits(tmp_path / "a.fits", 30e6, nx=32, ny=32, pixel_scale=0.5),
            _make_lwa_fits(tmp_path / "b.fits", 40e6, nx=32, ny=32, pixel_scale=0.5),
        ]
        out_dir = tmp_path / "hips3d_desc"
        fits_to_hips_cube(
            files,
            out_dir,
            tile_size=16,
            tile_depth=4,
            threads=False,
            properties={"obs_description": "Custom provenance note"},
        )
        props = (out_dir / "properties").read_text()
        assert "obs_description      = Custom provenance note" in props
        assert "not built by Hipsgen" not in props

    def test_upgrade_hips3d_patches_existing(self, tmp_path):
        files = [
            _make_lwa_fits(tmp_path / "a.fits", 30e6, nx=32, ny=32, pixel_scale=0.5),
            _make_lwa_fits(tmp_path / "b.fits", 50e6, nx=32, ny=32, pixel_scale=0.5),
        ]
        out_dir = tmp_path / "hips3d_old"
        fits_to_hips_cube(
            files, out_dir, tile_size=16, tile_depth=4, threads=False,
        )
        (out_dir / "Moc.fits").unlink()
        props_path = out_dir / "properties"
        old_props = props_path.read_text()
        old_props = old_props.replace(
            "dataproduct_type     = spectral-cube",
            "dataproduct_type     = image",
        )
        props_path.write_text(old_props)
        (out_dir / "index.html").write_text("<html>old 2d viewer</html>")

        upgrade_hips3d(
            out_dir, freq_min_hz=25e6, freq_max_hz=55e6, overwrite=True,
        )

        props = props_path.read_text()
        assert "dataproduct_type     = spectral-cube" in props
        assert "obs_restfreq" in props
        assert (out_dir / "Moc.fits").exists()
        assert "newImageSurvey" in (out_dir / "index.html").read_text()

        em_min, em_max = wavelength_range_from_freq(25e6, 55e6)
        assert f"em_min               = {em_min}" in props or "em_min" in props

    def test_upgrade_hips3d_no_overwrite_skips_existing_files(self, tmp_path):
        files = [
            _make_lwa_fits(tmp_path / "a.fits", 30e6, nx=32, ny=32, pixel_scale=0.5),
            _make_lwa_fits(tmp_path / "b.fits", 50e6, nx=32, ny=32, pixel_scale=0.5),
        ]
        out_dir = tmp_path / "hips3d_partial"
        fits_to_hips_cube(
            files, out_dir, tile_size=16, tile_depth=4, threads=False,
        )
        custom_index = "<html>custom viewer</html>"
        (out_dir / "index.html").write_text(custom_index)
        props_path = out_dir / "properties"
        props_before = props_path.read_text()
        props_path.write_text(
            props_before.replace(
                "obs_regime           = Radio",
                "obs_regime           = Custom",
            ),
        )
        moc_mtime = (out_dir / "Moc.fits").stat().st_mtime

        upgrade_hips3d(out_dir, freq_min_hz=25e6, freq_max_hz=55e6)

        assert (out_dir / "index.html").read_text() == custom_index
        assert (out_dir / "Moc.fits").stat().st_mtime == moc_mtime
        assert "obs_regime           = Custom" in props_path.read_text()
