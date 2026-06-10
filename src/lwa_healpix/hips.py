"""Generate HiPS (2-D) and HiPS3D (spectral cube) tile sets."""

from __future__ import annotations

import logging
from typing import Literal

import shutil
from importlib import resources
from pathlib import Path

import numpy as np
from astropy.io import fits
from reproject import reproject_from_healpix, reproject_interp
from reproject.hips import reproject_to_hips
from reproject.hips.utils import load_properties, save_properties

from .coadd import combine_fits_to_spectral_cube
from .hips_moc import (
    C_LIGHT_M_S,
    coverage_freq_range_hz,
    freq_range_from_cube_header,
    wavelength_range_from_freq,
    write_hips3d_moc,
)

# Honest provenance strings for HiPS3D ``properties`` (not Hipsgen output).
_HIPS3D_BUILDER = "astropy/reproject via lwa-healpix"
logger = logging.getLogger(__name__)

_HIPS3D_DESCRIPTION = (
    "HiPS3D spectral-cube tiles produced by astropy.reproject (hips_version "
    "1.4).  Spectral tile indexing uses the CDS FMOC discretization described "
    "in HiPS3D 1.5-proto2, but this dataset was not built by Hipsgen.  "
    "Moc.fits is an approximate space-frequency coverage derived from tile "
    "footprints.  FITS tiles only (no PNG/checkerboard preview layer)."
)

__all__ = [
    "fits_to_hips",
    "fits_to_hips_cube",
    "healpix_to_hips",
    "upgrade_hips3d",
]

DEFAULT_CAR_HEADER = fits.Header.fromstring(
    """
NAXIS   =                    2
NAXIS1  =                 3600
NAXIS2  =                 1800
CTYPE1  = 'GLON-CAR'
CRPIX1  =               1800.5
CRVAL1  =                180.0
CDELT1  =                 -0.1
CUNIT1  = 'deg'
CTYPE2  = 'GLAT-CAR'
CRPIX2  =                900.5
CRVAL2  =                  0.0
CDELT2  =                  0.1
CUNIT2  = 'deg'
""",
    sep="\n",
)


def _car_header_for_nside(
    nside: int,
    coord_frame: str = "galactic",
) -> fits.Header:
    """Build a full-sky CAR header whose pixel scale matches *nside*."""
    pixel_scale = np.degrees(np.sqrt(4 * np.pi / (12 * nside**2)))
    nx = int(np.ceil(360.0 / pixel_scale))
    ny = int(np.ceil(180.0 / pixel_scale))
    cdelt = 360.0 / nx

    if coord_frame == "galactic":
        ctype1, ctype2 = "GLON-CAR", "GLAT-CAR"
    else:
        ctype1, ctype2 = "RA---CAR", "DEC--CAR"

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CTYPE1"] = ctype1
    header["CRPIX1"] = (nx + 1) / 2.0
    header["CRVAL1"] = 180.0 if coord_frame == "galactic" else 0.0
    header["CDELT1"] = -cdelt
    header["CUNIT1"] = "deg"
    header["CTYPE2"] = ctype2
    header["CRPIX2"] = (ny + 1) / 2.0
    header["CRVAL2"] = 0.0
    header["CDELT2"] = cdelt
    header["CUNIT2"] = "deg"
    return header


def _reproject_healpix_to_car(
    healpix_map: np.ndarray,
    coord_frame: str = "galactic",
    target_header: fits.Header | None = None,
    nested: bool = False,
) -> tuple[np.ndarray, fits.Header]:
    """Reproject a HEALPix map onto a Plate Carree (CAR) grid."""
    if target_header is None:
        nside = int(np.sqrt(len(healpix_map) / 12))
        target_header = _car_header_for_nside(nside, coord_frame)

    flat_array, _ = reproject_from_healpix(
        (healpix_map, coord_frame), target_header, nested=nested
    )
    return flat_array, target_header


def _copy_index_html(
    output_directory: Path,
    *,
    template: str = "index.html",
) -> None:
    """Copy a bundled Aladin Lite viewer into *output_directory*."""
    index_src = resources.files("lwa_healpix") / "data" / template
    shutil.copy2(index_src, output_directory / "index.html")


_MAX_SANE_PIXEL_CUT_ABS = 1e10


def _sanitize_hips_pixel_cut(props: dict[str, str | float]) -> None:
    """Drop ``hips_pixel_cut`` when reproject wrote nonsensical FITS limits."""
    raw = props.get("hips_pixel_cut")
    if raw is None:
        return
    try:
        parts = str(raw).split()
        if len(parts) != 2:
            raise ValueError("expected two numbers")
        lo, hi = float(parts[0]), float(parts[1])
    except (TypeError, ValueError):
        logger.warning("Removing invalid hips_pixel_cut: %r", raw)
        props.pop("hips_pixel_cut", None)
        return
    if (
        lo >= hi
        or abs(lo) > _MAX_SANE_PIXEL_CUT_ABS
        or abs(hi) > _MAX_SANE_PIXEL_CUT_ABS
    ):
        logger.warning("Removing nonsensical hips_pixel_cut: %s %s", lo, hi)
        props.pop("hips_pixel_cut", None)


def _finalize_hips3d_properties(
    output_directory: Path,
    *,
    freq_min_hz: float,
    freq_max_hz: float,
    initial_freq_hz: float,
    user_properties: dict[str, str] | None,
    overwrite: bool = True,
) -> None:
    """Add honest HiPS3D metadata without misrepresenting the generator."""
    props = load_properties(str(output_directory))
    user = user_properties or {}

    def _set(key: str, value) -> None:
        if key in user:
            return
        if overwrite or key not in props:
            props[key] = value

    _set("dataproduct_type", "spectral-cube")
    _set("hips_builder", _HIPS3D_BUILDER)
    _set("obs_description", _HIPS3D_DESCRIPTION)
    _set("hips_initial_freq", initial_freq_hz)
    _set("obs_restfreq", initial_freq_hz)
    _set("obs_regime", "Radio")

    need_em = (
        ("em_min" not in user and (overwrite or "em_min" not in props))
        or ("em_max" not in user and (overwrite or "em_max" not in props))
    )
    if need_em:
        try:
            eff_fmin, eff_fmax = coverage_freq_range_hz(
                output_directory, freq_min_hz, freq_max_hz,
            )
        except (ValueError, OSError):
            eff_fmin, eff_fmax = freq_min_hz, freq_max_hz
        em_min, em_max = wavelength_range_from_freq(eff_fmin, eff_fmax)
        if "em_min" not in user and (overwrite or "em_min" not in props):
            props["em_min"] = em_min
        if "em_max" not in user and (overwrite or "em_max" not in props):
            props["em_max"] = em_max

    if "hips_pixel_cut" not in user:
        _sanitize_hips_pixel_cut(props)

    props.update(user)
    save_properties(str(output_directory), props)


def upgrade_hips3d(
    output_directory: str | Path,
    *,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    initial_freq_hz: float | None = None,
    properties: dict[str, str] | None = None,
    overwrite: bool = False,
) -> None:
    """Patch an existing HiPS3D directory for Aladin Lite compatibility.

    Updates ``properties`` (``dataproduct_type``, ``em_min``/``em_max``,
    ``obs_restfreq``, …), regenerates ``Moc.fits``, and copies the HiPS3D
    ``index.html`` viewer.  Does not regenerate tiles.

    Parameters
    ----------
    output_directory : str or Path
        Root of an existing HiPS3D tile set.
    freq_min_hz, freq_max_hz : float or None, optional
        Cube frequency limits in hertz.  If omitted, inferred from
        ``em_min``/``em_max`` in ``properties`` when present.
    initial_freq_hz : float or None, optional
        Centre frequency for ``hips_initial_freq`` / ``obs_restfreq``.
        Defaults to the midpoint of the frequency range.
    properties : dict or None, optional
        Extra ``properties`` entries (same as :func:`fits_to_hips_cube`).
        These keys are always written, even when *overwrite* is ``False``.
    overwrite : bool, optional
        If ``True``, replace existing ``properties`` keys, ``Moc.fits``, and
        ``index.html``.  If ``False`` (default), only fill missing metadata
        and skip ``Moc.fits`` / ``index.html`` when they already exist.
    """
    output_directory = Path(output_directory)
    if not (output_directory / "properties").exists():
        msg = f"No HiPS properties file in {output_directory}"
        raise ValueError(msg)

    props = load_properties(str(output_directory))
    if props.get("dataproduct_type") != "spectral-cube":
        logger.warning(
            "Expected dataproduct_type=spectral-cube, got %r",
            props.get("dataproduct_type"),
        )

    if freq_min_hz is None or freq_max_hz is None:
        if "em_min" in props and "em_max" in props:
            em_min = float(props["em_min"])
            em_max = float(props["em_max"])
            inferred_max = C_LIGHT_M_S / em_min
            inferred_min = C_LIGHT_M_S / em_max
            freq_min_hz = inferred_min if freq_min_hz is None else freq_min_hz
            freq_max_hz = inferred_max if freq_max_hz is None else freq_max_hz
        else:
            msg = (
                "freq_min_hz and freq_max_hz are required when em_min/em_max "
                "are not in properties"
            )
            raise ValueError(msg)

    if initial_freq_hz is None:
        if "hips_initial_freq" in props:
            initial_freq_hz = float(props["hips_initial_freq"])
        else:
            initial_freq_hz = 0.5 * (freq_min_hz + freq_max_hz)

    _finalize_hips3d_properties(
        output_directory,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
        initial_freq_hz=initial_freq_hz,
        user_properties=properties,
        overwrite=overwrite,
    )
    moc_path = output_directory / "Moc.fits"
    if overwrite or not moc_path.exists():
        write_hips3d_moc(
            output_directory,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            overwrite=overwrite,
        )
    index_path = output_directory / "index.html"
    if overwrite or not index_path.exists():
        _copy_index_html(output_directory, template="index_cube.html")


def healpix_to_hips(
    healpix_map: np.ndarray,
    coord_frame: str = "galactic",
    output_directory: str | Path = "hips_output",
    nested: bool = False,
    target_header: fits.Header | None = None,
    threads: bool = True,
) -> None:
    """Reproject a HEALPix map to a CAR grid and generate HiPS tiles.

    Parameters
    ----------
    healpix_map : numpy.ndarray
        1-D HEALPix map array.
    coord_frame : str, optional
        Coordinate frame of the input map (e.g. ``"galactic"``).
    output_directory : str or Path, optional
        Directory to write HiPS tiles into. Default is ``"hips_output"``.
    nested : bool, optional
        If ``True``, the input uses NESTED pixel ordering. Default is ``False``.
    target_header : `~astropy.io.fits.Header`, optional
        WCS header for the intermediate CAR grid. If *None*, a full-sky
        CAR grid is generated automatically with a pixel scale matching
        the HEALPix NSIDE (derived from the length of *healpix_map*).
    threads : bool, optional
        Whether to use multi-threaded reprojection. Default is ``True``.
    """
    flat_array, header = _reproject_healpix_to_car(
        healpix_map, coord_frame=coord_frame,
        target_header=target_header, nested=nested,
    )

    output_directory = Path(output_directory)

    reproject_to_hips(
        (flat_array, header),
        output_directory=str(output_directory),
        coord_system_out=coord_frame,
        reproject_function=reproject_interp,
        threads=threads,
    )

    _copy_index_html(output_directory)


def fits_to_hips(
    input_data,
    output_directory: str | Path = "hips_output",
    *,
    coord_frame: str = "galactic",
    tile_size: int = 512,
    level: int | None = None,
    threads: bool = True,
    properties: dict[str, str] | None = None,
) -> None:
    """Generate 2-D HiPS tiles from a FITS image.

    Parameters
    ----------
    input_data
        Any input accepted by
        :func:`reproject.hips.reproject_to_hips`: a FITS file path, an
        ``(array, header)`` tuple, an HDU, or an
        `~astropy.nddata.NDData` object.
    output_directory : str or Path, optional
        Directory to write HiPS tiles into.  Default is
        ``"hips_output"``.
    coord_frame : str, optional
        Coordinate system for the HiPS output (``"galactic"``,
        ``"equatorial"``, or ``"ecliptic"``).  Default is ``"galactic"``.
    tile_size : int, optional
        Spatial tile size in pixels.  Default is 512.
    level : int or None, optional
        Maximum spatial HiPS order.  If *None*, ``reproject`` chooses
        automatically based on the input resolution.
    threads : bool, optional
        Enable multi-threaded tile generation.  Default is ``True``.
    properties : dict or None, optional
        Extra key/value pairs to write into the HiPS ``properties``
        file (e.g. ``obs_title``, ``creator_did``).
    """
    output_directory = Path(output_directory)

    reproject_to_hips(
        input_data,
        output_directory=str(output_directory),
        coord_system_out=coord_frame,
        reproject_function=reproject_interp,
        tile_size=tile_size,
        level=level,
        threads=threads,
        properties=properties,
    )

    _copy_index_html(output_directory)


def fits_to_hips_cube(
    file_paths: list[str | Path] | dict[float, list[str | Path]],
    output_directory: str | Path = "hips_cube_output",
    *,
    coord_frame: str = "galactic",
    freq_values: list[float] | np.ndarray | None = None,
    min_elevation: float | None = None,
    quality_max_rms: float | None = None,
    quality_outlier_sigma: float | None = None,
    quality_metric: Literal["std", "mad_sigma"] = "std",
    quality_center_fraction: float = 0.25,
    quality_center_max_pixels: int | None = 512,
    tile_size: int = 256,
    tile_depth: int = 16,
    level: int | None = None,
    level_depth: int | None = None,
    threads: bool = True,
    properties: dict[str, str] | None = None,
) -> None:
    """Build a HiPS cube from single-frequency FITS images.

    This is a convenience wrapper that calls
    :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube` to
    assemble a 3-D spectral cube and then passes it to
    :func:`reproject.hips.reproject_to_hips` to generate a HiPS3D tile
    set intended for Aladin Lite v3.8+ HiPS3D clients.

    Tiles are generated by ``astropy.reproject`` (``hips_version`` 1.4,
    ``hips_builder`` records ``astropy/reproject via lwa-healpix``).
    This is not Hipsgen ``1.5-proto2`` output; ``obs_description`` in
    the ``properties`` file states the actual provenance.  Tile defaults
    follow CDS HiPS3D recommendations (``tile_size=256``,
    ``tile_depth=16``).  An approximate ``Moc.fits``, ``hips_initial_freq``,
    and a HiPS3D-aware ``index.html`` viewer are also written.

    Each input file is expected to contain a 4-D FITS image with axes
    ``(RA, Dec, Freq, Stokes)`` where frequency and Stokes are both
    length 1.  See
    :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube` for
    details on the two accepted forms of *file_paths*.

    Parameters
    ----------
    file_paths : list or dict
        Paths to the FITS files.  Accepts the same forms as
        :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube`: a
        flat list (one file per channel, auto-groups duplicates) or a
        ``dict[float, list[path]]`` mapping frequencies to files.
    output_directory : str or Path, optional
        Directory to write HiPS tiles into.  Must not already exist.
        Default is ``"hips_cube_output"``.
    coord_frame : str, optional
        Coordinate system for the HiPS output (``"galactic"``,
        ``"equatorial"``, or ``"ecliptic"``).  Default is ``"galactic"``.
    freq_values : array-like or None, optional
        Explicit frequency values for each file (same order as
        *file_paths*).  Passed through to
        :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube`.
    min_elevation : float or None, optional
        Minimum elevation in degrees.  Passed through to
        :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube` for
        per-channel coadding.
    quality_max_rms : float or None, optional
        Passed to :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube`.
    quality_outlier_sigma : float or None, optional
        Passed to :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube`.
    quality_metric : {"std", "mad_sigma"}, optional
        Passed to :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube`.
    quality_center_fraction : float, optional
        Passed to :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube`.
    quality_center_max_pixels : int or None, optional
        Passed to :func:`~lwa_healpix.coadd.combine_fits_to_spectral_cube`.
    tile_size : int, optional
        Spatial tile size in pixels (default ``256``, per CDS HiPS3D).
    tile_depth : int, optional
        Depth of each tile along the spectral axis.  Must be a power of
        two and at least ``2``.  Default is ``16``.
    level : int or None, optional
        Maximum spatial HiPS order.  If *None*, ``reproject`` chooses
        automatically based on the input resolution.
    level_depth : int or None, optional
        Maximum spectral HiPS order.  If *None*, ``reproject`` chooses
        automatically.
    threads : bool, optional
        Enable multi-threaded tile generation.  Default is ``True``.
    properties : dict or None, optional
        Extra key/value pairs to write into the HiPS ``properties``
        file (e.g. ``obs_title``, ``creator_did``).
    """
    import tempfile

    output_directory = Path(output_directory)

    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = Path(tmpdir) / "cube.fits"
        hdul = combine_fits_to_spectral_cube(
            file_paths, cube_path,
            freq_values=freq_values,
            min_elevation=min_elevation,
            quality_max_rms=quality_max_rms,
            quality_outlier_sigma=quality_outlier_sigma,
            quality_metric=quality_metric,
            quality_center_fraction=quality_center_fraction,
            quality_center_max_pixels=quality_center_max_pixels,
        )

        cube_hdu = hdul[0]
        freq_min_hz, freq_max_hz, initial_freq_hz = freq_range_from_cube_header(
            cube_hdu.header,
        )

        hips_properties = dict(properties) if properties else {}
        hips_properties.setdefault("hips_initial_freq", str(initial_freq_hz))

        # reproject's lower-resolution tile generation uses
        # block_reduce(..., 2) along the spectral axis.  With
        # tile_depth=1, that reduces 1→0 elements, causing a
        # broadcast error.  Enforce a minimum of 2.
        if tile_depth < 2:
            tile_depth = 2

        reproject_to_hips(
            cube_hdu,
            output_directory=str(output_directory),
            coord_system_out=coord_frame,
            reproject_function=reproject_interp,
            tile_size=tile_size,
            tile_depth=tile_depth,
            level=level,
            level_depth=level_depth,
            threads=threads,
            properties=hips_properties,
        )

        _finalize_hips3d_properties(
            output_directory,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            initial_freq_hz=initial_freq_hz,
            user_properties=properties,
        )
        write_hips3d_moc(
            output_directory,
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
        )

    _copy_index_html(output_directory, template="index_cube.html")
