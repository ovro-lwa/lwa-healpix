"""Generate HiPS (2-D) and HiPS3D (spectral cube) tile sets."""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import numpy as np
from astropy.io import fits
from reproject import reproject_from_healpix, reproject_interp
from reproject.hips import reproject_to_hips

from .coadd import combine_fits_to_spectral_cube

__all__ = [
    "fits_to_hips",
    "fits_to_hips_cube",
    "healpix_to_hips",
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


def _copy_index_html(output_directory: Path) -> None:
    """Copy the bundled ``index.html`` viewer into *output_directory*."""
    index_src = resources.files("lwa_healpix") / "data" / "index.html"
    shutil.copy2(index_src, output_directory / "index.html")


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
    tile_size: int = 512,
    tile_depth: int = 2,
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
    set.

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
    tile_size : int, optional
        Spatial tile size in pixels.  Default is 512.
    tile_depth : int, optional
        Depth of each tile along the spectral axis.  Must be at least
        2 (reproject requires this for lower-resolution tile
        generation).  Default is 2.
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
        )

        cube_hdu = hdul[0]

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
            properties=properties,
        )

    _copy_index_html(output_directory)
