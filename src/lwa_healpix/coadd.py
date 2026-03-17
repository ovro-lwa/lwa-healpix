"""Reproject FITS images to HEALPix, coadd, and export to HiPS."""

from __future__ import annotations

import logging
import shutil
from importlib import resources
from pathlib import Path

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from reproject import reproject_from_healpix, reproject_interp, reproject_to_healpix
from reproject.hips import reproject_to_hips

__all__ = [
    "coadd_fits_to_healpix",
    "combine_fits_to_spectral_cube",
    "fits_to_hips_cube",
    "healpix_to_hips",
]

logger = logging.getLogger(__name__)

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


def _pixel_elevations(wcs_2d: wcs.WCS, shape: tuple[int, int]) -> np.ndarray:
    """Return the elevation in degrees of every pixel in a 2-D image.

    Elevation is defined as 90 degrees minus the angular separation from
    the image reference point (``CRVAL``), which is assumed to be the
    local zenith.
    """
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    sky = wcs_2d.pixel_to_world(x, y)
    center = SkyCoord(
        wcs_2d.wcs.crval[0], wcs_2d.wcs.crval[1],
        unit="deg", frame=sky.frame.name,
    )
    return 90.0 - sky.separation(center).deg


def coadd_fits_to_healpix(
    file_paths: list[str | Path],
    nside: int = 1024,
    coord_frame: str = "galactic",
    nested: bool = False,
    min_elevation: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject FITS images to HEALPix and coadd them.

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to FITS files to reproject and coadd.
    nside : int, optional
        HEALPix NSIDE parameter controlling resolution. Default is 1024.
    coord_frame : str, optional
        Coordinate frame for the HEALPix projection (e.g. ``"galactic"``).
    nested : bool, optional
        If ``True``, use the NESTED pixel ordering. Default is ``False`` (RING).
    min_elevation : float or None, optional
        Minimum elevation above the horizon in degrees.  Pixels below this
        elevation are blanked before reprojection so that noisy data near the
        horizon is excluded from the coadd.  Elevation is measured as the
        angular distance from the image reference point (assumed to be the
        local zenith).  If *None* (the default), no masking is applied.

    Returns
    -------
    combined : numpy.ndarray
        Weighted-average coadded HEALPix map.
    total_weight : numpy.ndarray
        Sum of footprint weights per pixel.
    """
    healpix_maps = []
    healpix_weights = []

    for fpath in file_paths:
        hdu = fits.open(fpath)[0]
        w0 = wcs.WCS(hdu)
        wcs_2d = w0.dropaxis(3).dropaxis(2)
        data = hdu.data.squeeze()

        if min_elevation is not None:
            elevation = _pixel_elevations(wcs_2d, data.shape)
            data = data.copy()
            data[elevation < min_elevation] = np.nan

        hpx_array, hpx_footprint = reproject_to_healpix(
            (data, wcs_2d), coord_frame, nside=nside, nested=nested
        )
        healpix_maps.append(np.nan_to_num(hpx_array, nan=0.0))
        healpix_weights.append(hpx_footprint)

    maps = np.array(healpix_maps)
    weights = np.array(healpix_weights)
    total_weight = np.sum(weights, axis=0)
    combined = np.where(
        total_weight > 0,
        np.sum(maps * weights, axis=0) / total_weight,
        0.0,
    )

    return combined, total_weight


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

    index_src = resources.files("lwa_healpix") / "data" / "index.html"
    shutil.copy2(index_src, output_directory / "index.html")


def _find_spectral_axis(header: fits.Header) -> int:
    """Return the 1-based FITS axis number of the frequency axis.

    Raises ``ValueError`` if no frequency axis can be found.
    """
    for i in range(1, header.get("NAXIS", 0) + 1):
        ctype = header.get(f"CTYPE{i}", "")
        if ctype.upper().startswith("FREQ"):
            return i

    msg = "Cannot find a FREQ axis in the FITS header"
    raise ValueError(msg)


def _find_stokes_axis(header: fits.Header) -> int:
    """Return the 1-based FITS axis number of the Stokes axis.

    Raises ``ValueError`` if no Stokes axis can be found.
    """
    for i in range(1, header.get("NAXIS", 0) + 1):
        ctype = header.get(f"CTYPE{i}", "")
        if ctype.upper().startswith("STOKES"):
            return i

    msg = "Cannot find a STOKES axis in the FITS header"
    raise ValueError(msg)


def combine_fits_to_spectral_cube(
    file_paths: list[str | Path],
    output_path: str | Path,
    *,
    freq_values: list[float] | np.ndarray | None = None,
) -> fits.HDUList:
    """Combine single-frequency FITS images into a spectral cube.

    Each input file is expected to contain a 4-D FITS image with axes
    ``(RA, Dec, Freq, Stokes)`` where the frequency and Stokes axes are
    both length 1.  The Stokes axis is dropped and the single-channel
    frequency planes are stacked to produce a 3-D cube
    ``(RA, Dec, Freq)`` suitable for HiPS3D generation via
    :func:`reproject.hips.reproject_to_hips`.

    All input images must share the same spatial pixel grid (``NAXIS1``,
    ``NAXIS2``, and spatial WCS).

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to the FITS files, one per frequency channel.
    output_path : str or Path
        Path for the output spectral-cube FITS file.
    freq_values : array-like or None, optional
        Explicit frequency values (in the same unit as the input FITS
        spectral axis) for each file, in the same order as *file_paths*.
        When provided, header frequency look-up is skipped.

    Returns
    -------
    hdul : `~astropy.io.fits.HDUList`
        The written spectral-cube HDU list.

    Raises
    ------
    ValueError
        If fewer than two files are given, the spatial dimensions are
        inconsistent, or frequency information cannot be determined.
    """
    if len(file_paths) < 2:
        msg = "At least two FITS files are required to build a spectral cube"
        raise ValueError(msg)

    hdus: list[fits.PrimaryHDU] = []
    freqs: list[float] = []

    for idx, fpath in enumerate(file_paths):
        hdu = fits.open(fpath)[0]
        hdus.append(hdu)

        if freq_values is not None:
            freqs.append(float(freq_values[idx]))
        else:
            freq_ax = _find_spectral_axis(hdu.header)
            freqs.append(float(hdu.header[f"CRVAL{freq_ax}"]))

    sort_idx = np.argsort(freqs)
    hdus = [hdus[i] for i in sort_idx]
    freqs_sorted = np.array([freqs[i] for i in sort_idx])

    ref_hdu = hdus[0]
    ref_header = ref_hdu.header
    freq_ax = _find_spectral_axis(ref_header)
    stokes_ax = _find_stokes_axis(ref_header)

    naxis = ref_header["NAXIS"]
    stokes_numpy = naxis - stokes_ax
    freq_numpy = naxis - freq_ax

    # Remove higher axis first so lower axis index is unaffected.
    ax_hi, ax_lo = sorted([stokes_numpy, freq_numpy], reverse=True)
    ref_data = np.take(np.take(ref_hdu.data, 0, axis=ax_hi), 0, axis=ax_lo)
    if ref_data.ndim != 2:
        msg = (
            f"Expected 2-D spatial image after removing Stokes and Freq axes, "
            f"got {ref_data.ndim}-D from {file_paths[sort_idx[0]]}"
        )
        raise ValueError(msg)

    ny, nx = ref_data.shape
    nfreq = len(hdus)
    cube = np.empty((nfreq, ny, nx), dtype=ref_data.dtype)

    for i, hdu in enumerate(hdus):
        data = np.take(np.take(hdu.data, 0, axis=ax_hi), 0, axis=ax_lo)
        if data.shape != (ny, nx):
            msg = (
                f"Spatial shape mismatch: expected ({ny}, {nx}), "
                f"got {data.shape} from {file_paths[sort_idx[i]]}"
            )
            raise ValueError(msg)
        cube[i] = data

    ref_wcs_full = wcs.WCS(ref_header)
    header_2d = ref_wcs_full.celestial.to_header()

    header_3d = fits.Header()
    header_3d["NAXIS"] = 3
    header_3d["NAXIS1"] = nx
    header_3d["NAXIS2"] = ny
    header_3d["NAXIS3"] = nfreq

    # Copy all spatial WCS keywords (CDELT, PC/CD matrix, LONPOLE, etc.)
    # but remap WCSAXES from 2 to 3.
    for card in header_2d.cards:
        if card.keyword == "WCSAXES":
            header_3d["WCSAXES"] = 3
        elif card.keyword:
            header_3d[card.keyword] = (card.value, card.comment)

    freq_suffix = str(freq_ax)
    header_3d["CTYPE3"] = ref_header.get(f"CTYPE{freq_suffix}", "FREQ")
    header_3d["CRPIX3"] = 1.0
    header_3d["CRVAL3"] = freqs_sorted[0]
    header_3d["CUNIT3"] = ref_header.get(f"CUNIT{freq_suffix}", "Hz")

    if nfreq > 1:
        cdelt3 = float(freqs_sorted[1] - freqs_sorted[0])
        if cdelt3 == 0:
            msg = "First two frequencies are identical; cannot determine channel spacing"
            raise ValueError(msg)
        header_3d["CDELT3"] = cdelt3

        expected = freqs_sorted[0] + np.arange(nfreq) * cdelt3
        max_deviation = np.max(np.abs(freqs_sorted - expected))
        if max_deviation > 0.01 * abs(cdelt3):
            logger.warning(
                "Frequencies are not uniformly spaced (max deviation: "
                "%.3g %s). The FITS WCS frequency axis assumes uniform "
                "spacing; downstream tools may misinterpret channel "
                "frequencies.",
                max_deviation,
                header_3d["CUNIT3"],
            )

    for key in ("TELESCOP", "INSTRUME", "OBSERVER", "OBJECT",
                "DATE-OBS", "BUNIT", "EQUINOX", "RADESYS"):
        val = ref_header.get(key)
        if val is not None:
            header_3d[key] = val

    primary = fits.PrimaryHDU(data=cube, header=header_3d)
    hdul = fits.HDUList([primary])

    output_path = Path(output_path)
    hdul.writeto(output_path, overwrite=True)
    logger.info(
        "Wrote spectral cube (%d channels, %d x %d) to %s",
        nfreq, nx, ny, output_path,
    )
    return hdul


def fits_to_hips_cube(
    file_paths: list[str | Path],
    output_directory: str | Path = "hips_cube_output",
    *,
    coord_frame: str = "galactic",
    freq_values: list[float] | np.ndarray | None = None,
    tile_size: int = 512,
    tile_depth: int = 16,
    level: int | None = None,
    level_depth: int | None = None,
    threads: bool = True,
    properties: dict[str, str] | None = None,
) -> None:
    """Build a HiPS cube from single-frequency FITS images.

    This is a convenience wrapper that calls
    :func:`combine_fits_to_spectral_cube` to assemble a 3-D spectral
    cube and then passes it to :func:`reproject.hips.reproject_to_hips`
    to generate a HiPS3D tile set.

    Each input file is expected to contain a 4-D FITS image with axes
    ``(RA, Dec, Freq, Stokes)`` where frequency and Stokes are both
    length 1.  See :func:`combine_fits_to_spectral_cube` for details.

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to the FITS files, one per frequency channel.
    output_directory : str or Path, optional
        Directory to write HiPS tiles into.  Must not already exist.
        Default is ``"hips_cube_output"``.
    coord_frame : str, optional
        Coordinate system for the HiPS output (``"galactic"``,
        ``"equatorial"``, or ``"ecliptic"``).  Default is ``"galactic"``.
    freq_values : array-like or None, optional
        Explicit frequency values for each file (same order as
        *file_paths*).  Passed through to
        :func:`combine_fits_to_spectral_cube`.
    tile_size : int, optional
        Spatial tile size in pixels.  Default is 512.
    tile_depth : int, optional
        Depth of each tile along the spectral axis.  Default is 16.
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
            file_paths, cube_path, freq_values=freq_values,
        )

        cube_hdu = hdul[0]

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

    index_src = resources.files("lwa_healpix") / "data" / "index.html"
    shutil.copy2(index_src, output_directory / "index.html")
