"""Reproject FITS images to HEALPix, coadd, and export to HiPS."""

from __future__ import annotations

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


def _reproject_healpix_to_car(
    healpix_map: np.ndarray,
    coord_frame: str = "galactic",
    target_header: fits.Header | None = None,
    nested: bool = False,
) -> tuple[np.ndarray, fits.Header]:
    """Reproject a HEALPix map onto a Plate Carree (CAR) grid."""
    if target_header is None:
        target_header = DEFAULT_CAR_HEADER

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
        WCS header for the intermediate CAR grid. If *None*, a default
        full-sky 0.1-degree Galactic CAR grid is used.
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
