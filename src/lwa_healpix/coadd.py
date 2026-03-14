"""Reproject FITS images to HEALPix, coadd, and export to HiPS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import wcs
from astropy.io import fits
from reproject import reproject_from_healpix, reproject_interp, reproject_to_healpix
from reproject.hips import reproject_to_hips

__all__ = [
    "coadd_fits_to_healpix",
    "make_hips",
    "reproject_healpix_to_car",
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


def coadd_fits_to_healpix(
    file_paths: list[str | Path],
    nside: int = 1024,
    coord_frame: str = "galactic",
    nested: bool = False,
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
        input_data = (hdu.data.squeeze(), w0.dropaxis(3).dropaxis(2))

        hpx_array, hpx_footprint = reproject_to_healpix(
            input_data, coord_frame, nside=nside, nested=nested
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


def reproject_healpix_to_car(
    healpix_map: np.ndarray,
    coord_frame: str = "galactic",
    target_header: fits.Header | None = None,
    nested: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a HEALPix map onto a Plate Carr\u00e9e (CAR) grid.

    Parameters
    ----------
    healpix_map : numpy.ndarray
        1-D HEALPix map array.
    coord_frame : str, optional
        Coordinate frame of the input map (e.g. ``"galactic"``).
    target_header : `~astropy.io.fits.Header`, optional
        WCS header describing the output grid. If *None*, a default
        full-sky 0.1-degree Galactic CAR grid is used.
    nested : bool, optional
        If ``True``, the input uses NESTED ordering. Default is ``False``.

    Returns
    -------
    flat_array : numpy.ndarray
        The reprojected 2-D image.
    footprint : numpy.ndarray
        Coverage footprint of the reprojection.
    """
    if target_header is None:
        target_header = DEFAULT_CAR_HEADER

    return reproject_from_healpix(
        (healpix_map, coord_frame), target_header, nested=nested
    )


def make_hips(
    image: np.ndarray,
    header: fits.Header | None = None,
    output_directory: str | Path = "hips_output",
    coord_system_out: str = "galactic",
    threads: bool = True,
) -> None:
    """Generate a HiPS tile set from a 2-D image with WCS.

    Parameters
    ----------
    image : numpy.ndarray
        2-D image array.
    header : `~astropy.io.fits.Header`, optional
        WCS header for *image*. If *None*, the default CAR header is used.
    output_directory : str or Path, optional
        Directory to write HiPS tiles into. Default is ``"hips_output"``.
    coord_system_out : str, optional
        Output coordinate system (e.g. ``"galactic"``).
    threads : bool, optional
        Whether to use multi-threaded reprojection. Default is ``True``.
    """
    if header is None:
        header = DEFAULT_CAR_HEADER

    reproject_to_hips(
        (image, header),
        output_directory=str(output_directory),
        coord_system_out=coord_system_out,
        reproject_function=reproject_interp,
        threads=threads,
    )
