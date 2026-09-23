"""Read/write HEALPix map + weight FITS products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .healpix_wcs import normalize_coord_frame

__all__ = [
    "read_healpix_fits",
    "write_healpix_fits",
]


def _coordsys_code(coord_frame: str) -> str:
    frame = normalize_coord_frame(coord_frame)
    if frame == "galactic":
        return "G"
    if frame == "equatorial":
        return "C"
    msg = f"Unsupported coord_frame for HEALPix FITS: {coord_frame!r}"
    raise ValueError(msg)


def _frame_from_coordsys(code: str) -> str:
    key = str(code).strip().upper()
    if key in {"G", "GALACTIC"}:
        return "galactic"
    if key in {"C", "E", "EQUATORIAL", "CELESTIAL"}:
        # HEALPix FITS uses C for celestial; E is ecliptic — treat E as equatorial
        # only when explicitly equatorial aliases are used; map E → ecliptic.
        if key == "E":
            return "ecliptic"
        return "equatorial"
    msg = f"Unrecognized COORDSYS={code!r}"
    raise ValueError(msg)


def write_healpix_fits(
    path: str | Path,
    healpix_map: np.ndarray,
    weight: np.ndarray,
    *,
    nside: int,
    nested: bool = True,
    coord_frame: str = "equatorial",
    overwrite: bool = False,
) -> Path:
    """Write a multi-extension FITS with Primary metadata + MAP + WEIGHT.

    Layout
    ------
    - HDU 0 (Primary): ``NSIDE``, ``ORDERING``, ``COORDSYS``, ``PIXTYPE``,
      ``INDXSCHM=IMPLICIT`` (no image data).
    - HDU 1 (``MAP``): 1-D ``float32`` map, length ``12*nside**2``.
    - HDU 2 (``WEIGHT``): 1-D ``float32`` weight, same length.

    Parameters
    ----------
    path : str or Path
        Output path.
    healpix_map, weight : numpy.ndarray
        1-D arrays of length ``12 * nside**2``.
    nside : int
        HEALPix NSIDE.
    nested : bool, optional
        Write ``ORDERING=NESTED`` if ``True``, else ``RING``.
    coord_frame : str, optional
        ``equatorial`` → ``COORDSYS=C``; ``galactic`` → ``G``.
    overwrite : bool, optional
        Overwrite existing file.
    """
    path = Path(path)
    expected = 12 * int(nside) ** 2
    for name, arr in (("healpix_map", healpix_map), ("weight", weight)):
        if np.asarray(arr).ndim != 1 or np.asarray(arr).size != expected:
            msg = f"{name} must be 1-D with length {expected}, got {np.shape(arr)}"
            raise ValueError(msg)

    primary = fits.PrimaryHDU()
    primary.header["PIXTYPE"] = "HEALPIX"
    primary.header["NSIDE"] = int(nside)
    primary.header["ORDERING"] = "NESTED" if nested else "RING"
    primary.header["COORDSYS"] = _coordsys_code(coord_frame)
    primary.header["INDXSCHM"] = "IMPLICIT"
    primary.header["OBJECT"] = "FULLSKY"

    map_hdu = fits.ImageHDU(
        data=np.asarray(healpix_map, dtype=np.float32),
        name="MAP",
    )
    map_hdu.header["PIXTYPE"] = "HEALPIX"
    map_hdu.header["NSIDE"] = int(nside)
    map_hdu.header["ORDERING"] = primary.header["ORDERING"]
    map_hdu.header["COORDSYS"] = primary.header["COORDSYS"]
    map_hdu.header["INDXSCHM"] = "IMPLICIT"

    weight_hdu = fits.ImageHDU(
        data=np.asarray(weight, dtype=np.float32),
        name="WEIGHT",
    )
    weight_hdu.header["PIXTYPE"] = "HEALPIX"
    weight_hdu.header["NSIDE"] = int(nside)
    weight_hdu.header["ORDERING"] = primary.header["ORDERING"]
    weight_hdu.header["COORDSYS"] = primary.header["COORDSYS"]
    weight_hdu.header["INDXSCHM"] = "IMPLICIT"

    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, map_hdu, weight_hdu]).writeto(
        path, overwrite=overwrite,
    )
    return path


def read_healpix_fits(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read a file written by :func:`write_healpix_fits`.

    Returns
    -------
    healpix_map : numpy.ndarray
        1-D ``float32`` MAP extension.
    weight : numpy.ndarray
        1-D ``float32`` WEIGHT extension.
    meta : dict
        Keys ``nside``, ``nested``, ``coord_frame``.
    """
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header
        try:
            map_hdu = hdul["MAP"]
            weight_hdu = hdul["WEIGHT"]
        except KeyError as exc:
            msg = f"{path} missing MAP/WEIGHT extensions"
            raise ValueError(msg) from exc

        nside = int(primary.get("NSIDE", map_hdu.header.get("NSIDE")))
        ordering = str(primary.get("ORDERING", map_hdu.header.get("ORDERING", "NESTED")))
        coordsys = str(primary.get("COORDSYS", map_hdu.header.get("COORDSYS", "C")))
        nested = ordering.upper().startswith("NEST")
        coord_frame = _frame_from_coordsys(coordsys)

        healpix_map = np.asarray(map_hdu.data, dtype=np.float32).ravel()
        weight = np.asarray(weight_hdu.data, dtype=np.float32).ravel()

    expected = 12 * nside**2
    if healpix_map.size != expected or weight.size != expected:
        msg = (
            f"MAP/WEIGHT length {healpix_map.size}/{weight.size} "
            f"!= 12*nside**2 ({expected})"
        )
        raise ValueError(msg)

    meta = {
        "nside": nside,
        "nested": nested,
        "coord_frame": coord_frame,
    }
    return healpix_map, weight, meta
