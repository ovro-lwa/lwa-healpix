"""HEALPix ↔ celestial WCS reverse projection and nested-tile helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import healpy as hp
import numpy as np
from astropy.io import fits
from reproject import reproject_from_healpix

__all__ = [
    "healpix_to_hdu",
    "iter_nested_tile_headers",
    "nested_tile_header",
    "pixel_scale_deg_for_nside",
    "reproject_healpix_to_wcs",
]


def pixel_scale_deg_for_nside(nside: int) -> float:
    """Return approximate HEALPix pixel scale in degrees for *nside*."""
    return float(np.degrees(np.sqrt(4.0 * np.pi / (12.0 * nside**2))))


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _is_galactic_frame(coord_frame: str) -> bool:
    return str(coord_frame).lower() in {"g", "galactic"}


def reproject_healpix_to_wcs(
    healpix_map: np.ndarray,
    target_header: fits.Header,
    *,
    coord_frame: str = "galactic",
    nested: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a 1-D HEALPix map onto a 2-D WCS grid.

    Parameters
    ----------
    healpix_map : numpy.ndarray
        1-D HEALPix map of length ``12 * nside**2``.
    target_header : `~astropy.io.fits.Header`
        WCS header describing the output grid (must include ``NAXIS1`` /
        ``NAXIS2`` and celestial ``CTYPE`` / ``CRVAL`` / ``CDELT`` / ``CRPIX``).
    coord_frame : str, optional
        Frame of *healpix_map*, passed to
        :func:`reproject.reproject_from_healpix` (e.g. ``\"galactic\"``,
        ``\"icrs\"``, ``\"g\"``, ``\"c\"``). Must match how the map was built.
        Default is ``\"galactic\"``.
    nested : bool, optional
        If ``True``, *healpix_map* uses NESTED ordering. Default is ``False``
        (RING).

    Returns
    -------
    data : numpy.ndarray
        2-D reprojected image with shape ``(NAXIS2, NAXIS1)``.
    footprint : numpy.ndarray
        2-D footprint / coverage weight from ``reproject_from_healpix``.
    """
    if healpix_map.ndim != 1:
        msg = f"healpix_map must be 1-D, got shape {healpix_map.shape}"
        raise ValueError(msg)
    if "NAXIS1" not in target_header or "NAXIS2" not in target_header:
        msg = "target_header must include NAXIS1 and NAXIS2"
        raise ValueError(msg)

    data, footprint = reproject_from_healpix(
        (healpix_map, coord_frame),
        target_header,
        nested=nested,
    )
    return data, footprint


def nested_tile_header(
    nside_tile: int,
    ipix: int,
    *,
    nside_map: int,
    overlap: float = 0.2,
    ctype: str = "TAN",
    coord_frame: str = "icrs",
) -> fits.Header:
    """Build a local TAN/SIN WCS header centered on a nested HEALPix pixel.

    Parameters
    ----------
    nside_tile : int
        HEALPix NSIDE of the tiling (must be a power of 2).
    ipix : int
        Nested pixel index at *nside_tile* (``0 .. 12*nside_tile**2 - 1``).
    nside_map : int
        NSIDE of the source HEALPix map (must be a power of 2 and divisible
        by *nside_tile*). Sets the output ``CDELT`` (map pixel scale).
    overlap : float, optional
        Fractional FOV growth beyond one tile pixel scale. Default is ``0.2``.
    ctype : {"TAN", "SIN"}, optional
        Celestial projection. Default is ``TAN``.
    coord_frame : str, optional
        Frame for ``CRVAL`` / ``CTYPE``. Use reproject healpix names:
        ``\"galactic\"`` / ``\"g\"`` → ``GLON-``/``GLAT-``; otherwise
        ``RA---``/``DEC--`` (e.g. ``\"icrs\"``, ``\"c\"``). Default is
        ``\"icrs\"``.

    Returns
    -------
    header : `~astropy.io.fits.Header`
        2-D WCS header suitable for :func:`reproject_healpix_to_wcs`.
    """
    if not _is_power_of_two(int(nside_tile)):
        msg = f"nside_tile must be a power of 2, got {nside_tile}"
        raise ValueError(msg)
    if not _is_power_of_two(int(nside_map)):
        msg = f"nside_map must be a power of 2, got {nside_map}"
        raise ValueError(msg)
    if int(nside_map) % int(nside_tile) != 0:
        msg = f"nside_map ({nside_map}) must be divisible by nside_tile ({nside_tile})"
        raise ValueError(msg)
    if overlap < 0:
        msg = f"overlap must be >= 0, got {overlap}"
        raise ValueError(msg)

    npix_tile = 12 * int(nside_tile) ** 2
    if not (0 <= int(ipix) < npix_tile):
        msg = f"ipix must be in [0, {npix_tile}), got {ipix}"
        raise ValueError(msg)

    ctype_key = str(ctype).upper()
    if ctype_key not in {"TAN", "SIN"}:
        msg = f"ctype must be 'TAN' or 'SIN', got {ctype!r}"
        raise ValueError(msg)

    galactic = _is_galactic_frame(coord_frame)
    theta, phi = hp.pix2ang(int(nside_tile), int(ipix), nest=True)
    lon_deg = float(np.degrees(phi))
    lat_deg = float(90.0 - np.degrees(theta))

    map_scale = pixel_scale_deg_for_nside(int(nside_map))
    tile_scale = pixel_scale_deg_for_nside(int(nside_tile))
    fov_deg = tile_scale * (1.0 + float(overlap))
    naxis = max(1, int(np.ceil(fov_deg / map_scale)))

    if galactic:
        ctype1, ctype2 = f"GLON-{ctype_key}", f"GLAT-{ctype_key}"
    else:
        ctype1, ctype2 = f"RA---{ctype_key}", f"DEC--{ctype_key}"

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = naxis
    header["NAXIS2"] = naxis
    header["CTYPE1"] = ctype1
    header["CTYPE2"] = ctype2
    header["CRVAL1"] = lon_deg
    header["CRVAL2"] = lat_deg
    header["CRPIX1"] = (naxis + 1) / 2.0
    header["CRPIX2"] = (naxis + 1) / 2.0
    header["CDELT1"] = -map_scale
    header["CDELT2"] = map_scale
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    if not galactic:
        header["RADESYS"] = "ICRS"
    return header


def iter_nested_tile_headers(
    nside_tile: int,
    nside_map: int,
    *,
    overlap: float = 0.2,
    ctype: str = "TAN",
    coord_frame: str = "icrs",
    weight: np.ndarray | None = None,
    min_weight_sum: float = 0.0,
) -> Iterator[tuple[int, fits.Header]]:
    """Yield ``(ipix, header)`` for nested tiles at *nside_tile*.

    If *weight* is provided (NESTED map at *nside_map*), tiles whose child
    pixels sum to ``<= min_weight_sum`` are skipped.
    """
    npix_tile = 12 * int(nside_tile) ** 2
    ratio = (int(nside_map) // int(nside_tile)) ** 2

    for ipix in range(npix_tile):
        if weight is not None:
            if weight.ndim != 1 or weight.size != 12 * int(nside_map) ** 2:
                msg = (
                    f"weight must have length 12*nside_map**2 "
                    f"({12 * int(nside_map) ** 2}), got {weight.size}"
                )
                raise ValueError(msg)
            start = int(ipix) * ratio
            stop = start + ratio
            if float(np.nansum(weight[start:stop])) <= float(min_weight_sum):
                continue
        yield ipix, nested_tile_header(
            nside_tile,
            ipix,
            nside_map=nside_map,
            overlap=overlap,
            ctype=ctype,
            coord_frame=coord_frame,
        )


def healpix_to_hdu(
    healpix_map: np.ndarray,
    target_header: fits.Header,
    *,
    weight: np.ndarray | None = None,
    coord_frame: str = "icrs",
    nested: bool = True,
    header_updates: Mapping[str, Any] | None = None,
) -> fits.PrimaryHDU:
    """Reproject HEALPix onto *target_header* and return a 2-D ``PrimaryHDU``.

    Pixels with non-positive footprint (or reprojected *weight* ``<= 0``) are
    set to NaN. Does **not** set ``BMAJ`` / ``BMIN`` / ``BPA``.

    Parameters
    ----------
    healpix_map : numpy.ndarray
        1-D HEALPix map.
    target_header : `~astropy.io.fits.Header`
        Output WCS (e.g. from :func:`nested_tile_header`).
    weight : numpy.ndarray or None, optional
        Optional 1-D weight map (same ordering as *healpix_map*). When given,
        pixels with reprojected weight ``<= 0`` are blanked.
    coord_frame : str, optional
        Frame of the HEALPix map (reproject healpix name). Default is
        ``\"icrs\"``.
    nested : bool, optional
        NESTED ordering if ``True``. Default is ``True``.
    header_updates : mapping or None, optional
        Extra header cards (e.g. ``BUNIT``) applied after the WCS copy.
    """
    data, footprint = reproject_healpix_to_wcs(
        healpix_map,
        target_header,
        coord_frame=coord_frame,
        nested=nested,
    )
    image = np.asarray(data, dtype=np.float32)
    blank = ~np.isfinite(image) | (footprint <= 0)

    if weight is not None:
        if weight.shape != healpix_map.shape:
            msg = (
                f"weight shape {weight.shape} must match "
                f"healpix_map shape {healpix_map.shape}"
            )
            raise ValueError(msg)
        weight_2d, _ = reproject_healpix_to_wcs(
            np.asarray(weight, dtype=np.float64),
            target_header,
            coord_frame=coord_frame,
            nested=nested,
        )
        blank |= ~np.isfinite(weight_2d) | (weight_2d <= 0)

    image[blank] = np.nan

    header = target_header.copy()
    if header_updates:
        for key, value in header_updates.items():
            header[key] = value
    return fits.PrimaryHDU(data=image, header=header)
