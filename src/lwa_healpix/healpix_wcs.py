"""HEALPix ↔ celestial WCS reverse projection and nested-tile helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs.utils import fit_wcs_from_points
from reproject import reproject_from_healpix

__all__ = [
    "healpix_to_hdu",
    "iter_nested_tile_headers",
    "nested_tile_header",
    "pixel_scale_deg_for_nside",
    "reproject_healpix_to_wcs",
]

TileAlign = Literal["diamond", "celestial"]


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


def _validate_tile_nsides(nside_tile: int, nside_map: int, ipix: int) -> None:
    if not _is_power_of_two(int(nside_tile)):
        msg = f"nside_tile must be a power of 2, got {nside_tile}"
        raise ValueError(msg)
    if not _is_power_of_two(int(nside_map)):
        msg = f"nside_map must be a power of 2, got {nside_map}"
        raise ValueError(msg)
    if int(nside_map) % int(nside_tile) != 0:
        msg = f"nside_map ({nside_map}) must be divisible by nside_tile ({nside_tile})"
        raise ValueError(msg)
    npix_tile = 12 * int(nside_tile) ** 2
    if not (0 <= int(ipix) < npix_tile):
        msg = f"ipix must be in [0, {npix_tile}), got {ipix}"
        raise ValueError(msg)


def _tile_center_lonlat_deg(nside_tile: int, ipix: int) -> tuple[float, float]:
    theta, phi = hp.pix2ang(int(nside_tile), int(ipix), nest=True)
    return float(np.degrees(phi)), float(90.0 - np.degrees(theta))


def _skycoord_frame_name(coord_frame: str) -> str:
    return "galactic" if _is_galactic_frame(coord_frame) else "icrs"


def _diamond_edge_frame(
    nside_tile: int,
    ipix: int,
    *,
    coord_frame: str,
) -> tuple[SkyCoord, np.ndarray, np.ndarray, float]:
    """Return ``(center, e1, e2, half_side_deg)`` for a diamond-aligned square.

    ``e1`` / ``e2`` are orthonormal tangent-plane axes (east, north) along the
    HEALPix diamond *edges*. ``half_side_deg`` is the half-width of the square
    that contains all pixel vertices in that frame.
    """
    lon0, lat0 = _tile_center_lonlat_deg(nside_tile, ipix)
    frame = _skycoord_frame_name(coord_frame)
    center = SkyCoord(lon0 * u.deg, lat0 * u.deg, frame=frame)

    vec = hp.boundaries(int(nside_tile), int(ipix), step=1, nest=True)
    theta, phi = hp.vec2ang(vec.T)
    lon = np.degrees(phi)
    lat = 90.0 - np.degrees(theta)
    verts = SkyCoord(lon * u.deg, lat * u.deg, frame=frame)

    sep = center.separation(verts).to(u.deg).value
    pa = center.position_angle(verts).to(u.rad).value
    east = sep * np.sin(pa)
    north = sep * np.cos(pa)
    xy = np.column_stack([east, north])

    norms = np.hypot(east, north)
    order = np.argsort(-norms)
    d1 = xy[order[0]].astype(float)
    d1 /= np.linalg.norm(d1) + 1e-15

    best_i, best_score = int(order[1]), -1.0
    for i in order[1:]:
        v = xy[i].astype(float)
        v /= np.linalg.norm(v) + 1e-15
        score = abs(d1[0] * v[1] - d1[1] * v[0])
        if score > best_score:
            best_score, best_i = score, int(i)
    d2 = xy[best_i].astype(float)
    d2 = d2 - np.dot(d2, d1) * d1
    n2 = np.linalg.norm(d2)
    d2 = d2 / n2 if n2 > 1e-15 else np.array([-d1[1], d1[0]], dtype=float)

    # Edge directions = 45° from the diamond diagonals.
    e1 = d1 + d2
    e2 = d1 - d2
    e1 /= np.linalg.norm(e1) + 1e-15
    e2 /= np.linalg.norm(e2) + 1e-15
    if e1[0] * e2[1] - e1[1] * e2[0] < 0:
        e2 = -e2

    half = float(max(np.abs(xy @ e1).max(), np.abs(xy @ e2).max()))
    return center, e1, e2, half


def _offset_sky(center: SkyCoord, east_deg: float, north_deg: float) -> SkyCoord:
    sep = float(np.hypot(east_deg, north_deg))
    if sep < 1e-15:
        return center
    pa = float(np.arctan2(east_deg, north_deg))
    return center.directional_offset_by(pa * u.rad, sep * u.deg)


def _diamond_aligned_header(
    nside_tile: int,
    ipix: int,
    *,
    nside_map: int,
    margin: float,
    ctype: str,
    coord_frame: str,
) -> fits.Header:
    """TAN/SIN header with image axes along HEALPix diamond edges."""
    ctype_key = str(ctype).upper()
    map_scale = pixel_scale_deg_for_nside(int(nside_map))
    center, e1, e2, half = _diamond_edge_frame(
        nside_tile, ipix, coord_frame=coord_frame
    )
    fov_deg = 2.0 * half * (1.0 + float(margin))
    naxis = max(1, int(np.ceil(fov_deg / map_scale)))
    crpix = (naxis + 1) / 2.0

    pix: list[list[float]] = []
    skies: list[SkyCoord] = []
    for di, dj in (
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (5, 0),
        (0, 5),
        (-5, 0),
        (0, -5),
        (5, 5),
        (-5, 5),
        (5, -5),
        (-5, -5),
    ):
        east = (di * e1[0] + dj * e2[0]) * map_scale
        north = (di * e1[1] + dj * e2[1]) * map_scale
        pix.append([crpix - 1.0 + di, crpix - 1.0 + dj])
        skies.append(_offset_sky(center, east, north))

    wcs = fit_wcs_from_points(
        np.array(pix).T,
        SkyCoord(skies),
        proj_point=center,
        projection=ctype_key,
    )
    # Prefer a CD matrix at map scale; force exact CRPIX / NAXIS.
    scale_matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float)
    galactic = _is_galactic_frame(coord_frame)
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
    header["CRVAL1"] = float(center.spherical.lon.degree)
    header["CRVAL2"] = float(center.spherical.lat.degree)
    header["CRPIX1"] = crpix
    header["CRPIX2"] = crpix
    header["CD1_1"] = float(scale_matrix[0, 0])
    header["CD1_2"] = float(scale_matrix[0, 1])
    header["CD2_1"] = float(scale_matrix[1, 0])
    header["CD2_2"] = float(scale_matrix[1, 1])
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    if not galactic:
        header["RADESYS"] = "ICRS"
    return header


def _celestial_aligned_header(
    nside_tile: int,
    ipix: int,
    *,
    nside_map: int,
    overlap: float,
    ctype: str,
    coord_frame: str,
) -> fits.Header:
    """North-aligned TAN/SIN square (legacy FOV = tile_scale * (1+overlap))."""
    ctype_key = str(ctype).upper()
    lon_deg, lat_deg = _tile_center_lonlat_deg(nside_tile, ipix)
    map_scale = pixel_scale_deg_for_nside(int(nside_map))
    tile_scale = pixel_scale_deg_for_nside(int(nside_tile))
    fov_deg = tile_scale * (1.0 + float(overlap))
    naxis = max(1, int(np.ceil(fov_deg / map_scale)))

    galactic = _is_galactic_frame(coord_frame)
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


def nested_tile_header(
    nside_tile: int,
    ipix: int,
    *,
    nside_map: int,
    overlap: float = 0.2,
    margin: float = 0.05,
    align: TileAlign = "diamond",
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
        by *nside_tile*). Sets the output pixel scale.
    overlap : float, optional
        Used only when ``align="celestial"``: fractional FOV growth beyond one
        mean tile pixel scale. Default is ``0.2``.
    margin : float, optional
        Used only when ``align="diamond"``: fractional growth of the
        diamond-aligned square beyond the HEALPix vertex hull. Default is
        ``0.05`` (closes all-sky coverage gaps that ``overlap=0.2`` leaves
        with celestial-aligned squares).
    align : {"diamond", "celestial"}, optional
        ``"diamond"`` (default) rotates the TAN/SIN axes onto the HEALPix
        diamond edges so a modest ``margin`` covers the cell. ``"celestial"``
        keeps a north-aligned square sized by ``overlap`` (legacy).
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
    _validate_tile_nsides(nside_tile, nside_map, ipix)
    if overlap < 0:
        msg = f"overlap must be >= 0, got {overlap}"
        raise ValueError(msg)
    if margin < 0:
        msg = f"margin must be >= 0, got {margin}"
        raise ValueError(msg)

    ctype_key = str(ctype).upper()
    if ctype_key not in {"TAN", "SIN"}:
        msg = f"ctype must be 'TAN' or 'SIN', got {ctype!r}"
        raise ValueError(msg)

    align_key = str(align).lower()
    if align_key == "diamond":
        return _diamond_aligned_header(
            nside_tile,
            ipix,
            nside_map=nside_map,
            margin=margin,
            ctype=ctype_key,
            coord_frame=coord_frame,
        )
    if align_key == "celestial":
        return _celestial_aligned_header(
            nside_tile,
            ipix,
            nside_map=nside_map,
            overlap=overlap,
            ctype=ctype_key,
            coord_frame=coord_frame,
        )
    msg = f"align must be 'diamond' or 'celestial', got {align!r}"
    raise ValueError(msg)


def iter_nested_tile_headers(
    nside_tile: int,
    nside_map: int,
    *,
    overlap: float = 0.2,
    margin: float = 0.05,
    align: TileAlign = "diamond",
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
            margin=margin,
            align=align,
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
