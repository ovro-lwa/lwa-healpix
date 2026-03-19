"""Reproject FITS images to HEALPix, coadd, and export to HiPS."""

from __future__ import annotations

import logging
import re
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
    "coadd_fits",
    "combine_fits_to_spectral_cube",
    "fits_to_hips_cube",
    "group_pipeline_files",
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


def _extract_2d(hdu: fits.PrimaryHDU) -> tuple[np.ndarray, wcs.WCS]:
    """Return the 2-D spatial data and WCS from a 4-axis HDU.

    Drops the Stokes and Freq axes (both assumed to be length 1).
    """
    header = hdu.header
    freq_ax = _find_spectral_axis(header)
    stokes_ax = _find_stokes_axis(header)

    naxis = header["NAXIS"]
    stokes_numpy = naxis - stokes_ax
    freq_numpy = naxis - freq_ax

    ax_hi, ax_lo = sorted([stokes_numpy, freq_numpy], reverse=True)
    data = np.take(np.take(hdu.data, 0, axis=ax_hi), 0, axis=ax_lo)
    wcs_2d = wcs.WCS(header).celestial
    return data, wcs_2d


def coadd_fits(
    file_paths: list[str | Path],
    *,
    nside: int | None = None,
    target_header: fits.Header | None = None,
    coord_frame: str = "galactic",
    nested: bool = False,
    min_elevation: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject FITS images onto a common grid and coadd them.

    Each input file is expected to have 4 axes ``(RA, Dec, Freq,
    Stokes)`` with length-1 frequency and Stokes dimensions.  The 2-D
    spatial plane is extracted, optionally masked by elevation, then
    reprojected and accumulated with footprint-based weighting.

    Exactly one of *nside* or *target_header* must be given:

    * **nside** — reproject onto a HEALPix grid.  Returns a 1-D map.
    * **target_header** — reproject onto a 2-D image grid described by
      the FITS header.  Returns a 2-D array.

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to FITS files to reproject and coadd.
    nside : int or None, optional
        HEALPix NSIDE parameter.  Mutually exclusive with
        *target_header*.
    target_header : `~astropy.io.fits.Header` or None, optional
        WCS header for a 2-D target grid.  Must include ``NAXIS1`` and
        ``NAXIS2``.  Mutually exclusive with *nside*.
    coord_frame : str, optional
        Coordinate frame for the output (e.g. ``"galactic"``).  Used
        when *nside* is given.  Default is ``"galactic"``.
    nested : bool, optional
        HEALPix NESTED ordering.  Only relevant when *nside* is given.
    min_elevation : float or None, optional
        Minimum elevation above the horizon in degrees.  Pixels below
        this elevation are blanked before reprojection.  Elevation is
        measured as the angular distance from the image reference point
        (assumed to be the local zenith).

    Returns
    -------
    combined : numpy.ndarray
        Weighted-average coadded map (1-D for HEALPix, 2-D for image).
    total_weight : numpy.ndarray
        Sum of footprint weights per pixel.
    """
    if (nside is None) == (target_header is None):
        msg = "Exactly one of 'nside' or 'target_header' must be given"
        raise ValueError(msg)

    if target_header is not None:
        shape_out = (
            int(target_header["NAXIS2"]),
            int(target_header["NAXIS1"]),
        )

    sum_data: np.ndarray | None = None
    sum_weight: np.ndarray | None = None

    for fpath in file_paths:
        hdu = fits.open(fpath)[0]
        data_2d, wcs_2d = _extract_2d(hdu)

        if min_elevation is not None:
            elevation = _pixel_elevations(wcs_2d, data_2d.shape)
            data_2d = data_2d.copy()
            data_2d[elevation < min_elevation] = np.nan

        if nside is not None:
            reprojected, footprint = reproject_to_healpix(
                (data_2d, wcs_2d), coord_frame,
                nside=nside, nested=nested,
            )
        else:
            reprojected, footprint = reproject_interp(
                (data_2d, wcs_2d), target_header,
                shape_out=shape_out,
            )

        reprojected = np.nan_to_num(reprojected, nan=0.0)

        if sum_data is None:
            sum_data = reprojected * footprint
            sum_weight = footprint.copy()
        else:
            sum_data += reprojected * footprint
            sum_weight += footprint

    total_weight = sum_weight
    combined = np.zeros_like(sum_data)
    np.divide(sum_data, total_weight, out=combined, where=total_weight > 0)
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


def combine_fits_to_spectral_cube(
    file_paths: list[str | Path] | dict[float, list[str | Path]],
    output_path: str | Path,
    *,
    freq_values: list[float] | np.ndarray | None = None,
    min_elevation: float | None = None,
) -> fits.HDUList:
    """Combine FITS images into a spectral cube, with optional coadding.

    Each input file is expected to contain a 4-D FITS image with axes
    ``(RA, Dec, Freq, Stokes)`` where the frequency and Stokes axes are
    both length 1.

    *file_paths* may be given in two forms:

    * **flat list** -- one file per frequency channel.  All files must
      share the same spatial pixel grid.
    * **dict** -- keys are frequency values and values are lists of
      files at that frequency.  Files within each group are reprojected
      onto a common spatial grid (from the first file of the lowest-
      frequency group) and coadded via :func:`coadd_fits` before being
      stacked into the cube.

    When a flat list contains duplicate frequencies, files at the same
    frequency are automatically grouped and coadded.

    Parameters
    ----------
    file_paths : list or dict
        Paths to the FITS files.  A flat list gives one file per channel
        (or auto-groups by frequency).  A ``dict[float, list[path]]``
        maps each frequency to files to be coadded.
    output_path : str or Path
        Path for the output spectral-cube FITS file.
    freq_values : array-like or None, optional
        Explicit frequency values for each file in the *flat-list* form.
        Ignored when *file_paths* is a dict (keys are the frequencies).
    min_elevation : float or None, optional
        Minimum elevation in degrees.  Passed through to
        :func:`coadd_fits` for per-channel coadding.

    Returns
    -------
    hdul : `~astropy.io.fits.HDUList`
        The written spectral-cube HDU list.

    Raises
    ------
    ValueError
        If fewer than two distinct frequency channels result, spatial
        dimensions are inconsistent, or frequency information cannot be
        determined.
    """
    # ------------------------------------------------------------------
    # Normalise input to  freq_groups: dict[float, list[Path]]
    # ------------------------------------------------------------------
    if isinstance(file_paths, dict):
        freq_groups: dict[float, list[str | Path]] = {
            float(k): list(v) for k, v in file_paths.items()
        }
    else:
        if len(file_paths) < 2:
            msg = "At least two FITS files are required to build a spectral cube"
            raise ValueError(msg)

        freqs_raw: list[float] = []
        for idx, fpath in enumerate(file_paths):
            if freq_values is not None:
                freqs_raw.append(float(freq_values[idx]))
            else:
                hdr = fits.getheader(fpath)
                ax = _find_spectral_axis(hdr)
                freqs_raw.append(float(hdr[f"CRVAL{ax}"]))

        freq_groups = {}
        for freq, fpath in zip(freqs_raw, file_paths):
            freq_groups.setdefault(freq, []).append(fpath)

    sorted_freqs = sorted(freq_groups)
    if len(sorted_freqs) < 2:
        msg = "At least two distinct frequency channels are required"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # Reference spatial grid from the first file at the lowest frequency.
    # ------------------------------------------------------------------
    first_hdu = fits.open(freq_groups[sorted_freqs[0]][0])[0]
    ref_data, ref_wcs_2d = _extract_2d(first_hdu)
    if ref_data.ndim != 2:
        msg = (
            f"Expected 2-D spatial image after removing Stokes and Freq "
            f"axes, got {ref_data.ndim}-D"
        )
        raise ValueError(msg)
    ny, nx = ref_data.shape
    ref_header_full = first_hdu.header

    ref_target = ref_wcs_2d.to_header()
    ref_target["NAXIS"] = 2
    ref_target["NAXIS1"] = nx
    ref_target["NAXIS2"] = ny

    # ------------------------------------------------------------------
    # Build each frequency plane
    # ------------------------------------------------------------------
    nfreq = len(sorted_freqs)
    cube = np.empty((nfreq, ny, nx), dtype=np.float32)

    for i, freq in enumerate(sorted_freqs):
        group = freq_groups[freq]
        if len(group) == 1:
            hdu = fits.open(group[0])[0]
            plane, _ = _extract_2d(hdu)
            if plane.shape != (ny, nx):
                msg = (
                    f"Spatial shape mismatch: expected ({ny}, {nx}), "
                    f"got {plane.shape} from {group[0]}"
                )
                raise ValueError(msg)
        else:
            plane, _ = coadd_fits(
                group,
                target_header=ref_target,
                min_elevation=min_elevation,
            )
        cube[i] = plane

    # ------------------------------------------------------------------
    # Assemble the 3-D WCS header
    # ------------------------------------------------------------------
    header_2d = ref_wcs_2d.to_header()
    freqs_sorted = np.array(sorted_freqs)

    header_3d = fits.Header()
    header_3d["NAXIS"] = 3
    header_3d["NAXIS1"] = nx
    header_3d["NAXIS2"] = ny
    header_3d["NAXIS3"] = nfreq

    for card in header_2d.cards:
        if card.keyword == "WCSAXES":
            header_3d["WCSAXES"] = 3
        elif card.keyword:
            header_3d[card.keyword] = (card.value, card.comment)

    freq_ax = _find_spectral_axis(ref_header_full)
    freq_suffix = str(freq_ax)
    header_3d["CTYPE3"] = ref_header_full.get(f"CTYPE{freq_suffix}", "FREQ")
    header_3d["CRPIX3"] = 1.0
    header_3d["CRVAL3"] = freqs_sorted[0]
    header_3d["CUNIT3"] = ref_header_full.get(f"CUNIT{freq_suffix}", "Hz")

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
        val = ref_header_full.get(key)
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


_FREQ_DIR_RE = re.compile(r"(\d+)\s*MHz", re.IGNORECASE)


def group_pipeline_files(
    file_paths: list[str | Path],
) -> dict[float, list[Path]]:
    """Group OVRO-LWA pipeline FITS files by frequency.

    Parses the directory structure produced by the OVRO-LWA imaging
    pipeline, where each file lives under a path like::

        /lustre/pipeline/images/{lst}/{date}/Run_{id}/{freq}MHz/I/deep/{name}.fits

    The frequency is extracted from the ``{freq}MHz`` directory
    component.  Files at the same frequency (but different LSTs, dates,
    or runs) are grouped together so they can be coadded by
    :func:`combine_fits_to_spectral_cube` or :func:`coadd_fits`.

    Parameters
    ----------
    file_paths : list of str or Path
        Paths to FITS files produced by the pipeline.

    Returns
    -------
    groups : dict[float, list[Path]]
        Mapping from frequency in Hz to the list of files at that
        frequency, sorted by ascending frequency.

    Raises
    ------
    ValueError
        If no ``{freq}MHz`` directory component can be found in a path.
    """
    groups: dict[float, list[Path]] = {}
    for fpath in file_paths:
        p = Path(fpath)
        freq_hz: float | None = None
        for part in p.parts:
            m = _FREQ_DIR_RE.fullmatch(part)
            if m:
                freq_hz = float(m.group(1)) * 1e6
                break
        if freq_hz is None:
            msg = f"Cannot determine frequency from path: {fpath}"
            raise ValueError(msg)
        groups.setdefault(freq_hz, []).append(p)

    return dict(sorted(groups.items()))


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
    :func:`combine_fits_to_spectral_cube` to assemble a 3-D spectral
    cube and then passes it to :func:`reproject.hips.reproject_to_hips`
    to generate a HiPS3D tile set.

    Each input file is expected to contain a 4-D FITS image with axes
    ``(RA, Dec, Freq, Stokes)`` where frequency and Stokes are both
    length 1.  See :func:`combine_fits_to_spectral_cube` for details
    on the two accepted forms of *file_paths*.

    Parameters
    ----------
    file_paths : list or dict
        Paths to the FITS files.  Accepts the same forms as
        :func:`combine_fits_to_spectral_cube`: a flat list (one file
        per channel, auto-groups duplicates) or a
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
        :func:`combine_fits_to_spectral_cube`.
    min_elevation : float or None, optional
        Minimum elevation in degrees.  Passed through to
        :func:`combine_fits_to_spectral_cube` for per-channel coadding.
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

    index_src = resources.files("lwa_healpix") / "data" / "index.html"
    shutil.copy2(index_src, output_directory / "index.html")
