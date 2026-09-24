"""Tools for HEALPix reprojection and coadding of OVRO-LWA images."""

__version__ = "0.1.0"

from .coadd import (
    coadd_fits,
    combine_fits_to_spectral_cube,
    screen_fits_by_quality,
    temporal_std_healpix,
)
from .healpix_io import read_healpix_fits, write_healpix_fits
from .healpix_wcs import (
    healpix_to_hdu,
    iter_nested_tile_headers,
    nested_tile_header,
    pixel_scale_deg_for_nside,
    reproject_healpix_to_wcs,
)
from .hips import fits_to_hips, fits_to_hips_cube, healpix_to_hips, upgrade_hips3d
from .utils import (
    OVRO_LATITUDE_DEG,
    altaz_to_ha_dec,
    center_patch_rms_from_fits,
    elliptical_mask_edge_zenith_angle_deg,
    group_pipeline_files,
    lst_hour_from_path,
    parallactic_angle_deg,
    parallactic_delta_q_edge_summary,
)
