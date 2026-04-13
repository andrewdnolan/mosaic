from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import shapefile

from cartopy.io import Downloader, shapereader
from cartopy.io.shapereader import NEShpDownloader
from shapely.geometry import shape as shapely_shape
from shapely.ops import transform as shapely_transform

from mosaic import config

_NATURAL_EARTH_GEOM_CACHE = {}


class NaturalEarthFeature(cfeature.Feature):
    """
    A simple interface to Natural Earth shapefiles.

    See https://www.naturalearthdata.com/

    """

    def __init__(self, category, name, scale, **kwargs):
        """
        Parameters
        ----------
        category
            The category of the dataset, i.e. either 'cultural' or 'physical'.
        name
            The name of the dataset, e.g. 'admin_0_boundary_lines_land'.
        scale
            The dataset scale, i.e. one of '10m', '50m', or '110m',
            or Scaler object. Dataset scales correspond to 1:10,000,000,
            1:50,000,000, and 1:110,000,000 respectively.

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments to be used when drawing this feature.

        """
        super().__init__(
            ccrs.PlateCarree(globe=ccrs.Globe(ellipse="sphere")), **kwargs
        )
        self.category = category
        self.name = name

        # Cast the given scale to a (constant) Scaler if a string is passed.
        if isinstance(scale, str):
            scale = cfeature.Scaler(scale)

        self.scaler = scale
        # Make sure this is a valid resolution
        self._validate_scale()

    @property
    def scale(self):
        return self.scaler.scale

    def _validate_scale(self):
        if self.scale not in ("110m", "50m", "10m"):
            msg = (
                f"{self.scale!r} is not a valid Natural Earth scale. "
                'Valid scales are "110m", "50m", and "10m".'
            )
            raise ValueError(msg)

    def geometries(self):
        """
        Returns an iterator of (shapely) geometries for this feature.

        """
        key = (self.name, self.category, self.scale)
        if key not in _NATURAL_EARTH_GEOM_CACHE:
            path = natural_earth(
                resolution=self.scale, category=self.category, name=self.name
            )
            reader = shapereader.Reader(path)
            geometries = tuple(reader.geometries())
            _NATURAL_EARTH_GEOM_CACHE[key] = geometries
        else:
            geometries = _NATURAL_EARTH_GEOM_CACHE[key]

        return iter(geometries)

    def intersecting_geometries(self, extent):
        """
        Returns an iterator of shapely geometries that intersect with
        the given extent.
        The extent is assumed to be in the CRS of the feature.
        If extent is None, the method returns all geometries for this dataset.
        """
        self.scaler.scale_from_extent(extent)
        return super().intersecting_geometries(extent)

    def with_scale(self, new_scale):
        """
        Return a copy of the feature with a new scale.

        Parameters
        ----------
        new_scale
            The new dataset scale, i.e. one of '10m', '50m', or '110m'.
            Corresponding to 1:10,000,000, 1:50,000,000, and 1:110,000,000
            respectively.

        """
        return NaturalEarthFeature(
            self.category, self.name, new_scale, **self.kwargs
        )


def natural_earth(resolution="110m", category="physical", name="coastline"):
    """
    Return the path to a spherical Natural Earth shapefile, downloading
    and reprojecting from WGS84 if necessary.

    Coordinates in the returned shapefile are geocentric (spherical)
    lat/lon rather than WGS84 geodetic lat/lon, matching the coordinate
    system used by MPAS meshes.

    To identify valid components for this function, either browse
    NaturalEarthData.com, or if you know what you are looking for, go to
    https://github.com/nvkelso/natural-earth-vector/ to see the actual
    files which will be downloaded.

    Note
    ----
        Some of the Natural Earth shapefiles have special features which are
        described in the name. For example, the 110m resolution
        "admin_0_countries" data also has a sibling shapefile called
        "admin_0_countries_lakes" which excludes lakes in the country
        outlines. For details of what is available refer to the Natural Earth
        website, and look at the "download" link target to identify
        appropriate names.

    """
    ne_downloader = Downloader.from_config(
        ("shapefiles", "natural_earth_on_sphere", resolution, category, name),
        config_dict=config["downloaders"],
    )

    format_dict = {
        "config": config,
        "category": category,
        "name": name,
        "resolution": resolution,
    }

    return ne_downloader.path(format_dict)


class NESphericalShpDownloader(NEShpDownloader):
    """
    Specialise :class:`cartopy.io.shapereader.NEShpDownloader` to download
    Natural Earth shapefiles and reproject them from WGS84 geodetic
    coordinates to geocentric (spherical) coordinates.

    The reprojected shapefiles are cached separately from Cartopy's standard
    WGS84 cache so both can coexist.
    """

    def acquire_resource(self, target_path, format_dict):
        """
        Obtain the WGS84 Natural Earth shapefile via Cartopy's standard
        downloader, reproject every geometry to spherical coordinates, and
        write the result to *target_path*.

        Parameters
        ----------
        target_path : path-like
            Destination ``.shp`` file for the reprojected shapefile.
        format_dict : dict
            Must contain ``resolution``, ``category``, and ``name`` keys.

        Returns
        -------
        pathlib.Path
            Path to the written ``.shp`` file.
        """
        from cartopy.io.shapereader import (
            natural_earth as cartopy_natural_earth,
        )

        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        src_path = cartopy_natural_earth(
            resolution=format_dict["resolution"],
            category=format_dict["category"],
            name=format_dict["name"],
        )

        spherical = ccrs.PlateCarree(globe=ccrs.Globe(ellipse="sphere"))
        geodetic = ccrs.Geodetic(globe=ccrs.Globe(datum="WGS84"))

        with shapefile.Reader(str(src_path)) as reader:
            with shapefile.Writer(
                str(target_path), shapeType=reader.shapeType
            ) as writer:
                writer.fields = list(reader.fields[1:])  # skip deletion flag
                for shape_rec in reader.iterShapeRecords():
                    if shape_rec.shape.shapeType == shapefile.NULL:
                        writer.null()
                        writer.record(*shape_rec.record)
                        continue
                    geom = shapely_shape(shape_rec.shape)
                    new_geom = shapely_transform(
                        _transform_fn_factory(spherical, geodetic), geom
                    )
                    writer.shape(new_geom)
                    writer.record(*shape_rec.record)

        target_path.with_suffix(".prj").write_text(spherical.to_wkt())

        return target_path

    @staticmethod
    def default_downloader():
        """
        Return a generic, standard, NESphericalShpDownloader instance.

        Typically, a user will not need to call this staticmethod.
        """
        default_spec = (
            "shapefiles",
            "natural_earth_on_sphere",
            "{category}",
            "ne_{resolution}_{name}.shp",
        )
        ne_path_template = str(
            Path("{config[data_dir]}").joinpath(*default_spec)
        )
        pre_path_template = str(
            Path("{config[pre_existing_data_dir]}").joinpath(*default_spec)
        )
        return NESphericalShpDownloader(
            target_path_template=ne_path_template,
            pre_downloaded_path_template=pre_path_template,
        )


# Register the downloader so Downloader.from_config can find it.
_ne_key = ("shapefiles", "natural_earth_on_sphere")
config["downloaders"].setdefault(
    _ne_key, NESphericalShpDownloader.default_downloader()
)


def _transform_fn_factory(target_crs, source_crs):
    """
    Return a function which can be used by ``shapely.op.transform``
    to transform the coordinate points of a geometry.

    The function explicitly *does not* do any interpolation or clever
    transformation of the coordinate points, so there is no guarantee
    that the resulting geometry would make any sense.

    """

    def transform_fn(x, y, z=None):
        new_coords = target_crs.transform_points(
            source_crs, np.asanyarray(x), np.asanyarray(y)
        )
        return new_coords[:, 0], new_coords[:, 1], new_coords[:, 2]

    return transform_fn


BORDERS = NaturalEarthFeature(
    "cultural",
    "admin_0_boundary_lines_land",
    cfeature.auto_scaler,
    edgecolor="black",
    facecolor="never",
)
"""Automatically scaled country boundaries."""


STATES = NaturalEarthFeature(
    "cultural",
    "admin_1_states_provinces_lakes",
    cfeature.auto_scaler,
    edgecolor="black",
    facecolor="none",
)
"""Automatically scaled state and province boundaries."""
