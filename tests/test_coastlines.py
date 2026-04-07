from __future__ import annotations

from unittest.mock import patch

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
import shapely

import mosaic
from mosaic.coastlines import MPASCoastlineGenerator

mpl.use("Agg", force=True)


class TestCoastlineKwargs:
    ds = mosaic.datasets.open_dataset("QU.240km")
    descriptor = mosaic.Descriptor(ds, ccrs.Orthographic(), ccrs.Geodetic())

    def _capture_add_feature_kwargs(self, **kwargs):
        fig, ax = plt.subplots(
            subplot_kw={"projection": self.descriptor.projection}
        )
        try:
            with patch.object(ax, "add_feature") as mock_add:
                mosaic.coastlines(ax, self.descriptor, **kwargs)
                return mock_add.call_args.kwargs
        finally:
            plt.close(fig)

    def test_default_color(self):
        kwargs = self._capture_add_feature_kwargs()
        assert kwargs["edgecolor"] == "black"
        assert kwargs["facecolor"] == "none"

    def test_color_parameter(self):
        kwargs = self._capture_add_feature_kwargs(color="red")
        assert kwargs["edgecolor"] == "red"

    def test_edgecolor_aliases_override_color(self):
        kwargs = self._capture_add_feature_kwargs(
            color="black",
            edgecolor="red",
        )
        assert kwargs["edgecolor"] == "red"
        assert "ec" not in kwargs

    def test_ec_aliases_override_color(self):
        kwargs = self._capture_add_feature_kwargs(
            color="black",
            ec="blue",
        )
        assert kwargs["edgecolor"] == "blue"
        assert "ec" not in kwargs

    def test_edgecolor_and_ec_raises(self):
        fig, ax = plt.subplots(
            subplot_kw={"projection": self.descriptor.projection}
        )
        try:
            with pytest.raises(
                TypeError, match="Cannot specify both 'edgecolor' and 'ec'"
            ):
                mosaic.coastlines(
                    ax, self.descriptor, edgecolor="red", ec="blue"
                )
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("name", ["facecolor", "fc"])
    def test_facecolor_aliases_warn_and_are_ignored(self, name):
        with pytest.warns(
            UserWarning,
            match=r"'facecolor \(fc\)' is not supported for",
        ):
            kwargs = self._capture_add_feature_kwargs(**{name: "red"})
        assert kwargs["facecolor"] == "none"
        assert "fc" not in kwargs

    def test_facecolor_and_fc_raises(self):
        fig, ax = plt.subplots(
            subplot_kw={"projection": self.descriptor.projection}
        )
        try:
            with pytest.raises(
                TypeError, match="Cannot specify both 'facecolor' and 'fc'"
            ):
                mosaic.coastlines(
                    ax, self.descriptor, facecolor="red", fc="blue"
                )
        finally:
            plt.close(fig)


class TestCoastlines:
    ds = mosaic.datasets.open_dataset("QU.240km")

    def test_coastlines_are_simple(self, iterate_supported_projections):
        # do the test setup with the parameterized projection
        descriptor = iterate_supported_projections(self.ds)

        generator = MPASCoastlineGenerator(descriptor)
        coastlines = generator.create_coastlines()

        geometries = shapely.GeometryCollection(
            [shapely.LineString(cl) for cl in coastlines]
        )

        assert all(line.is_simple for line in geometries.geoms)
