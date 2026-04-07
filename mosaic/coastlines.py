from __future__ import annotations

import warnings

import cartopy.feature as cfeature
import numpy as np
import shapely
from cartopy.mpl.geoaxes import GeoAxes

from mosaic.contour import ContourGraph, MPASContourGenerator
from mosaic.descriptor import Descriptor


def coastlines(
    ax: GeoAxes, descriptor: Descriptor, color: str = "black", **kwargs
) -> None:
    """
    Plot coastal **outlines** using the connectivity info from the MPAS mesh

    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The cartopy axes to add the coastlines to.
    descriptor : Descriptor
        The descriptor containing the projection and dataset information.
    **kwargs
       Additional keyword arguments. See
       :py:class:`matplotlib.collections.Collection` for supported options.
    """

    if not isinstance(ax, GeoAxes):
        msg = (
            "Must provide a `cartopy.mpl.geoaxes` instance for "
            "`mosaic.coastlines` to work. "
        )
        raise TypeError(msg)

    if "edgecolor" in kwargs and "ec" in kwargs:
        msg = "Cannot specify both 'edgecolor' and 'ec'."
        raise TypeError(msg)
    if "edgecolor" in kwargs:
        color = kwargs.pop("edgecolor")
    elif "ec" in kwargs:
        color = kwargs.pop("ec")

    if "facecolor" in kwargs and "fc" in kwargs:
        msg = "Cannot specify both 'facecolor' and 'fc'."
        raise TypeError(msg)
    if "facecolor" in kwargs or "fc" in kwargs:
        warnings.warn(
            "'facecolor (fc)' is not supported for `mosaic.coastlines` "
            "and will be ignored.",
            stacklevel=2,
        )
        kwargs.pop("facecolor", None)
        kwargs.pop("fc", None)

    kwargs["edgecolor"] = color
    kwargs["facecolor"] = "none"

    generator = MPASCoastlineGenerator(descriptor)
    coastlines = generator.create_coastlines()

    geometires = shapely.GeometryCollection(
        [shapely.LineString(cl) for cl in coastlines]
    )

    feature = cfeature.ShapelyFeature(geometires, descriptor.projection)
    ax.add_feature(feature, **kwargs)


class MPASCoastlineGenerator(MPASContourGenerator):
    def __init__(self, descriptor: Descriptor):
        # pass a dummy field to the parent class
        super().__init__(descriptor, descriptor.ds.nCells)

        self.domain = descriptor.projection.domain
        self.boundary = descriptor.projection.boundary

        shapely.prepare(self.domain)

    def create_coastlines(self) -> np.ndarray:
        graph = self._create_coastline_graph()
        lines = self._split_and_order_graph(graph)

        return self._snap_lines_to_boundary(lines)

    def _create_coastline_graph(self) -> ContourGraph:
        edge_mask = (self.ds.cellsOnEdge == -1).any("TWO").values

        vertex_1 = self.ds.verticesOnEdge[edge_mask].isel(TWO=0).values
        vertex_2 = self.ds.verticesOnEdge[edge_mask].isel(TWO=1).values

        return ContourGraph(vertex_1, vertex_2)

    def _snap_lines_to_boundary(
        self, lines: list[np.ndarray]
    ) -> list[np.ndarray]:
        def snap(point: np.ndarray):
            return self.boundary.interpolate(
                self.boundary.project(shapely.Point(point))
            )

        complete_lines = []
        for line in lines:
            # only need to snap lines that are not already closed loops
            if np.array_equal(line[0], line[-1]):
                complete_lines.append(line)
                continue

            contain_mask = shapely.contains_xy(self.domain, *line.T)
            if not contain_mask.any():
                continue

            clipped = line[contain_mask]

            if len(clipped) == 1:
                # if only one point inside domain,
                # all snapped points will lie along the same line
                continue

            # TODO: For coastlines with end points outside domain it would be
            # better to cut at boundary intersection point rather than snapping
            p0, p1 = snap(clipped[0]), snap(clipped[-1])

            complete_lines.append(
                np.concatenate([np.array(p0.xy).T, clipped, np.array(p1.xy).T])
            )

        return complete_lines
