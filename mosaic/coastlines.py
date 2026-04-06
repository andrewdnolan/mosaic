from __future__ import annotations

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
    Add coastal **outlines** to the current axes using the
    coastline information from the MPAS dataset.

    Parameters
    ----------
    ax : Axes
        The axes to add the coastlines to.
    descriptor : Descriptor
        The descriptor containing the projection and dataset information.
    **kwargs
       Additional keyword arguments to pass to ...
    """

    if not isinstance(ax, GeoAxes):
        msg = (
            "Must provide a `cartopy.mpl.geoaxes` instance for "
            "`mosaic.coastlines` to work. "
        )
        raise TypeError(msg)

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

    def _snap_lines_to_boundary(self, lines: np.ndarray) -> np.ndarray:
        def snap(point: np.ndarray):
            return self.boundary.interpolate(
                self.boundary.project(shapely.Point(point))
            )

        for i, line in enumerate(lines):
            # only need to snap lines that are not already closed loops
            if np.array_equal(line[0], line[-1]):
                continue

            contain_mask = shapely.contains_xy(self.domain, *line.T)
            if not contain_mask.any():
                continue

            clipped = line[contain_mask]
            p0, p1 = snap(clipped[0]), snap(clipped[-1])

            lines[i] = np.concatenate(
                [np.array(p0.xy).T, clipped, np.array(p1.xy).T]
            )

        return lines
