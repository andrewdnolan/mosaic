from __future__ import annotations

import shapely

import mosaic
from mosaic.coastlines import MPASCoastlineGenerator


class TestCoastlines:
    ds = mosaic.datasets.open_dataset("QU.240km")

    def test_coastlines_are_simple(self, iterate_supported_projections):
        # do the test setup with the parameterized projection
        descriptor = iterate_supported_projections(self.ds)

        generator = MPASCoastlineGenerator(descriptor)
        coastlines = generator.create_coastlines()

        geometires = shapely.GeometryCollection(
            [shapely.LineString(cl) for cl in coastlines]
        )

        assert all(line.is_simple for line in geometires.geoms)
