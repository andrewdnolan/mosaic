from __future__ import annotations

import inspect

import cartopy.crs as ccrs
import pytest

import mosaic

# get the names, as strings, of unsupported projections for spherical meshes
unsupported = [
    p.__name__ for p in mosaic.descriptor.UNSUPPORTED_SPHERICAL_PROJECTIONS
]

PROJECTIONS = [
    obj()
    for name, obj in inspect.getmembers(ccrs)
    if inspect.isclass(obj)
    and issubclass(obj, ccrs.Projection)
    and not name.startswith("_")  # skip internal classes
    and obj is not ccrs.Projection  # skip base Projection class
    and name not in unsupported  # skip unsupported projections
]


def id_func(projection):
    return type(projection).__name__


@pytest.fixture(scope="module", params=PROJECTIONS, ids=id_func)
def iterate_supported_projections(request):
    def _factory(ds):
        return mosaic.Descriptor(ds, request.param, ccrs.Geodetic())

    return _factory
