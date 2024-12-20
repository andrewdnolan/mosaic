from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike
from xarray.core.dataarray import DataArray

from mosaic.descriptor import Descriptor


def polypcolor(
    ax: Axes,
    descriptor: Descriptor,
    c: DataArray,
    alpha: float = 1.0,
    norm: str | Normalize | None = None,
    cmap: str | Normalize | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    facecolors: ArrayLike | None = None,
    **kwargs
) -> PolyCollection:
    """
    Create a pseudocolor plot of a unstructured MPAS grid.

    The unstructured grid is specified by passing a
    :py:class:`~mosaic.Descriptor` object as the second parameter.
    See :py:class:`mosaic.Descriptor` for an explanation of what the
    ``Descriptor`` is and how to construct it.

    Parameters
    ----------

    ax : matplotlib axes object
        Axes, or GeoAxes, on which to plot

    descriptor : :py:class:`Descriptor`
        An already created ``Descriptor`` object

    c : DataArray
        The color values to plot. Must have a dimension named either
        ``nCells``, ``nEdges``, or ``nVertices``.

    other_parameters
        All other parameters are the same as for
        :py:func:`~matplotlib.pyplot.pcolor`.
    """

    if "nCells" in c.dims:
        verts = descriptor.cell_patches

    elif "nEdges" in c.dims:
        verts = descriptor.edge_patches

    elif "nVertices" in c.dims:
        verts = descriptor.vertex_patches

    transform = descriptor.transform

    collection = PolyCollection(verts, alpha=alpha, array=c,
                                cmap=cmap, norm=norm, **kwargs)

    # only set the transform if GeoAxes
    if isinstance(ax, GeoAxes):
        collection.set_transform(transform)

    collection._scale_norm(norm, vmin, vmax)
    ax.add_collection(collection)

    # Update the datalim for this polycollection
    limits = collection.get_datalim(ax.transData)
    # TODO: account for nodes of patches that lie outside of projection bounds
    #       (i.e. as a result of patch wrapping at the antimeridian)
    ax.update_datalim(limits)
    ax.autoscale_view()

    return collection
