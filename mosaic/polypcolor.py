import numpy as np

from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, Colormap
from mosaic.descriptor import Descriptor
from numpy.typing import ArrayLike
from xarray.core.dataarray import DataArray


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
    transform = None,
    **kwargs
    ) -> PolyCollection:
    """
    Create a pseudocolor plot of a unstructured MPAS grid.

    Call signatures::
        
        polypcolor(ax, descriptor, c, *, ...)

    The unstructued grid can be specified either by passing a `.Descriptor`
    object as the second parameter, or by passing the mesh datatset. See 
    `.Descriptor` for an explination of what the mesh_dataset has to be. 

    Parameters:
        ax : 
            An Axes or GeoAxes where the pseduocolor plot will be added

        descriptor : Descriptor
            An already created `Descriptor` object

        c : xarray.DataArray
            The color values to plot. Must have a dimension named either
            `nCells`, `nEdges`, or `nVertices`.
        
        other_parameters
            All other parameters including the `kwargs` are the same as 
            for `matplotlib`'s `pcolor`. See `pcolor`'s documentation for 
            definitions
    """

    if "nCells" in c.dims:
        verts = descriptor.cell_patches

    elif "nEdges" in c.dims:
        verts = descriptor.edge_patches

    elif "nVertices" in c.dims:
        verts = descriptor.vertex_patches
    
    transform = descriptor.get_transform()

    collection = PolyCollection(verts, alpha=alpha, array=c, closed=True,
                                cmap=cmap, **kwargs)
    
    collection.set_array(c)
    collection.set_transform(transform)
    collection._scale_norm(norm, vmin, vmax)
    
    ax.add_collection(collection)

    # Update the datalim for this polycollection
    limits = collection.get_datalim(ax.transData)
    ax.update_datalim(limits)
    ax.autoscale_view()

    return collection
