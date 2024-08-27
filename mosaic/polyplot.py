import numpy as np

from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from mosaic.descriptor import Descriptor
from typing import Literal


def polyplot(
    ax: Axes,
    descriptor: Descriptor,
    mesh_type: Literal['primary', 'dual', 'egde'],
    alpha: float = 1.0,
    transform = None,
    **kwargs
    ) -> LineCollection:
    """
    Create wireplot of a unstructured MPAS grid.

    Call signatures::
        
        polypcolor(ax, descriptor, mesh_type, *, ...)

    The unstructued grid can be specified either by passing a `.Descriptor`
    object as the second parameter, or by passing the mesh datatset. See 
    `.Descriptor` for an explination of what the mesh_dataset has to be. 

    Parameters:
        ax : 
            An Axes or GeoAxes where the pseduocolor plot will be added

        descriptor : Descriptor
            An already created `Descriptor` object

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

    collection = LineCollection(verts, alpha=alpha, **kwargs)
    collection.set_transform(transform)
    
    ax.add_collection(collection)

    # Update the datalim for this polycollection
    limits = collection.get_datalim(ax.transData)
    ax.update_datalim(limits)
    ax.autoscale_view()

    return collection
