import numpy as np

import cartopy.crs as ccrs
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

    The unstructued grid can be specified either by passing a
    :py:class:`mosaic.Descriptor` object as the second parameter, or by passing the
    mesh datatset. See  :py:class:`mosaic.Descriptor` for an explanation of what
    the ``mesh_dataset`` has to be. 

    Parameters:
        ax : 
            An Axes or GeoAxes where the pseduocolor plot will be added

        descriptor : :py:class:`Descriptor`
            An already created ``Descriptor`` object

        c : :py:class:`xarray.DataArray`
            The color values to plot. Must have a dimension named either
            ``nCells``, ``nEdges``, or ``nVertices``.
        
        other_parameters
            All other parameters including the ``kwargs`` are the same as 
            for :py:func:`matplotlib.pyplot.pcolor`. 
    """

    # Do some special handling to deal with wrapping/interpolation
    collection = _wrap_polypcolor(ax, descriptor, c,
                                  alpha, norm, cmap, vmin, vmax,
                                  facecolors, transform, **kwargs)

    # Update the datalim for this polycollection
    limits = collection.get_datalim(ax.transData)
    ax.update_datalim(limits)
    ax.autoscale_view()

    return collection

def _wrap_polypcolor(ax, descriptor, c, alpha, norm, cmap, vmin, vmax,
                     facecolors, transform, **kwargs):
    
    transform = descriptor.get_transform()

    if "nCells" in c.dims:
        verts = descriptor.cell_patches
        boundary_mask = descriptor._boundaryCells

    elif "nEdges" in c.dims:
        verts = descriptor.edge_patches

    elif "nVertices" in c.dims:
        verts = descriptor.vertex_patches
    
    # in this case there are no patches that cross boundary 
    if np.all(boundary_mask == False):
        
        collection = PolyCollection(verts, alpha=alpha, array=c, closed=True,
                                    cmap=cmap, **kwargs)
        
        collection.set_array(c)
        collection.set_transform(transform)
        collection._scale_norm(norm, vmin, vmax)
    
        ax.add_collection(collection)

        return collection

    # first create the collection on interior cells
    collection = PolyCollection(verts[~boundary_mask],
                                alpha=alpha,
                                array=c[~boundary_mask],
                                closed=True,
                                cmap=cmap,
                                **kwargs)
    
    collection.set_array(c[~boundary_mask])
    collection.set_transform(transform)

    ax.add_collection(collection)
    
    # transform the boundary cells to Geodetic crs
    geo = ccrs.Geodetic()
    
    geo_patches = geo.transform_points(transform,
                                       verts[boundary_mask,:,0],
                                       verts[boundary_mask,:,1])[...,0:2] 

    # first create the collection on interior cells
    _geo_collection = PolyCollection(geo_patches,
                                     alpha=alpha,
                                     array=c[boundary_mask],
                                     closed=True,
                                     cmap=cmap,
                                     **kwargs)
    
    _geo_collection.set_array(c[boundary_mask])
    _geo_collection.set_transform(geo)

    ax.add_collection(_geo_collection)

    return collection
