import numpy as np 

from functools import cached_property
from xarray.core.dataset import Dataset  

renaming_dict = {"lonCell": "xCell",
                 "latCell": "yCell",
                 "lonEdge": "xEdge",
                 "latEdge": "yEdge",
                 "lonVertex": "xVertex",
                 "latVertex": "yVertex"}

connectivity_arrays = ["cellsOnEdge",
                       "cellsOnVertex",
                       "verticesOnEdge",
                       "verticesOnCell"]

class Descriptor:
    """
    Class describing unstructured MPAS meshes in order to support plotting
    within ``matplotlib``. The class contains various methods to create 
    :py:class:`matplotlib.collections.PolyCollection` objects for
    variables defined at cell centers, vertices, and edges.
    

    Attributes
    ----------
    latlon : boolean
        Whethere to use the lat/lon coordinates in patch construction

        NOTE: I don't think this is needed if the projection arg is
              properly used at initilaization 

    projection : :py:class:`cartopy.crs.CRS`

    transform : :py:class:`cartopy.crs.CRS`

    cell_patches : :py:class:`numpy.ndarray`

    edge_patches : :py:class:`numpy.ndarray`

    vertex_patches : :py:class:`numpy.ndarray`
    """
    def __init__(self, ds, projection=None, transform=None, use_latlon=True): 
        """
        """
        self.latlon     = use_latlon
        self.projection = projection
        self.transform  = transform
        self._pre_projected = False
    
        # if mesh is on a sphere, force the use of lat lon coords
        if ds.attrs["on_a_sphere"].strip().upper() == 'YES':
            self.latlon = True
        # also check if projection requires lat/lon coords
        
        # create a minimal dataset, stored as an attr, for patch creation
        self.ds = self.create_minimal_dataset(ds) 
        
        # reproject the minimal dataset, even for non-spherical meshes
        if projection and transform: 
            self._transform_coordinates(projection, transform)
            self._pre_projected = True

    def create_minimal_dataset(self, ds): 
        """
        Create a xarray.Dataset that contains the minimal subset of 
        coordinate / connectivity arrays needed to create pathces for plotting
        """
        
        if self.latlon:
            coordinate_arrays = list(renaming_dict.keys())
        else:
            coordinate_arrays = list(renaming_dict.values())

        # list of coordinate / connectivity arrays needed to create patches
        mesh_arrays = coordinate_arrays + connectivity_arrays
        
        # get the subset of arrays from the mesh dataset
        minimal_ds = ds[mesh_arrays]

        # delete the attributes in the minimal dataset to avoid confusion
        minimal_ds.attrs.clear()
    
        # should zero index the connectivity arrays here. 

        if self.latlon:

            # convert lat/lon coordinates from radian to degrees
            for loc in ["Cell", "Edge", "Vertex"]:
                minimal_ds[f"lon{loc}"] = np.rad2deg(minimal_ds[f"lon{loc}"])
                minimal_ds[f"lat{loc}"] = np.rad2deg(minimal_ds[f"lat{loc}"])
            
            # rename the coordinate arrays to all be named x.../y...
            # irrespective of whether spherical or cartesian coords are used
            minimal_ds = minimal_ds.rename(renaming_dict)

        return minimal_ds

    @cached_property
    def cell_patches(self):
        patches = _compute_cell_patches(self.ds)
        patches = self._fix_antimeridian(patches, "Cell")
        return patches

    @cached_property
    def edge_patches(self):
        patches = _compute_edge_patches(self.ds)
        patches = self._fix_antimeridian(patches, "Edge")
        return patches

    @cached_property
    def vertex_patches(self):
        patches = _compute_vertex_patches(self.ds)
        patches = self._fix_antimeridian(patches, "Vertex")
        return patches

    def get_transform(self):

        if self._pre_projected:
            transform = self.projection
        else:
            transform = self.transform
        
        return transform

    def _transform_coordinates(self, projection, transform):
        """
        """

        for loc in ["Cell", "Edge", "Vertex"]:

            transformed_coords = projection.transform_points(transform,
                self.ds[f"x{loc}"], self.ds[f"y{loc}"])
           
            # transformed_coords is a np array so need to assign to the values
            self.ds[f"x{loc}"].values = transformed_coords[:, 0]
            self.ds[f"y{loc}"].values = transformed_coords[:, 1]
    
    def _fix_antimeridian(self, patches, loc, projection=None): 
        """Correct vertices of patches that cross the antimeridian. 

        NOTE: Can this be a decorator? 
        """
        # coordinate arrays are transformed at initalization, so using the 
        # transform size limit, not the projection 
        if not projection: 
            projection = self.projection

        # should be able to come up with a default size limit here, or maybe
        # it's already an attribute(?) Should also factor in a precomputed
        # axis period, as set in the attributes of the input dataset
        if projection: 
            # convert to numpy array to that broadcasting below will work
            x_center = np.array(self.ds[f"x{loc}"])

            # get distance b/w the center and vertices of the patches
            # NOTE: using data from masked patches array so that we compute
            #       mask only corresponds to patches that cross the boundary, 
            #       (i.e. NOT a mask of all invalid cells). May need to be 
            #       carefull about the fillvalue depending on the transform
            half_distance = x_center[:, np.newaxis] - patches[...,0].data

            period = np.abs(projection.x_limits[1] - projection.x_limits[0])

            # get the size limit of the projection; 
            size_limit = period / (2 * np.sqrt(2))
    
            # left and right mask, with same number of dims as the patches
            l_mask = (half_distance >= size_limit)
            r_mask = (half_distance < -size_limit)

            self.l_mask = l_mask
            self.r_mask = r_mask
            
            patches[l_mask, 0] += period
            patches[r_mask, 0] -= period

        return patches

    def transform_patches(self, patches, projection, transform):
        """
        """

        raise NotImplementedError("This is a place holder. Do not use.")
         
        transformed_patches = projection.transform_points(transform,
            patches[..., 0], patches[..., 1])
    
        # transformation will return x,y,z values. Only need x and y
        patches.data[...] = transformed_patches[..., 0:2] 

        return patches

def _compute_cell_patches(ds):
    
    maxEdges = ds.sizes["maxEdges"]
    
    # insert the first vertex as the last, so that all polygons are closed
    verticesOnCell = np.hstack((ds.verticesOnCell, ds.verticesOnCell[:,0:1]))

    # get a mask of the active vertices
    mask = verticesOnCell == 0
    
    # get the coordinates needed to patch construction
    xVertex = ds.xVertex.values
    yVertex = ds.yVertex.values
    
    # account for zero indexing
    verticesOnCell = verticesOnCell - 1
    # tile the first vertices index
    firstVertex = np.tile(verticesOnCell[:, 0], (maxEdges + 1, 1)).T 
    # set masked cell indicies to the first index of the polygon
    verticesOnCell = np.where(mask, firstVertex, verticesOnCell)

    # reshape/expand the vertices coordinate arrays
    x_vert = xVertex[verticesOnCell]
    y_vert = yVertex[verticesOnCell]

    verts = np.stack((x_vert, y_vert), axis=-1)

    return verts

def _compute_edge_patches(ds, latlon=False):
    
    TWO = ds.sizes["TWO"]

    cellsOnEdge = ds.cellsOnEdge
    verticesOnEdge = ds.verticesOnEdge

    # get a mask of the active vertices
    cellMask = cellsOnEdge == 0
    vertexMask = verticesOnEdge == 0

    # account for zeros indexing
    cellsOnEdge = cellsOnEdge - 1
    verticesOnEdge = verticesOnEdge - 1
    # tile the first vertices index
    firstCellVertex = np.tile(cellsOnEdge[:, 0], (TWO, 1)).T 
    firstVertexVertex = np.tile(verticesOnEdge[:, 0], (TWO, 1)).T 
    # set masked cell indicies to the first index of the polygon
    cellsOnEdge = np.where(cellMask, firstCellVertex, cellsOnEdge)
    verticesOnEdge = np.where(vertexMask, firstVertexVertex, verticesOnEdge)

    # get the coordinates needed to patch construction
    xCell = ds.xCell.values
    yCell = ds.yCell.values
    xVertex = ds.xVertex.values
    yVertex = ds.yVertex.values

    # get subset of cell coordinate arrays corresponding to edge patches
    xCell = xCell[cellsOnEdge]
    yCell = yCell[cellsOnEdge]
    # get subset of vertex coordinate arrays corresponding to edge patches
    xVertex = xVertex[verticesOnEdge]
    yVertex = yVertex[verticesOnEdge]

    # manually insert first vertex as the last, so that polygons are closed
    x_vert = np.stack((xCell[:,0], xVertex[:,0],
                       xCell[:,1], xVertex[:,1],
                       xCell[:,0]), axis=-1)
    
    y_vert = np.stack((yCell[:,0], yVertex[:,0],
                       yCell[:,1], yVertex[:,1],
                       yCell[:,0]), axis=-1)

    verts = np.stack((x_vert, y_vert), axis=-1)

    return verts

def _compute_vertex_patches(ds, latlon=False):
    
    vertexDegree = ds.sizes["vertexDegree"]

    # insert the first vertex as the last, so that all polygons are closed
    cellsOnVertex = np.hstack((ds.cellsOnVertex, ds.cellsOnVertex[:, 0:1]))

    # get a mask of the active vertices
    mask = cellsOnVertex == 0
    
    # get the coordinates needed to patch construction
    xCell = ds.xCell.values
    yCell = ds.yCell.values
    
    # account for zero indexing
    cellsOnVertex = cellsOnVertex - 1
    # tile the first vertices index
    firstVertex = np.tile(cellsOnVertex[:, 0], (vertexDegree + 1, 1)).T 
    # set masked cell indicies to the first index of the polygon
    cellsOnVertex = np.where(mask, firstVertex, cellsOnVertex)

    # reshape/expand the vertices coordinate arrays
    x_vert = xCell[cellsOnVertex]
    y_vert = yCell[cellsOnVertex]

    verts = np.stack((x_vert, y_vert), axis=-1)

    return verts
