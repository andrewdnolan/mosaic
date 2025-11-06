from __future__ import annotations

import numpy as np
from matplotlib import artist, transforms
from matplotlib.collections import PolyCollection


class PolyMeshCollection(PolyCollection):
    """ """

    def __init__(self, verts, antialiased=True, **kwargs):
        kwargs.setdefault("pickradius", 0)
        super().__init__(verts=verts, **kwargs)
        self._bbox = transforms.Bbox.unit()
        self._bbox.update_from_data_xy(verts.reshape(-1, 2))
        self._antialiased = antialiased
        self.set_mouseover(False)

        self._coordinates = verts

    @artist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__class__.__name__, self.get_gid())
        transform = self.get_transform()
        offset_trf = self.get_offset_transform()
        offsets = self.get_offsets()

        if self.have_units():
            xs = self.convert_xunits(offsets[:, 0])
            ys = self.convert_yunits(offsets[:, 1])
            offsets = np.column_stack([xs, ys])

        self.update_scalarmappable()

        if not transform.is_affine:
            coordinates = self._coordinates.reshape((-1, 2))
            coordinates = transform.transform(coordinates)
            coordinates = coordinates.reshape(self._coordinates.shape)
            transform = transforms.IdentityTransform()
        else:
            coordinates = self._coordinates

        if not offset_trf.is_affine:
            offsets = offset_trf.transform_non_affine(offsets)
            offset_trf = offset_trf.get_affine()

        gc = renderer.new_gc()
        gc.set_snap(self.get_snap())
        self._set_gc_clip(gc)
        gc.set_linewidth(self.get_linewidth()[0])

        renderer.draw_poly_mesh(
            gc,
            transform.frozen(),
            coordinates.shape[1] - 1,
            coordinates.shape[0] - 1,
            coordinates,
            offsets,
            offset_trf,
            self.get_facecolor().reshape((-1, 4)),
            self._antialiased,
            self.get_edgecolors().reshape((-1, 4)),
        )

        gc.restore()
        renderer.close_group(self.__class__.__name__)
        self.stale = False


class MPASCollection(PolyMeshCollection):
    """
    A PolyCollection designed to mirror patches across periodic boundaries

    Closely follows ``cartopy.mpl.geocollection.GeoCollection`` implementation.
    """

    def get_array(self):
        # Retrieve the array - use copy to avoid any chance of overwrite
        return super().get_array().copy()

    def set_array(self, A):
        # Only use the mirrored indices if they are there
        if hasattr(self, "_mirrored_idxs"):
            self._mirrored_collection_fix.set_array(A[self._mirrored_idxs])

        # Update array for interior patches using underlying implementation
        super().set_array(A)

    def set_clim(self, vmin=None, vmax=None):
        # Update _mirrored_collection_fix color limits if it is there.
        if hasattr(self, "_mirrored_collection_fix"):
            self._mirrored_collection_fix.set_clim(vmin, vmax)

        # Update color limits for the rest of the cells.
        super().set_clim(vmin, vmax)

    def get_datalim(self, transData):
        # TODO: Return corners that were calculated in the polypcolor routine
        # (i.e.: return self._corners). In for the datalims to ignore the
        # extent of mirrored patches.
        return super().get_datalim(transData)
