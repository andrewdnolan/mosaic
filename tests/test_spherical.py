from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

import mosaic
import mosaic.utils

mpl.use("Agg", force=True)


class TestSphericalWrapping:
    ds = mosaic.datasets.open_dataset("QU.240km")

    @pytest.mark.timeout(30)
    @pytest.mark.parametrize("patch", ["Cell", "Edge", "Vertex"])
    def test_timeout(self, iterate_supported_projections, tmp_path, patch):
        # do the test setup with the parameterized projection
        descriptor = iterate_supported_projections(self.ds)

        projection = descriptor.projection

        # get the projection name
        proj_name = type(projection).__name__

        # setup with figure with the parameterized projection
        fig, ax = plt.subplots(subplot_kw={"projection": projection})

        # get the appropriate dataarray for the parameterized patch location
        da = self.ds[f"indexTo{patch}ID"]

        # just testing that this doesn't hang, not for correctness
        mosaic.polypcolor(ax, descriptor, da, antialiaseds=True)

        # save the figure so that patches are rendered
        fig.savefig(f"{tmp_path}/{proj_name}-{patch}.png")
        plt.close()

    @pytest.mark.parametrize("patch", ["Cell", "Edge", "Vertex"])
    def test_valid_patches(self, iterate_supported_projections, patch):
        # do the test setup with the parameterized projection
        descriptor = iterate_supported_projections(self.ds)

        # extract the patches
        patches = descriptor.__getattribute__(f"{patch.lower()}_patches")

        # get list of invalid patches
        invalid = mosaic.utils.get_invalid_patches(patches)

        # assert that all the patches are valid
        assert invalid is None
