from __future__ import annotations

import os
from pathlib import Path

# for the writable data directory (i.e. the one where new data goes), follow
# the XDG guidelines found at
# https://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html
_writable_dir = Path.home() / ".local" / "share"
_data_dir = Path(os.environ.get("XDG_DATA_HOME", _writable_dir)) / "mosaic"

config = {
    "pre_existing_data_dir": Path(os.environ.get("CARTOPY_DATA_DIR", "")),
    "data_dir": _data_dir,
    "downloaders": {},
}

from mosaic import datasets
from mosaic.coastlines import coastlines
from mosaic.contour import contour, contourf
from mosaic.descriptor import Descriptor
from mosaic.polypcolor import polypcolor

import mosaic.feature

__all__ = [
    "Descriptor",
    "coastlines",
    "contour",
    "contourf",
    "datasets",
    "polypcolor",
]
