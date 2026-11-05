"""Read large external ONNX tensors without copying through protobuf raw_data."""

import os
import stat
import sys
from math import prod

import numpy as np
from onnx import checker
from onnx.external_data_helper import ExternalDataInfo
from onnx.helper import tensor_dtype_to_np_dtype


def read_external_array(initializer, directory):
    """Return a dense NumPy array, or None to let ONNX load other encodings."""
    resolver = getattr(checker.C, "_resolve_external_data_location", None)
    if resolver is None or sys.byteorder != "little" or initializer.HasField("segment"):
        return None
    dtype = np.dtype(tensor_dtype_to_np_dtype(initializer.data_type))
    if dtype.kind not in "biufc" or any(dim < 0 for dim in initializer.dims):
        return None
    size = prod(initializer.dims) * dtype.itemsize
    if size < 256 * 1024:
        return None

    # Keep ONNX's path checks and validate the opened file before allocating.
    info = ExternalDataInfo(initializer)
    location = resolver(directory, info.location, initializer.name)
    offset = 0 if info.offset is None else info.offset
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(location, flags), "rb", buffering=0) as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"External weight is not a regular single-link file: {initializer.name!r}")
        available = metadata.st_size - offset
        length = available if info.length is None else info.length
        if offset < 0 or length != size or length > available:
            raise ValueError(f"Invalid external weight offset or length: {initializer.name!r}")
        array = np.empty(tuple(initializer.dims), dtype=dtype)
        target = memoryview(array).cast("B")
        source.seek(offset)
        while target:
            read = source.readinto(target)
            if not read:
                raise ValueError(f"Truncated external weight: {initializer.name!r}")
            target = target[read:]
    return array
