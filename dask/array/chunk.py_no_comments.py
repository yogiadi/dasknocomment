
from collections.abc import Container, Iterable, Sequence
from functools import wraps
from toolz import concat
import numpy as np
from . import numpy_compat as npcompat
from ..core import flatten
from ..utils import ignoring
from numbers import Integral
try:
    from numpy import take_along_axis
except ImportError:  # pragma: no cover
    take_along_axis = npcompat.take_along_axis
def keepdims_wrapper(a_callable):
    
    @wraps(a_callable)
    def keepdims_wrapped_callable(x, axis=None, keepdims=None, *args, **kwargs):
        r = a_callable(x, axis=axis, *args, **kwargs)
        if not keepdims:
            return r
        axes = axis
        if axes is None:
            axes = range(x.ndim)
        if not isinstance(axes, (Container, Iterable, Sequence)):
            axes = [axes]
        r_slice = tuple()
        for each_axis in range(x.ndim):
            if each_axis in axes:
                r_slice += (None,)
            else:
                r_slice += (slice(None),)
        r = r[r_slice]
        return r
    return keepdims_wrapped_callable
# Wrap NumPy functions to ensure they provide keepdims.
sum = np.sum
prod = np.prod
min = np.min
max = np.max
argmin = keepdims_wrapper(np.argmin)
nanargmin = keepdims_wrapper(np.nanargmin)
argmax = keepdims_wrapper(np.argmax)
nanargmax = keepdims_wrapper(np.nanargmax)
any = np.any
all = np.all
nansum = np.nansum
nanprod = np.nanprod
nancumprod = np.nancumprod
nancumsum = np.nancumsum
nanmin = np.nanmin
nanmax = np.nanmax
mean = np.mean
with ignoring(AttributeError):
    nanmean = np.nanmean
var = np.var
with ignoring(AttributeError):
    nanvar = np.nanvar
std = np.std
with ignoring(AttributeError):
    nanstd = np.nanstd
def coarsen(reduction, x, axes, trim_excess=False):
    
    # Insert singleton dimensions if they don't exist already
    for i in range(x.ndim):
        if i not in axes:
            axes[i] = 1
    if trim_excess:
        ind = tuple(
            slice(0, -(d % axes[i])) if d % axes[i] else slice(None, None)
            for i, d in enumerate(x.shape)
        )
        x = x[ind]
    # (10, 10) -> (5, 2, 5, 2)
    newshape = tuple(concat([(x.shape[i] // axes[i], axes[i]) for i in range(x.ndim)]))
    return reduction(x.reshape(newshape), axis=tuple(range(1, x.ndim * 2, 2)))
def trim(x, axes=None):
    
    if isinstance(axes, Integral):
        axes = [axes] * x.ndim
    if isinstance(axes, dict):
        axes = [axes.get(i, 0) for i in range(x.ndim)]
    return x[tuple(slice(ax, -ax if ax else None) for ax in axes)]
def topk(a, k, axis, keepdims):
    
    assert keepdims is True
    axis = axis[0]
    if abs(k) >= a.shape[axis]:
        return a
    a = np.partition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    return a[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]
def topk_aggregate(a, k, axis, keepdims):
    
    assert keepdims is True
    a = topk(a, k, axis, keepdims)
    axis = axis[0]
    a = np.sort(a, axis=axis)
    if k < 0:
        return a
    return a[
        tuple(
            slice(None, None, -1) if i == axis else slice(None) for i in range(a.ndim)
        )
    ]
def argtopk_preprocess(a, idx):
    
    return a, idx
def argtopk(a_plus_idx, k, axis, keepdims):
    
    assert keepdims is True
    axis = axis[0]
    if isinstance(a_plus_idx, list):
        a_plus_idx = list(flatten(a_plus_idx))
        a = np.concatenate([ai for ai, _ in a_plus_idx], axis)
        idx = np.concatenate(
            [np.broadcast_to(idxi, ai.shape) for ai, idxi in a_plus_idx], axis
        )
    else:
        a, idx = a_plus_idx
    if abs(k) >= a.shape[axis]:
        return a_plus_idx
    idx2 = np.argpartition(a, -k, axis=axis)
    k_slice = slice(-k, None) if k > 0 else slice(-k)
    idx2 = idx2[tuple(k_slice if i == axis else slice(None) for i in range(a.ndim))]
    return take_along_axis(a, idx2, axis), take_along_axis(idx, idx2, axis)
def argtopk_aggregate(a_plus_idx, k, axis, keepdims):
    
    assert keepdims is True
    a, idx = argtopk(a_plus_idx, k, axis, keepdims)
    axis = axis[0]
    idx2 = np.argsort(a, axis=axis)
    idx = take_along_axis(idx, idx2, axis)
    if k < 0:
        return idx
    return idx[
        tuple(
            slice(None, None, -1) if i == axis else slice(None) for i in range(idx.ndim)
        )
    ]
def arange(start, stop, step, length, dtype):
    res = np.arange(start, stop, step, dtype)
    return res[:-1] if len(res) > length else res
def astype(x, astype_dtype=None, **kwargs):
    return x.astype(astype_dtype, **kwargs)
def view(x, dtype, order="C"):
    if order == "C":
        x = np.ascontiguousarray(x)
        return x.view(dtype)
    else:
        x = np.asfortranarray(x)
        return x.T.view(dtype).T
def slice_with_int_dask_array(x, idx, offset, x_size, axis):
    
    # Needed when idx is unsigned
    idx = idx.astype(np.int64)
    # Normalize negative indices
    idx = np.where(idx < 0, idx + x_size, idx)
    # A chunk of the offset dask Array is a numpy array with shape (1, ).
    # It indicates the index of the first element along axis of the current
    # chunk of x.
    idx = idx - offset
    # Drop elements of idx that do not fall inside the current chunk of x
    idx_filter = (idx >= 0) & (idx < x.shape[axis])
    idx = idx[idx_filter]
    # np.take does not support slice indices
    # return np.take(x, idx, axis)
    return x[tuple(idx if i == axis else slice(None) for i in range(x.ndim))]
def slice_with_int_dask_array_aggregate(idx, chunk_outputs, x_chunks, axis):
    
    # Needed when idx is unsigned
    idx = idx.astype(np.int64)
    # Normalize negative indices
    idx = np.where(idx < 0, idx + sum(x_chunks), idx)
    x_chunk_offset = 0
    chunk_output_offset = 0
    # Assemble the final index that picks from the output of the previous
    # kernel by adding together one layer per chunk of x
    # FIXME: this could probably be reimplemented with a faster search-based
    # algorithm
    idx_final = np.zeros_like(idx)
    for x_chunk in x_chunks:
        idx_filter = (idx >= x_chunk_offset) & (idx < x_chunk_offset + x_chunk)
        idx_cum = np.cumsum(idx_filter)
        idx_final += np.where(idx_filter, idx_cum - 1 + chunk_output_offset, 0)
        x_chunk_offset += x_chunk
        if idx_cum.size > 0:
            chunk_output_offset += idx_cum[-1]
    # np.take does not support slice indices
    # return np.take(chunk_outputs, idx_final, axis)
    return chunk_outputs[
        tuple(
            idx_final if i == axis else slice(None) for i in range(chunk_outputs.ndim)
        )
    ]
