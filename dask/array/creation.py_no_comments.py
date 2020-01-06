from collections.abc import Sequence
from functools import partial, reduce
from itertools import product
from operator import add, getitem
from numbers import Integral, Number
import numpy as np
from toolz import accumulate, sliding_window
from ..highlevelgraph import HighLevelGraph
from ..base import tokenize
from ..utils import derived_from
from . import chunk
from .core import (
    Array,
    asarray,
    normalize_chunks,
    stack,
    concatenate,
    block,
    broadcast_to,
    broadcast_arrays,
    cached_cumsum,
)
from .wrap import empty, ones, zeros, full
from .utils import AxisError, meta_from_array, zeros_like_safe
def empty_like(a, dtype=None, chunks=None):
    
    a = asarray(a, name=False)
    return empty(
        a.shape,
        dtype=(dtype or a.dtype),
        chunks=(chunks if chunks is not None else a.chunks),
    )
def ones_like(a, dtype=None, chunks=None):
    
    a = asarray(a, name=False)
    return ones(
        a.shape,
        dtype=(dtype or a.dtype),
        chunks=(chunks if chunks is not None else a.chunks),
    )
def zeros_like(a, dtype=None, chunks=None):
    
    a = asarray(a, name=False)
    return zeros(
        a.shape,
        dtype=(dtype or a.dtype),
        chunks=(chunks if chunks is not None else a.chunks),
    )
def full_like(a, fill_value, dtype=None, chunks=None):
    
    a = asarray(a, name=False)
    return full(
        a.shape,
        fill_value,
        dtype=(dtype or a.dtype),
        chunks=(chunks if chunks is not None else a.chunks),
    )
def linspace(
    start, stop, num=50, endpoint=True, retstep=False, chunks="auto", dtype=None
):
    
    num = int(num)
    if dtype is None:
        dtype = np.linspace(0, 1, 1).dtype
    chunks = normalize_chunks(chunks, (num,), dtype=dtype)
    range_ = stop - start
    div = (num - 1) if endpoint else num
    step = float(range_) / div
    name = "linspace-" + tokenize((start, stop, num, endpoint, chunks, dtype))
    dsk = {}
    blockstart = start
    for i, bs in enumerate(chunks[0]):
        bs_space = bs - 1 if endpoint else bs
        blockstop = blockstart + (bs_space * step)
        task = (
            partial(np.linspace, endpoint=endpoint, dtype=dtype),
            blockstart,
            blockstop,
            bs,
        )
        blockstart = blockstart + (step * bs)
        dsk[(name, i)] = task
    if retstep:
        return Array(dsk, name, chunks, dtype=dtype), step
    else:
        return Array(dsk, name, chunks, dtype=dtype)
def arange(*args, **kwargs):
    
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start = args[0]
        stop = args[1]
        step = 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise TypeError(
            
        )
    chunks = kwargs.pop("chunks", "auto")
    num = int(max(np.ceil((stop - start) / step), 0))
    dtype = kwargs.pop("dtype", None)
    if dtype is None:
        dtype = np.arange(start, stop, step * num if num else step).dtype
    chunks = normalize_chunks(chunks, (num,), dtype=dtype)
    if kwargs:
        raise TypeError("Unexpected keyword argument(s): %s" % ",".join(kwargs.keys()))
    name = "arange-" + tokenize((start, stop, step, chunks, dtype))
    dsk = {}
    elem_count = 0
    for i, bs in enumerate(chunks[0]):
        blockstart = start + (elem_count * step)
        blockstop = start + ((elem_count + bs) * step)
        task = (chunk.arange, blockstart, blockstop, step, bs, dtype)
        dsk[(name, i)] = task
        elem_count += bs
    return Array(dsk, name, chunks, dtype=dtype)
@derived_from(np)
def meshgrid(*xi, **kwargs):
    indexing = kwargs.pop("indexing", "xy")
    sparse = bool(kwargs.pop("sparse", False))
    if "copy" in kwargs:
        raise NotImplementedError("`copy` not supported")
    if kwargs:
        raise TypeError("unsupported keyword argument(s) provided")
    if indexing not in ("ij", "xy"):
        raise ValueError("`indexing` must be `'ij'` or `'xy'`")
    xi = [asarray(e) for e in xi]
    xi = [e.flatten() for e in xi]
    if indexing == "xy" and len(xi) > 1:
        xi[0], xi[1] = xi[1], xi[0]
    grid = []
    for i in range(len(xi)):
        s = len(xi) * [None]
        s[i] = slice(None)
        s = tuple(s)
        r = xi[i][s]
        grid.append(r)
    if not sparse:
        grid = broadcast_arrays(*grid)
    if indexing == "xy" and len(xi) > 1:
        grid[0], grid[1] = grid[1], grid[0]
    return grid
def indices(dimensions, dtype=int, chunks="auto"):
    
    dimensions = tuple(dimensions)
    dtype = np.dtype(dtype)
    chunks = normalize_chunks(chunks, shape=dimensions, dtype=dtype)
    if len(dimensions) != len(chunks):
        raise ValueError("Need same number of chunks as dimensions.")
    xi = []
    for i in range(len(dimensions)):
        xi.append(arange(dimensions[i], dtype=dtype, chunks=(chunks[i],)))
    grid = []
    if np.prod(dimensions):
        grid = meshgrid(*xi, indexing="ij")
    if grid:
        grid = stack(grid)
    else:
        grid = empty((len(dimensions),) + dimensions, dtype=dtype, chunks=(1,) + chunks)
    return grid
def eye(N, chunks="auto", M=None, k=0, dtype=float):
    
    eye = {}
    if M is None:
        M = N
    if not isinstance(chunks, (int, str)):
        raise ValueError("chunks must be an int or string")
    elif isinstance(chunks, str):
        chunks = normalize_chunks(chunks, shape=(N, M), dtype=dtype)
        chunks = chunks[0][0]
    token = tokenize(N, chunks, M, k, dtype)
    name_eye = "eye-" + token
    vchunks = [chunks] * (N // chunks)
    if N % chunks != 0:
        vchunks.append(N % chunks)
    hchunks = [chunks] * (M // chunks)
    if M % chunks != 0:
        hchunks.append(M % chunks)
    for i, vchunk in enumerate(vchunks):
        for j, hchunk in enumerate(hchunks):
            if (j - i - 1) * chunks <= k <= (j - i + 1) * chunks:
                eye[name_eye, i, j] = (
                    np.eye,
                    vchunk,
                    hchunk,
                    k - (j - i) * chunks,
                    dtype,
                )
            else:
                eye[name_eye, i, j] = (np.zeros, (vchunk, hchunk), dtype)
    return Array(eye, name_eye, shape=(N, M), chunks=(chunks, chunks), dtype=dtype)
@derived_from(np)
def diag(v):
    name = "diag-" + tokenize(v)
    meta = meta_from_array(v, 2 if v.ndim == 1 else 1)
    if isinstance(v, np.ndarray) or (
        hasattr(v, "__array_function__") and not isinstance(v, Array)
    ):
        if v.ndim == 1:
            chunks = ((v.shape[0],), (v.shape[0],))
            dsk = {(name, 0, 0): (np.diag, v)}
        elif v.ndim == 2:
            chunks = ((min(v.shape),),)
            dsk = {(name, 0): (np.diag, v)}
        else:
            raise ValueError("Array must be 1d or 2d only")
        return Array(dsk, name, chunks, meta=meta)
    if not isinstance(v, Array):
        raise TypeError(
            "v must be a dask array or numpy array, got {0}".format(type(v))
        )
    if v.ndim != 1:
        if v.chunks[0] == v.chunks[1]:
            dsk = {
                (name, i): (np.diag, row[i]) for i, row in enumerate(v.__dask_keys__())
            }
            graph = HighLevelGraph.from_collections(name, dsk, dependencies=[v])
            return Array(graph, name, (v.chunks[0],), meta=meta)
        else:
            raise NotImplementedError(
                "Extracting diagonals from non-square chunked arrays"
            )
    chunks_1d = v.chunks[0]
    blocks = v.__dask_keys__()
    dsk = {}
    for i, m in enumerate(chunks_1d):
        for j, n in enumerate(chunks_1d):
            key = (name, i, j)
            if i == j:
                dsk[key] = (np.diag, blocks[i])
            else:
                dsk[key] = (np.zeros, (m, n))
                dsk[key] = (partial(zeros_like_safe, shape=(m, n)), meta)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[v])
    return Array(graph, name, (chunks_1d, chunks_1d), meta=meta)
@derived_from(np)
def diagonal(a, offset=0, axis1=0, axis2=1):
    name = "diagonal-" + tokenize(a, offset, axis1, axis2)
    if a.ndim < 2:
        # NumPy uses `diag` as we do here.
        raise ValueError("diag requires an array of at least two dimensions")
    def _axis_fmt(axis, name, ndim):
        if axis < 0:
            t = ndim + axis
            if t < 0:
                msg = "{}: axis {} is out of bounds for array of dimension {}"
                raise AxisError(msg.format(name, axis, ndim))
            axis = t
        return axis
    axis1 = _axis_fmt(axis1, "axis1", a.ndim)
    axis2 = _axis_fmt(axis2, "axis2", a.ndim)
    if axis1 == axis2:
        raise ValueError("axis1 and axis2 cannot be the same")
    a = asarray(a)
    if axis1 > axis2:
        axis1, axis2 = axis2, axis1
        offset = -offset
    def _diag_len(dim1, dim2, offset):
        return max(0, min(min(dim1, dim2), dim1 + offset, dim2 - offset))
    diag_chunks = []
    chunk_offsets = []
    cum1 = list(cached_cumsum(a.chunks[axis1], initial_zero=True)[:-1])
    cum2 = list(cached_cumsum(a.chunks[axis2], initial_zero=True)[:-1])
    for co1, c1 in zip(cum1, a.chunks[axis1]):
        chunk_offsets.append([])
        for co2, c2 in zip(cum2, a.chunks[axis2]):
            k = offset + co1 - co2
            diag_chunks.append(_diag_len(c1, c2, k))
            chunk_offsets[-1].append(k)
    dsk = {}
    idx_set = set(range(a.ndim)) - set([axis1, axis2])
    n1 = len(a.chunks[axis1])
    n2 = len(a.chunks[axis2])
    for idx in product(*(range(len(a.chunks[i])) for i in idx_set)):
        for i, (i1, i2) in enumerate(product(range(n1), range(n2))):
            tsk = reduce(getitem, idx[:axis1], a.__dask_keys__())[i1]
            tsk = reduce(getitem, idx[axis1 : axis2 - 1], tsk)[i2]
            tsk = reduce(getitem, idx[axis2 - 1 :], tsk)
            k = chunk_offsets[i1][i2]
            dsk[(name,) + idx + (i,)] = (np.diagonal, tsk, k, axis1, axis2)
    left_shape = tuple(a.shape[i] for i in idx_set)
    right_shape = (_diag_len(a.shape[axis1], a.shape[axis2], offset),)
    shape = left_shape + right_shape
    left_chunks = tuple(a.chunks[i] for i in idx_set)
    right_shape = (tuple(diag_chunks),)
    chunks = left_chunks + right_shape
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[a])
    meta = meta_from_array(a, len(shape))
    return Array(graph, name, shape=shape, chunks=chunks, meta=meta)
def triu(m, k=0):
    
    if m.ndim != 2:
        raise ValueError("input must be 2 dimensional")
    if m.chunks[0][0] != m.chunks[1][0]:
        msg = (
            "chunks must be a square. "
            "Use .rechunk method to change the size of chunks."
        )
        raise NotImplementedError(msg)
    rdim = len(m.chunks[0])
    hdim = len(m.chunks[1])
    chunk = m.chunks[0][0]
    token = tokenize(m, k)
    name = "triu-" + token
    dsk = {}
    for i in range(rdim):
        for j in range(hdim):
            if chunk * (j - i + 1) < k:
                dsk[(name, i, j)] = (
                    partial(zeros_like_safe, shape=(m.chunks[0][i], m.chunks[1][j])),
                    m._meta,
                )
            elif chunk * (j - i - 1) < k <= chunk * (j - i + 1):
                dsk[(name, i, j)] = (np.triu, (m.name, i, j), k - (chunk * (j - i)))
            else:
                dsk[(name, i, j)] = (m.name, i, j)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[m])
    return Array(graph, name, shape=m.shape, chunks=m.chunks, meta=m)
def tril(m, k=0):
    
    if m.ndim != 2:
        raise ValueError("input must be 2 dimensional")
    if not len(set(m.chunks[0] + m.chunks[1])) == 1:
        msg = (
            "All chunks must be a square matrix to perform lu decomposition. "
            "Use .rechunk method to change the size of chunks."
        )
        raise ValueError(msg)
    rdim = len(m.chunks[0])
    hdim = len(m.chunks[1])
    chunk = m.chunks[0][0]
    token = tokenize(m, k)
    name = "tril-" + token
    dsk = {}
    for i in range(rdim):
        for j in range(hdim):
            if chunk * (j - i + 1) < k:
                dsk[(name, i, j)] = (m.name, i, j)
            elif chunk * (j - i - 1) < k <= chunk * (j - i + 1):
                dsk[(name, i, j)] = (np.tril, (m.name, i, j), k - (chunk * (j - i)))
            else:
                dsk[(name, i, j)] = (
                    partial(zeros_like_safe, shape=(m.chunks[0][i], m.chunks[1][j])),
                    m._meta,
                )
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[m])
    return Array(graph, name, shape=m.shape, chunks=m.chunks, meta=m)
def _np_fromfunction(func, shape, dtype, offset, func_kwargs):
    def offset_func(*args, **kwargs):
        args2 = list(map(add, args, offset))
        return func(*args2, **kwargs)
    return np.fromfunction(offset_func, shape, dtype=dtype, **func_kwargs)
@derived_from(np)
def fromfunction(func, chunks="auto", shape=None, dtype=None, **kwargs):
    chunks = normalize_chunks(chunks, shape, dtype=dtype)
    name = "fromfunction-" + tokenize(func, chunks, shape, dtype, kwargs)
    keys = list(product([name], *[range(len(bd)) for bd in chunks]))
    aggdims = [list(accumulate(add, (0,) + bd[:-1])) for bd in chunks]
    offsets = list(product(*aggdims))
    shapes = list(product(*chunks))
    dtype = dtype or float
    values = [
        (_np_fromfunction, func, shp, dtype, offset, kwargs)
        for offset, shp in zip(offsets, shapes)
    ]
    dsk = dict(zip(keys, values))
    return Array(dsk, name, chunks, dtype=dtype)
@derived_from(np)
def repeat(a, repeats, axis=None):
    if axis is None:
        if a.ndim == 1:
            axis = 0
        else:
            raise NotImplementedError("Must supply an integer axis value")
    if not isinstance(repeats, Integral):
        raise NotImplementedError("Only integer valued repeats supported")
    if -a.ndim <= axis < 0:
        axis += a.ndim
    elif not 0 <= axis <= a.ndim - 1:
        raise ValueError("axis(=%d) out of bounds" % axis)
    if repeats == 1:
        return a
    cchunks = cached_cumsum(a.chunks[axis], initial_zero=True)
    slices = []
    for c_start, c_stop in sliding_window(2, cchunks):
        ls = np.linspace(c_start, c_stop, repeats).round(0)
        for ls_start, ls_stop in sliding_window(2, ls):
            if ls_start != ls_stop:
                slices.append(slice(ls_start, ls_stop))
    all_slice = slice(None, None, None)
    slices = [
        (all_slice,) * axis + (s,) + (all_slice,) * (a.ndim - axis - 1) for s in slices
    ]
    slabs = [a[slc] for slc in slices]
    out = []
    for slab in slabs:
        chunks = list(slab.chunks)
        assert len(chunks[axis]) == 1
        chunks[axis] = (chunks[axis][0] * repeats,)
        chunks = tuple(chunks)
        result = slab.map_blocks(
            np.repeat, repeats, axis=axis, chunks=chunks, dtype=slab.dtype
        )
        out.append(result)
    return concatenate(out, axis=axis)
@derived_from(np)
def tile(A, reps):
    try:
        tup = tuple(reps)
    except TypeError:
        tup = (reps,)
    if any(i < 0 for i in tup):
        raise ValueError("Negative `reps` are not allowed.")
    c = asarray(A)
    if all(tup):
        for nrep in tup[::-1]:
            c = nrep * [c]
        return block(c)
    d = len(tup)
    if d < c.ndim:
        tup = (1,) * (c.ndim - d) + tup
    if c.ndim < d:
        shape = (1,) * (d - c.ndim) + c.shape
    else:
        shape = c.shape
    shape_out = tuple(s * t for s, t in zip(shape, tup))
    return empty(shape=shape_out, dtype=c.dtype)
def expand_pad_value(array, pad_value):
    if isinstance(pad_value, Number):
        pad_value = array.ndim * ((pad_value, pad_value),)
    elif (
        isinstance(pad_value, Sequence)
        and all(isinstance(pw, Number) for pw in pad_value)
        and len(pad_value) == 1
    ):
        pad_value = array.ndim * ((pad_value[0], pad_value[0]),)
    elif (
        isinstance(pad_value, Sequence)
        and len(pad_value) == 2
        and all(isinstance(pw, Number) for pw in pad_value)
    ):
        pad_value = tuple((pad_value[0], pad_value[1]) for _ in range(array.ndim))
    elif (
        isinstance(pad_value, Sequence)
        and len(pad_value) == array.ndim
        and all(isinstance(pw, Sequence) for pw in pad_value)
        and all((len(pw) == 2) for pw in pad_value)
        and all(all(isinstance(w, Number) for w in pw) for pw in pad_value)
    ):
        pad_value = tuple((pw[0], pw[1]) for pw in pad_value)
    else:
        raise TypeError("`pad_value` must be composed of integral typed values.")
    return pad_value
def get_pad_shapes_chunks(array, pad_width, axes):
    
    pad_shapes = [list(array.shape), list(array.shape)]
    pad_chunks = [list(array.chunks), list(array.chunks)]
    for d in axes:
        for i in range(2):
            pad_shapes[i][d] = pad_width[d][i]
            pad_chunks[i][d] = (pad_width[d][i],)
    pad_shapes = [tuple(s) for s in pad_shapes]
    pad_chunks = [tuple(c) for c in pad_chunks]
    return pad_shapes, pad_chunks
def linear_ramp_chunk(start, stop, num, dim, step):
    
    num1 = num + 1
    shape = list(start.shape)
    shape[dim] = num
    shape = tuple(shape)
    dtype = np.dtype(start.dtype)
    result = np.empty(shape, dtype=dtype)
    for i in np.ndindex(start.shape):
        j = list(i)
        j[dim] = slice(None)
        j = tuple(j)
        result[j] = np.linspace(start[i], stop, num1, dtype=dtype)[1:][::step]
    return result
def pad_edge(array, pad_width, mode, *args):
    
    args = tuple(expand_pad_value(array, e) for e in args)
    result = array
    for d in range(array.ndim):
        pad_shapes, pad_chunks = get_pad_shapes_chunks(result, pad_width, (d,))
        pad_arrays = [result, result]
        if mode == "constant":
            constant_values = args[0][d]
            constant_values = [asarray(c).astype(result.dtype) for c in constant_values]
            pad_arrays = [
                broadcast_to(v, s, c)
                for v, s, c in zip(constant_values, pad_shapes, pad_chunks)
            ]
        elif mode in ["edge", "linear_ramp"]:
            pad_slices = [result.ndim * [slice(None)], result.ndim * [slice(None)]]
            pad_slices[0][d] = slice(None, 1, None)
            pad_slices[1][d] = slice(-1, None, None)
            pad_slices = [tuple(sl) for sl in pad_slices]
            pad_arrays = [result[sl] for sl in pad_slices]
            if mode == "edge":
                pad_arrays = [
                    broadcast_to(a, s, c)
                    for a, s, c in zip(pad_arrays, pad_shapes, pad_chunks)
                ]
            elif mode == "linear_ramp":
                end_values = args[0][d]
                pad_arrays = [
                    a.map_blocks(
                        linear_ramp_chunk,
                        ev,
                        pw,
                        chunks=c,
                        dtype=result.dtype,
                        dim=d,
                        step=(2 * i - 1),
                    )
                    for i, (a, ev, pw, c) in enumerate(
                        zip(pad_arrays, end_values, pad_width[d], pad_chunks)
                    )
                ]
        result = concatenate([pad_arrays[0], result, pad_arrays[1]], axis=d)
    return result
def pad_reuse(array, pad_width, mode, *args):
    
    if mode in ["reflect", "symmetric"] and "odd" in args:
        raise NotImplementedError("`pad` does not support `reflect_type` of `odd`.")
    result = np.empty(array.ndim * (3,), dtype=object)
    for idx in np.ndindex(result.shape):
        select = []
        orient = []
        for i, s, pw in zip(idx, array.shape, pad_width):
            if mode == "wrap":
                pw = pw[::-1]
            if i < 1:
                if mode == "reflect":
                    select.append(slice(1, pw[0] + 1, None))
                else:
                    select.append(slice(None, pw[0], None))
            elif i > 1:
                if mode == "reflect":
                    select.append(slice(s - pw[1] - 1, s - 1, None))
                else:
                    select.append(slice(s - pw[1], None, None))
            else:
                select.append(slice(None))
            if i != 1 and mode in ["reflect", "symmetric"]:
                orient.append(slice(None, None, -1))
            else:
                orient.append(slice(None))
        select = tuple(select)
        orient = tuple(orient)
        if mode == "wrap":
            idx = tuple(2 - i for i in idx)
        result[idx] = array[select][orient]
    result = block(result.tolist())
    return result
def pad_stats(array, pad_width, mode, *args):
    
    if mode == "median":
        raise NotImplementedError("`pad` does not support `mode` of `median`.")
    stat_length = expand_pad_value(array, args[0])
    result = np.empty(array.ndim * (3,), dtype=object)
    for idx in np.ndindex(result.shape):
        axes = []
        select = []
        pad_shape = []
        pad_chunks = []
        for d, (i, s, c, w, l) in enumerate(
            zip(idx, array.shape, array.chunks, pad_width, stat_length)
        ):
            if i < 1:
                axes.append(d)
                select.append(slice(None, l[0], None))
                pad_shape.append(w[0])
                pad_chunks.append(w[0])
            elif i > 1:
                axes.append(d)
                select.append(slice(s - l[1], None, None))
                pad_shape.append(w[1])
                pad_chunks.append(w[1])
            else:
                select.append(slice(None))
                pad_shape.append(s)
                pad_chunks.append(c)
        axes = tuple(axes)
        select = tuple(select)
        pad_shape = tuple(pad_shape)
        pad_chunks = tuple(pad_chunks)
        result_idx = array[select]
        if axes:
            if mode == "maximum":
                result_idx = result_idx.max(axis=axes, keepdims=True)
            elif mode == "mean":
                result_idx = result_idx.mean(axis=axes, keepdims=True)
            elif mode == "minimum":
                result_idx = result_idx.min(axis=axes, keepdims=True)
            result_idx = broadcast_to(result_idx, pad_shape, chunks=pad_chunks)
        result[idx] = result_idx
    result = block(result.tolist())
    return result
def wrapped_pad_func(array, pad_func, iaxis_pad_width, iaxis, pad_func_kwargs):
    result = np.empty_like(array)
    for i in np.ndindex(array.shape[:iaxis] + array.shape[iaxis + 1 :]):
        i = i[:iaxis] + (slice(None),) + i[iaxis:]
        result[i] = pad_func(array[i], iaxis_pad_width, iaxis, pad_func_kwargs)
    return result
def pad_udf(array, pad_width, mode, **kwargs):
    
    result = pad_edge(array, pad_width, "constant", 0)
    chunks = result.chunks
    for d in range(result.ndim):
        result = result.rechunk(
            chunks[:d] + (result.shape[d : d + 1],) + chunks[d + 1 :]
        )
        result = result.map_blocks(
            wrapped_pad_func,
            name="pad",
            dtype=result.dtype,
            pad_func=mode,
            iaxis_pad_width=pad_width[d],
            iaxis=d,
            pad_func_kwargs=kwargs,
        )
        result = result.rechunk(chunks)
    return result
@derived_from(np)
def pad(array, pad_width, mode, **kwargs):
    array = asarray(array)
    pad_width = expand_pad_value(array, pad_width)
    if mode in ["maximum", "mean", "median", "minimum"]:
        kwargs.setdefault("stat_length", array.shape)
    elif mode == "constant":
        kwargs.setdefault("constant_values", 0)
    elif mode == "linear_ramp":
        kwargs.setdefault("end_values", 0)
    elif mode in ["reflect", "symmetric"]:
        kwargs.setdefault("reflect_type", "even")
    elif mode in ["edge", "wrap"]:
        if kwargs:
            raise TypeError("Got unsupported keyword arguments.")
    elif callable(mode):
        kwargs.setdefault("kwargs", {})
    else:
        raise ValueError("Got an unsupported `mode`.")
    if not callable(mode) and len(kwargs) > 1:
        raise TypeError("Got too many keyword arguments.")
    if mode in ["maximum", "mean", "median", "minimum"]:
        return pad_stats(array, pad_width, mode, *kwargs.values())
    elif mode in ["constant", "edge", "linear_ramp"]:
        return pad_edge(array, pad_width, mode, *kwargs.values())
    elif mode in ["reflect", "symmetric", "wrap"]:
        return pad_reuse(array, pad_width, mode, *kwargs.values())
    elif callable(mode):
        return pad_udf(array, pad_width, mode, **kwargs)
    else:
        raise ValueError("Unsupported mode selected.")
