from itertools import product
import math
from numbers import Integral, Number
from operator import getitem, itemgetter
import warnings
import functools
import numpy as np
from toolz import memoize, merge, pluck, concat
from .. import core
from ..highlevelgraph import HighLevelGraph
from ..base import tokenize, is_dask_collection
colon = slice(None, None, None)
def _sanitize_index_element(ind):
    
    if isinstance(ind, Number):
        ind2 = int(ind)
        if ind2 != ind:
            raise IndexError("Bad index.  Must be integer-like: %s" % ind)
        else:
            return ind2
    elif ind is None:
        return None
    else:
        raise TypeError("Invalid index type", type(ind), ind)
def sanitize_index(ind):
    
    if ind is None:
        return None
    elif isinstance(ind, slice):
        return slice(
            _sanitize_index_element(ind.start),
            _sanitize_index_element(ind.stop),
            _sanitize_index_element(ind.step),
        )
    elif isinstance(ind, Number):
        return _sanitize_index_element(ind)
    elif is_dask_collection(ind):
        return ind
    index_array = np.asanyarray(ind)
    if index_array.dtype == bool:
        nonzero = np.nonzero(index_array)
        if len(nonzero) == 1:
            # If a 1-element tuple, unwrap the element
            nonzero = nonzero[0]
        return np.asanyarray(nonzero)
    elif np.issubdtype(index_array.dtype, np.integer):
        return index_array
    elif np.issubdtype(index_array.dtype, np.floating):
        int_index = index_array.astype(np.intp)
        if np.allclose(index_array, int_index):
            return int_index
        else:
            check_int = np.isclose(index_array, int_index)
            first_err = index_array.ravel()[np.flatnonzero(~check_int)[0]]
            raise IndexError("Bad index.  Must be integer-like: %s" % first_err)
    else:
        raise TypeError("Invalid index type", type(ind), ind)
def slice_array(out_name, in_name, blockdims, index):
    
    blockdims = tuple(map(tuple, blockdims))
    # x[:, :, :] - Punt and return old value
    if all(
        isinstance(index, slice) and index == slice(None, None, None) for index in index
    ):
        suffixes = product(*[range(len(bd)) for bd in blockdims])
        dsk = dict(((out_name,) + s, (in_name,) + s) for s in suffixes)
        return dsk, blockdims
    # Add in missing colons at the end as needed.  x[5] -> x[5, :, :]
    not_none_count = sum(i is not None for i in index)
    missing = len(blockdims) - not_none_count
    index += (slice(None, None, None),) * missing
    # Pass down to next function
    dsk_out, bd_out = slice_with_newaxes(out_name, in_name, blockdims, index)
    bd_out = tuple(map(tuple, bd_out))
    return dsk_out, bd_out
def slice_with_newaxes(out_name, in_name, blockdims, index):
    
    # Strip Nones from index
    index2 = tuple([ind for ind in index if ind is not None])
    where_none = [i for i, ind in enumerate(index) if ind is None]
    where_none_orig = list(where_none)
    for i, x in enumerate(where_none):
        n = sum(isinstance(ind, Integral) for ind in index[:x])
        if n:
            where_none[i] -= n
    # Pass down and do work
    dsk, blockdims2 = slice_wrap_lists(out_name, in_name, blockdims, index2)
    if where_none:
        expand = expander(where_none)
        expand_orig = expander(where_none_orig)
        # Insert ",0" into the key:  ('x', 2, 3) -> ('x', 0, 2, 0, 3)
        dsk2 = {
            (out_name,) + expand(k[1:], 0): (v[:2] + (expand_orig(v[2], None),))
            for k, v in dsk.items()
            if k[0] == out_name
        }
        # Add back intermediate parts of the dask that weren't the output
        dsk3 = merge(dsk2, {k: v for k, v in dsk.items() if k[0] != out_name})
        # Insert (1,) into blockdims:  ((2, 2), (3, 3)) -> ((2, 2), (1,), (3, 3))
        blockdims3 = expand(blockdims2, (1,))
        return dsk3, blockdims3
    else:
        return dsk, blockdims2
def slice_wrap_lists(out_name, in_name, blockdims, index):
    
    assert all(isinstance(i, (slice, list, Integral, np.ndarray)) for i in index)
    if not len(blockdims) == len(index):
        raise IndexError("Too many indices for array")
    # Do we have more than one list in the index?
    where_list = [
        i for i, ind in enumerate(index) if isinstance(ind, np.ndarray) and ind.ndim > 0
    ]
    if len(where_list) > 1:
        raise NotImplementedError("Don't yet support nd fancy indexing")
    # Is the single list an empty list? In this case just treat it as a zero
    # length slice
    if where_list and not index[where_list[0]].size:
        index = list(index)
        index[where_list.pop()] = slice(0, 0, 1)
        index = tuple(index)
    # No lists, hooray! just use slice_slices_and_integers
    if not where_list:
        return slice_slices_and_integers(out_name, in_name, blockdims, index)
    # Replace all lists with full slices  [3, 1, 0] -> slice(None, None, None)
    index_without_list = tuple(
        slice(None, None, None) if isinstance(i, np.ndarray) else i for i in index
    )
    # lists and full slices.  Just use take
    if all(isinstance(i, np.ndarray) or i == slice(None, None, None) for i in index):
        axis = where_list[0]
        blockdims2, dsk3 = take(
            out_name, in_name, blockdims, index[where_list[0]], axis=axis
        )
    # Mixed case. Both slices/integers and lists. slice/integer then take
    else:
        # Do first pass without lists
        tmp = "slice-" + tokenize((out_name, in_name, blockdims, index))
        dsk, blockdims2 = slice_slices_and_integers(
            tmp, in_name, blockdims, index_without_list
        )
        # After collapsing some axes due to int indices, adjust axis parameter
        axis = where_list[0]
        axis2 = axis - sum(
            1 for i, ind in enumerate(index) if i < axis and isinstance(ind, Integral)
        )
        # Do work
        blockdims2, dsk2 = take(out_name, tmp, blockdims2, index[axis], axis=axis2)
        dsk3 = merge(dsk, dsk2)
    return dsk3, blockdims2
def slice_slices_and_integers(out_name, in_name, blockdims, index):
    
    from .core import unknown_chunk_message
    shape = tuple(cached_cumsum(dim, initial_zero=True)[-1] for dim in blockdims)
    for dim, ind in zip(shape, index):
        if np.isnan(dim) and ind != slice(None, None, None):
            raise ValueError(
                "Arrays chunk sizes are unknown: %s%s" % (shape, unknown_chunk_message)
            )
    assert all(isinstance(ind, (slice, Integral)) for ind in index)
    assert len(index) == len(blockdims)
    # Get a list (for each dimension) of dicts{blocknum: slice()}
    block_slices = list(map(_slice_1d, shape, blockdims, index))
    sorted_block_slices = [sorted(i.items()) for i in block_slices]
    # (in_name, 1, 1, 2), (in_name, 1, 1, 4), (in_name, 2, 1, 2), ...
    in_names = list(product([in_name], *[pluck(0, s) for s in sorted_block_slices]))
    # (out_name, 0, 0, 0), (out_name, 0, 0, 1), (out_name, 0, 1, 0), ...
    out_names = list(
        product(
            [out_name],
            *[
                range(len(d))[::-1] if i.step and i.step < 0 else range(len(d))
                for d, i in zip(block_slices, index)
                if not isinstance(i, Integral)
            ]
        )
    )
    all_slices = list(product(*[pluck(1, s) for s in sorted_block_slices]))
    dsk_out = {
        out_name: (getitem, in_name, slices)
        for out_name, in_name, slices in zip(out_names, in_names, all_slices)
    }
    new_blockdims = [
        new_blockdim(d, db, i)
        for d, i, db in zip(shape, index, blockdims)
        if not isinstance(i, Integral)
    ]
    return dsk_out, new_blockdims
def _slice_1d(dim_shape, lengths, index):
    
    chunk_boundaries = cached_cumsum(lengths)
    if isinstance(index, Integral):
        # use right-side search to be consistent with previous result
        i = chunk_boundaries.searchsorted(index, side="right")
        if i > 0:
            # the very first chunk has no relative shift
            ind = index - chunk_boundaries[i - 1]
        else:
            ind = index
        return {int(i): int(ind)}
    assert isinstance(index, slice)
    if index == colon:
        return {k: colon for k in range(len(lengths))}
    step = index.step or 1
    if step > 0:
        start = index.start or 0
        stop = index.stop if index.stop is not None else dim_shape
    else:
        start = index.start if index.start is not None else dim_shape - 1
        start = dim_shape - 1 if start >= dim_shape else start
        stop = -(dim_shape + 1) if index.stop is None else index.stop
    # posify start and stop
    if start < 0:
        start += dim_shape
    if stop < 0:
        stop += dim_shape
    d = dict()
    if step > 0:
        istart = chunk_boundaries.searchsorted(start, side="right")
        istop = chunk_boundaries.searchsorted(stop, side="left")
        # the bound is not exactly tight; make it tighter?
        istop = min(istop + 1, len(lengths))
        # jump directly to istart
        if istart > 0:
            start = start - chunk_boundaries[istart - 1]
            stop = stop - chunk_boundaries[istart - 1]
        for i in range(istart, istop):
            length = lengths[i]
            if start < length and stop > 0:
                d[i] = slice(start, min(stop, length), step)
                start = (start - length) % step
            else:
                start = start - length
            stop -= length
    else:
        rstart = start  # running start
        istart = chunk_boundaries.searchsorted(start, side="left")
        istop = chunk_boundaries.searchsorted(stop, side="right")
        # the bound is not exactly tight; make it tighter?
        istart = min(istart + 1, len(chunk_boundaries) - 1)
        istop = max(istop - 1, -1)
        for i in range(istart, istop, -1):
            chunk_stop = chunk_boundaries[i]
            # create a chunk start and stop
            if i == 0:
                chunk_start = 0
            else:
                chunk_start = chunk_boundaries[i - 1]
            # if our slice is in this chunk
            if (chunk_start <= rstart < chunk_stop) and (rstart > stop):
                d[i] = slice(
                    rstart - chunk_stop,
                    max(chunk_start - chunk_stop - 1, stop - chunk_stop),
                    step,
                )
                # compute the next running start point,
                offset = (rstart - (chunk_start - 1)) % step
                rstart = chunk_start + offset - 1
    # replace 0:20:1 with : if appropriate
    for k, v in d.items():
        if v == slice(0, lengths[k], 1):
            d[k] = slice(None, None, None)
    if not d:  # special case x[:0]
        d[0] = slice(0, 0, 1)
    return d
def partition_by_size(sizes, seq):
    
    seq = np.asanyarray(seq)
    left = np.empty(len(sizes) + 1, dtype=int)
    left[0] = 0
    right = np.cumsum(sizes, out=left[1:])
    locations = np.empty(len(sizes) + 1, dtype=int)
    locations[0] = 0
    locations[1:] = np.searchsorted(seq, right)
    return [(seq[j:k] - l) for j, k, l in zip(locations[:-1], locations[1:], left)]
def issorted(seq):
    
    if len(seq) == 0:
        return True
    return np.all(seq[:-1] <= seq[1:])
def slicing_plan(chunks, index):
    
    index = np.asanyarray(index)
    cum_chunks = cached_cumsum(chunks)
    chunk_locations = np.searchsorted(cum_chunks, index, side="right")
    where = np.where(np.diff(chunk_locations))[0] + 1
    where = np.concatenate([[0], where, [len(chunk_locations)]])
    out = []
    for i in range(len(where) - 1):
        sub_index = index[where[i] : where[i + 1]]
        chunk = chunk_locations[where[i]]
        if chunk > 0:
            sub_index = sub_index - cum_chunks[chunk - 1]
        out.append((chunk, sub_index))
    return out
def take(outname, inname, chunks, index, axis=0):
    
    plan = slicing_plan(chunks[axis], index)
    if len(plan) >= len(chunks[axis]) * 10:
        factor = math.ceil(len(plan) / len(chunks[axis]))
        from .core import PerformanceWarning
        warnings.warn(
            "Slicing with an out-of-order index is generating %d "
            "times more chunks" % factor,
            PerformanceWarning,
            stacklevel=6,
        )
    index_lists = [idx for _, idx in plan]
    where_index = [i for i, _ in plan]
    dims = [range(len(bd)) for bd in chunks]
    indims = list(dims)
    indims[axis] = list(range(len(where_index)))
    keys = list(product([outname], *indims))
    outdims = list(dims)
    outdims[axis] = where_index
    slices = [[colon] * len(bd) for bd in chunks]
    slices[axis] = index_lists
    slices = list(product(*slices))
    inkeys = list(product([inname], *outdims))
    values = [(getitem, inkey, slc) for inkey, slc in zip(inkeys, slices)]
    chunks2 = list(chunks)
    chunks2[axis] = tuple(map(len, index_lists))
    dsk = dict(zip(keys, values))
    return tuple(chunks2), dsk
def posify_index(shape, ind):
    
    if isinstance(ind, tuple):
        return tuple(map(posify_index, shape, ind))
    if isinstance(ind, Integral):
        if ind < 0 and not math.isnan(shape):
            return ind + shape
        else:
            return ind
    if isinstance(ind, (np.ndarray, list)) and not math.isnan(shape):
        ind = np.asanyarray(ind)
        return np.where(ind < 0, ind + shape, ind)
    return ind
@memoize
def _expander(where):
    if not where:
        def expand(seq, val):
            return seq
        return expand
    else:
        decl = 
        left = []
        j = 0
        for i in range(max(where) + 1):
            if i in where:
                left.append("val, ")
            else:
                left.append("seq[%d], " % j)
                j += 1
        right = "seq[%d:]" % j
        left = "".join(left)
        decl = decl.format(**locals())
        ns = {}
        exec(compile(decl, "<dynamic>", "exec"), ns, ns)
        return ns["expand"]
def expander(where):
    
    return _expander(tuple(where))
def new_blockdim(dim_shape, lengths, index):
    
    if index == slice(None, None, None):
        return lengths
    if isinstance(index, list):
        return [len(index)]
    assert not isinstance(index, Integral)
    pairs = sorted(_slice_1d(dim_shape, lengths, index).items(), key=itemgetter(0))
    slices = [
        slice(0, lengths[i], 1) if slc == slice(None, None, None) else slc
        for i, slc in pairs
    ]
    if isinstance(index, slice) and index.step and index.step < 0:
        slices = slices[::-1]
    return [int(math.ceil((1.0 * slc.stop - slc.start) / slc.step)) for slc in slices]
def replace_ellipsis(n, index):
    
    # Careful about using in or index because index may contain arrays
    isellipsis = [i for i, ind in enumerate(index) if ind is Ellipsis]
    if not isellipsis:
        return index
    else:
        loc = isellipsis[0]
    extra_dimensions = n - (len(index) - sum(i is None for i in index) - 1)
    return (
        index[:loc] + (slice(None, None, None),) * extra_dimensions + index[loc + 1 :]
    )
def normalize_slice(idx, dim):
    
    if isinstance(idx, slice):
        if math.isnan(dim):
            return idx
        start, stop, step = idx.indices(dim)
        if step > 0:
            if start == 0:
                start = None
            if stop >= dim:
                stop = None
            if step == 1:
                step = None
            if stop is not None and start is not None and stop < start:
                stop = start
        elif step < 0:
            if start >= dim - 1:
                start = None
            if stop < 0:
                stop = None
        return slice(start, stop, step)
    return idx
def normalize_index(idx, shape):
    
    if not isinstance(idx, tuple):
        idx = (idx,)
    idx = replace_ellipsis(len(shape), idx)
    n_sliced_dims = 0
    for i in idx:
        if hasattr(i, "ndim") and i.ndim >= 1:
            n_sliced_dims += i.ndim
        elif i is None:
            continue
        else:
            n_sliced_dims += 1
    idx = idx + (slice(None),) * (len(shape) - n_sliced_dims)
    if len([i for i in idx if i is not None]) > len(shape):
        raise IndexError("Too many indices for array")
    none_shape = []
    i = 0
    for ind in idx:
        if ind is not None:
            none_shape.append(shape[i])
            i += 1
        else:
            none_shape.append(None)
    for i, d in zip(idx, none_shape):
        if d is not None:
            check_index(i, d)
    idx = tuple(map(sanitize_index, idx))
    idx = tuple(map(normalize_slice, idx, none_shape))
    idx = posify_index(none_shape, idx)
    return idx
def check_index(ind, dimension):
    
    # unknown dimension, assumed to be in bounds
    if np.isnan(dimension):
        return
    elif isinstance(ind, (list, np.ndarray)):
        x = np.asanyarray(ind)
        if x.dtype == bool:
            if x.size != dimension:
                raise IndexError(
                    "Boolean array length %s doesn't equal dimension %s"
                    % (x.size, dimension)
                )
        elif (x >= dimension).any() or (x < -dimension).any():
            raise IndexError("Index out of bounds %s" % dimension)
    elif isinstance(ind, slice):
        return
    elif is_dask_collection(ind):
        return
    elif ind is None:
        return
    elif ind >= dimension:
        raise IndexError(
            "Index is not smaller than dimension %d >= %d" % (ind, dimension)
        )
    elif ind < -dimension:
        msg = "Negative index is not greater than negative dimension %d <= -%d"
        raise IndexError(msg % (ind, dimension))
def slice_with_int_dask_array(x, index):
    
    from .core import Array
    assert len(index) == x.ndim
    fancy_indexes = [
        isinstance(idx, (tuple, list))
        or (isinstance(idx, (np.ndarray, Array)) and idx.ndim > 0)
        for idx in index
    ]
    if sum(fancy_indexes) > 1:
        raise NotImplementedError("Don't yet support nd fancy indexing)")
    out_index = []
    dropped_axis_cnt = 0
    for in_axis, idx in enumerate(index):
        out_axis = in_axis - dropped_axis_cnt
        if isinstance(idx, Array) and idx.dtype.kind in "iu":
            if idx.ndim == 0:
                idx = idx[np.newaxis]
                x = slice_with_int_dask_array_on_axis(x, idx, out_axis)
                x = x[tuple(0 if i == out_axis else slice(None) for i in range(x.ndim))]
                dropped_axis_cnt += 1
            elif idx.ndim == 1:
                x = slice_with_int_dask_array_on_axis(x, idx, out_axis)
                out_index.append(slice(None))
            else:
                raise NotImplementedError(
                    "Slicing with dask.array of ints only permitted when "
                    "the indexer has zero or one dimensions"
                )
        else:
            out_index.append(idx)
    return x, tuple(out_index)
def slice_with_int_dask_array_on_axis(x, idx, axis):
    
    from .core import Array, blockwise, from_array
    from . import chunk
    assert 0 <= axis < x.ndim
    if np.isnan(x.chunks[axis]).any():
        raise NotImplementedError(
            "Slicing an array with unknown chunks with "
            "a dask.array of ints is not supported"
        )
    # Calculate the offset at which each chunk starts along axis
    # e.g. chunks=(..., (5, 3, 4), ...) -> offset=[0, 5, 8]
    offset = np.roll(np.cumsum(x.chunks[axis]), 1)
    offset[0] = 0
    offset = from_array(offset, chunks=1)
    # Tamper with the declared chunks of offset to make blockwise align it with
    # x[axis]
    offset = Array(offset.dask, offset.name, (x.chunks[axis],), offset.dtype)
    # Define axis labels for blockwise
    x_axes = tuple(range(x.ndim))
    idx_axes = (x.ndim,)  # arbitrary index not already in x_axes
    offset_axes = (axis,)
    p_axes = x_axes[: axis + 1] + idx_axes + x_axes[axis + 1 :]
    y_axes = x_axes[:axis] + idx_axes + x_axes[axis + 1 :]
    # Calculate the cartesian product of every chunk of x vs every chunk of idx
    p = blockwise(
        chunk.slice_with_int_dask_array,
        p_axes,
        x,
        x_axes,
        idx,
        idx_axes,
        offset,
        offset_axes,
        x_size=x.shape[axis],
        axis=axis,
        dtype=x.dtype,
    )
    # Aggregate on the chunks of x along axis
    y = blockwise(
        chunk.slice_with_int_dask_array_aggregate,
        y_axes,
        idx,
        idx_axes,
        p,
        p_axes,
        concatenate=True,
        x_chunks=x.chunks[axis],
        axis=axis,
        dtype=x.dtype,
    )
    return y
def slice_with_bool_dask_array(x, index):
    
    from .core import Array, blockwise, elemwise
    out_index = [
        slice(None) if isinstance(ind, Array) and ind.dtype == bool else ind
        for ind in index
    ]
    if len(index) == 1 and index[0].ndim == x.ndim:
        if not np.isnan(x.shape).any() and not np.isnan(index[0].shape).any():
            x = x.ravel()
            index = tuple(i.ravel() for i in index)
        elif x.ndim > 1:
            warnings.warn(
                "When slicing a Dask array of unknown chunks with a boolean mask "
                "Dask array, the output array may have a different ordering "
                "compared to the equivalent NumPy operation. This will raise an "
                "error in a future release of Dask.",
                stacklevel=3,
            )
        y = elemwise(getitem, x, *index, dtype=x.dtype)
        name = "getitem-" + tokenize(x, index)
        dsk = {(name, i): k for i, k in enumerate(core.flatten(y.__dask_keys__()))}
        chunks = ((np.nan,) * y.npartitions,)
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[y])
        return Array(graph, name, chunks, x.dtype), out_index
    if any(
        isinstance(ind, Array) and ind.dtype == bool and ind.ndim != 1 for ind in index
    ):
        raise NotImplementedError(
            "Slicing with dask.array of bools only permitted when "
            "the indexer has only one dimension or when "
            "it has the same dimension as the sliced "
            "array"
        )
    indexes = [
        ind if isinstance(ind, Array) and ind.dtype == bool else slice(None)
        for ind in index
    ]
    arginds = []
    i = 0
    for ind in indexes:
        if isinstance(ind, Array) and ind.dtype == bool:
            new = (ind, tuple(range(i, i + ind.ndim)))
            i += x.ndim
        else:
            new = (slice(None), None)
            i += 1
        arginds.append(new)
    arginds = list(concat(arginds))
    out = blockwise(
        getitem_variadic,
        tuple(range(x.ndim)),
        x,
        tuple(range(x.ndim)),
        *arginds,
        dtype=x.dtype
    )
    chunks = []
    for ind, chunk in zip(index, out.chunks):
        if isinstance(ind, Array) and ind.dtype == bool:
            chunks.append((np.nan,) * len(chunk))
        else:
            chunks.append(chunk)
    out._chunks = tuple(chunks)
    return out, tuple(out_index)
def getitem_variadic(x, *index):
    return x[index]
def make_block_sorted_slices(index, chunks):
    
    from .core import slices_from_chunks
    slices = slices_from_chunks(chunks)
    if len(slices[0]) > 1:
        slices = [slice_[0] for slice_ in slices]
    offsets = np.roll(np.cumsum(chunks[0]), 1)
    offsets[0] = 0
    index2 = np.empty_like(index)
    index3 = np.empty_like(index)
    for slice_, offset in zip(slices, offsets):
        a = index[slice_]
        b = np.sort(a)
        c = offset + np.argsort(b.take(np.argsort(a)))
        index2[slice_] = b
        index3[slice_] = c
    return index2, index3
def shuffle_slice(x, index):
    
    from .core import PerformanceWarning
    chunks1 = chunks2 = x.chunks
    if x.ndim > 1:
        chunks1 = (chunks1[0],)
    index2, index3 = make_block_sorted_slices(index, chunks1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        return x[index2].rechunk(chunks2)[index3]
class _HashIdWrapper(object):
    
    def __init__(self, wrapped):
        self.wrapped = wrapped
    def __eq__(self, other):
        if not isinstance(other, _HashIdWrapper):
            return NotImplemented
        return self.wrapped is other.wrapped
    def __ne__(self, other):
        if not isinstance(other, _HashIdWrapper):
            return NotImplemented
        return self.wrapped is not other.wrapped
    def __hash__(self):
        return id(self.wrapped)
@functools.lru_cache()
def _cumsum(seq):
    if isinstance(seq, _HashIdWrapper):
        return _cumsum(seq.wrapped)
    seq = np.array(seq)
    dtype = np.int64 if np.issubdtype(seq.dtype, np.integer) else seq.dtype
    out = np.empty(len(seq) + 1, dtype)
    out[0] = 0
    np.cumsum(seq, out=out[1:], dtype=dtype)
    return out
def cached_cumsum(seq, initial_zero=False):
    
    if isinstance(seq, tuple):
        # Look up by identity first, to avoid a linear-time __hash__
        # if we've seen this tuple object before.
        result = _cumsum(_HashIdWrapper(seq))
    else:
        # Construct a temporary tuple, and look up by value.
        result = _cumsum(tuple(seq))
    if not initial_zero:
        result = result[1:]
    return result
