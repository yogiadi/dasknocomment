import copy
import math
import operator
import os
import pickle
import re
import sys
import traceback
import uuid
import warnings
from bisect import bisect
from collections.abc import Iterable, Iterator, Mapping
from functools import partial, wraps
from itertools import product, zip_longest
from numbers import Number, Integral
from operator import add, getitem, mul
from threading import Lock
try:
    from cytoolz import partition, concat, first, groupby, accumulate
    from cytoolz.curried import pluck
except ImportError:
    from toolz import partition, concat, first, groupby, accumulate
    from toolz.curried import pluck
from toolz import map, reduce, frequencies
import numpy as np
from . import chunk
from .. import config, compute
from ..base import (
    DaskMethodsMixin,
    tokenize,
    dont_optimize,
    compute_as_if_collection,
    persist,
    is_dask_collection,
)
from ..blockwise import broadcast_dimensions, subs
from ..context import globalmethod
from ..utils import (
    ndeepmap,
    ignoring,
    concrete,
    derived_from,
    is_integer,
    IndexCallable,
    funcname,
    SerializableLock,
    Dispatch,
    factors,
    parse_bytes,
    has_keyword,
    M,
    ndimlist,
    format_bytes,
    typename,
)
from ..core import quote
from ..delayed import delayed, Delayed
from .. import threaded, core
from ..sizeof import sizeof
from ..highlevelgraph import HighLevelGraph
from .numpy_compat import _Recurser, _make_sliced_dtype
from .slicing import slice_array, replace_ellipsis, cached_cumsum
from .blockwise import blockwise
config.update_defaults({"array": {"chunk-size": "128MiB", "rechunk-threshold": 4}})
concatenate_lookup = Dispatch("concatenate")
tensordot_lookup = Dispatch("tensordot")
einsum_lookup = Dispatch("einsum")
concatenate_lookup.register((object, np.ndarray), np.concatenate)
tensordot_lookup.register((object, np.ndarray), np.tensordot)
einsum_lookup.register((object, np.ndarray), np.einsum)
unknown_chunk_message = (
    "\n\n"
    "A possible solution: "
    "https://docs.dask.org/en/latest/array-chunks.html#unknown-chunks\n"
    "Summary: to compute chunks sizes, use\n\n"
    "   x.compute_chunk_sizes()  # for Dask Array `x`\n"
    "   ddf.to_dask_array(lengths=True)  # for Dask DataFrame `ddf`"
)
class PerformanceWarning(Warning):
    
def getter(a, b, asarray=True, lock=None):
    if isinstance(b, tuple) and any(x is None for x in b):
        b2 = tuple(x for x in b if x is not None)
        b3 = tuple(
            None if x is None else slice(None, None)
            for x in b
            if not isinstance(x, Integral)
        )
        return getter(a, b2, asarray=asarray, lock=lock)[b3]
    if lock:
        lock.acquire()
    try:
        c = a[b]
        if asarray:
            c = np.asarray(c)
    finally:
        if lock:
            lock.release()
    return c
def getter_nofancy(a, b, asarray=True, lock=None):
    
    return getter(a, b, asarray=asarray, lock=lock)
def getter_inline(a, b, asarray=True, lock=None):
    
    return getter(a, b, asarray=asarray, lock=lock)
from .optimization import optimize, fuse_slice
# __array_function__ dict for mapping aliases and mismatching names
_HANDLED_FUNCTIONS = {}
def implements(*numpy_functions):
    
    def decorator(dask_func):
        for numpy_function in numpy_functions:
            _HANDLED_FUNCTIONS[numpy_function] = dask_func
        return dask_func
    return decorator
def slices_from_chunks(chunks):
    
    cumdims = [list(accumulate(add, (0,) + bds[:-1])) for bds in chunks]
    shapes = product(*chunks)
    starts = product(*cumdims)
    return [
        tuple(slice(s, s + dim) for s, dim in zip(start, shape))
        for start, shape in zip(starts, shapes)
    ]
def getem(
    arr,
    chunks,
    getitem=getter,
    shape=None,
    out_name=None,
    lock=False,
    asarray=True,
    dtype=None,
):
    
    out_name = out_name or arr
    chunks = normalize_chunks(chunks, shape, dtype=dtype)
    keys = list(product([out_name], *[range(len(bds)) for bds in chunks]))
    slices = slices_from_chunks(chunks)
    if (
        has_keyword(getitem, "asarray")
        and has_keyword(getitem, "lock")
        and (not asarray or lock)
    ):
        values = [(getitem, arr, x, asarray, lock) for x in slices]
    else:
        # Common case, drop extra parameters
        values = [(getitem, arr, x) for x in slices]
    return dict(zip(keys, values))
def dotmany(A, B, leftfunc=None, rightfunc=None, **kwargs):
    
    if leftfunc:
        A = map(leftfunc, A)
    if rightfunc:
        B = map(rightfunc, B)
    return sum(map(partial(np.dot, **kwargs), A, B))
def _concatenate2(arrays, axes=[]):
    
    if axes == ():
        if isinstance(arrays, list):
            return arrays[0]
        else:
            return arrays
    if isinstance(arrays, Iterator):
        arrays = list(arrays)
    if not isinstance(arrays, (list, tuple)):
        return arrays
    if len(axes) > 1:
        arrays = [_concatenate2(a, axes=axes[1:]) for a in arrays]
    concatenate = concatenate_lookup.dispatch(
        type(max(arrays, key=lambda x: getattr(x, "__array_priority__", 0)))
    )
    return concatenate(arrays, axis=axes[0])
def apply_infer_dtype(func, args, kwargs, funcname, suggest_dtype="dtype", nout=None):
    
    args = [
        np.ones((1,) * x.ndim, dtype=x.dtype) if isinstance(x, Array) else x
        for x in args
    ]
    try:
        with np.errstate(all="ignore"):
            o = func(*args, **kwargs)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = "".join(traceback.format_tb(exc_traceback))
        suggest = (
            (
                "Please specify the dtype explicitly using the "
                "`{dtype}` kwarg.\n\n".format(dtype=suggest_dtype)
            )
            if suggest_dtype
            else ""
        )
        msg = (
            "`dtype` inference failed in `{0}`.\n\n"
            "{1}"
            "Original error is below:\n"
            "------------------------\n"
            "{2}\n\n"
            "Traceback:\n"
            "---------\n"
            "{3}"
        ).format(funcname, suggest, repr(e), tb)
    else:
        msg = None
    if msg is not None:
        raise ValueError(msg)
    return o.dtype if nout is None else tuple(e.dtype for e in o)
def normalize_arg(x):
    
    if is_dask_collection(x):
        return x
    elif isinstance(x, str) and re.match(r"_\d+", x):
        return delayed(x)
    elif isinstance(x, list) and len(x) >= 10:
        return delayed(x)
    elif sizeof(x) > 1e6:
        return delayed(x)
    else:
        return x
def map_blocks(
    func,
    *args,
    name=None,
    token=None,
    dtype=None,
    chunks=None,
    drop_axis=[],
    new_axis=None,
    meta=None,
    **kwargs
):
    
    if not callable(func):
        msg = (
            "First argument must be callable function, not %s\n"
            "Usage:   da.map_blocks(function, x)\n"
            "   or:   da.map_blocks(function, x, y, z)"
        )
        raise TypeError(msg % type(func).__name__)
    if token:
        warnings.warn("The token= keyword to map_blocks has been moved to name=")
        name = token
    name = "%s-%s" % (name or funcname(func), tokenize(func, *args, **kwargs))
    new_axes = {}
    if isinstance(drop_axis, Number):
        drop_axis = [drop_axis]
    if isinstance(new_axis, Number):
        new_axis = [new_axis]  # TODO: handle new_axis
    arrs = [a for a in args if isinstance(a, Array)]
    argpairs = [
        (a, tuple(range(a.ndim))[::-1]) if isinstance(a, Array) else (a, None)
        for a in args
    ]
    if arrs:
        out_ind = tuple(range(max(a.ndim for a in arrs)))[::-1]
    else:
        out_ind = ()
    if has_keyword(func, "block_id"):
        kwargs["block_id"] = "__block_id_dummy__"
    if has_keyword(func, "block_info"):
        kwargs["block_info"] = "__block_info_dummy__"
    original_kwargs = kwargs
    if dtype is None and meta is None:
        dtype = apply_infer_dtype(func, args, original_kwargs, "map_blocks")
    if drop_axis:
        out_ind = tuple(x for i, x in enumerate(out_ind) if i not in drop_axis)
    if new_axis is None and chunks is not None and len(out_ind) < len(chunks):
        new_axis = range(len(chunks) - len(out_ind))
    if new_axis:
        # new_axis = [x + len(drop_axis) for x in new_axis]
        out_ind = list(out_ind)
        for ax in sorted(new_axis):
            n = len(out_ind) + len(drop_axis)
            out_ind.insert(ax, n)
            if chunks is not None:
                new_axes[n] = chunks[ax]
            else:
                new_axes[n] = 1
        out_ind = tuple(out_ind)
        if max(new_axis) > max(out_ind):
            raise ValueError("New_axis values do not fill in all dimensions")
    out = blockwise(
        func,
        out_ind,
        *concat(argpairs),
        name=name,
        new_axes=new_axes,
        dtype=dtype,
        concatenate=True,
        align_arrays=False,
        meta=meta,
        **kwargs
    )
    if has_keyword(func, "block_id") or has_keyword(func, "block_info") or drop_axis:
        dsk = out.dask.layers[out.name]
        dsk = dict(dsk)
        out.dask.layers[out.name] = dsk
    if has_keyword(func, "block_id"):
        for k, vv in dsk.items():
            v = copy.copy(vv[0])  # Need to copy and unpack subgraph callable
            v.dsk = copy.copy(v.dsk)
            [(key, task)] = v.dsk.items()
            task = subs(task, {"__block_id_dummy__": k[1:]})
            v.dsk[key] = task
            dsk[k] = (v,) + vv[1:]
    if chunks is not None:
        if len(chunks) != len(out.numblocks):
            raise ValueError(
                "Provided chunks have {0} dims, expected {1} "
                "dims.".format(len(chunks), len(out.numblocks))
            )
        chunks2 = []
        for i, (c, nb) in enumerate(zip(chunks, out.numblocks)):
            if isinstance(c, tuple):
                # We only check cases where numblocks > 1. Because of
                # broadcasting, we can't (easily) validate the chunks
                # when the number of blocks is 1.
                # See https://github.com/dask/dask/issues/4299 for more.
                if nb > 1 and len(c) != nb:
                    raise ValueError(
                        "Dimension {0} has {1} blocks, "
                        "chunks specified with "
                        "{2} blocks".format(i, nb, len(c))
                    )
                chunks2.append(c)
            else:
                chunks2.append(nb * (c,))
        out._chunks = normalize_chunks(chunks2)
    # If func has block_info as an argument, add it to the kwargs for each call
    if has_keyword(func, "block_info"):
        starts = {}
        num_chunks = {}
        shapes = {}
        for i, (arg, in_ind) in enumerate(argpairs):
            if in_ind is not None:
                shapes[i] = arg.shape
                if drop_axis:
                    # We concatenate along dropped axes, so we need to treat them
                    # as if there is only a single chunk.
                    starts[i] = [
                        (
                            cached_cumsum(arg.chunks[j], initial_zero=True)
                            if ind in out_ind
                            else np.array([0, arg.shape[j]])
                        )
                        for j, ind in enumerate(in_ind)
                    ]
                    num_chunks[i] = tuple(len(s) - 1 for s in starts[i])
                else:
                    starts[i] = [
                        cached_cumsum(c, initial_zero=True) for c in arg.chunks
                    ]
                    num_chunks[i] = arg.numblocks
        out_starts = [cached_cumsum(c, initial_zero=True) for c in out.chunks]
        for k, v in dsk.items():
            vv = v
            v = v[0]
            [(key, task)] = v.dsk.items()  # unpack subgraph callable
            # Get position of chunk, indexed by axis labels
            location = {out_ind[i]: loc for i, loc in enumerate(k[1:])}
            info = {}
            for i, shape in shapes.items():
                # Compute chunk key in the array, taking broadcasting into
                # account. We don't directly know which dimensions are
                # broadcast, but any dimension with only one chunk can be
                # treated as broadcast.
                arr_k = tuple(
                    location.get(ind, 0) if num_chunks[i][j] > 1 else 0
                    for j, ind in enumerate(argpairs[i][1])
                )
                info[i] = {
                    "shape": shape,
                    "num-chunks": num_chunks[i],
                    "array-location": [
                        (starts[i][ij][j], starts[i][ij][j + 1])
                        for ij, j in enumerate(arr_k)
                    ],
                    "chunk-location": arr_k,
                }
            info[None] = {
                "shape": out.shape,
                "num-chunks": out.numblocks,
                "array-location": [
                    (out_starts[ij][j], out_starts[ij][j + 1])
                    for ij, j in enumerate(k[1:])
                ],
                "chunk-location": k[1:],
                "chunk-shape": tuple(out.chunks[ij][j] for ij, j in enumerate(k[1:])),
                "dtype": dtype,
            }
            v = copy.copy(v)  # Need to copy and unpack subgraph callable
            v.dsk = copy.copy(v.dsk)
            [(key, task)] = v.dsk.items()
            task = subs(task, {"__block_info_dummy__": info})
            v.dsk[key] = task
            dsk[k] = (v,) + vv[1:]
    return out
def broadcast_chunks(*chunkss):
    
    if not chunkss:
        return ()
    elif len(chunkss) == 1:
        return chunkss[0]
    n = max(map(len, chunkss))
    chunkss2 = [((1,),) * (n - len(c)) + c for c in chunkss]
    result = []
    for i in range(n):
        step1 = [c[i] for c in chunkss2]
        if all(c == (1,) for c in step1):
            step2 = step1
        else:
            step2 = [c for c in step1 if c != (1,)]
        if len(set(step2)) != 1:
            raise ValueError("Chunks do not align: %s" % str(step2))
        result.append(step2[0])
    return tuple(result)
def store(
    sources,
    targets,
    lock=True,
    regions=None,
    compute=True,
    return_stored=False,
    **kwargs
):
    
    if isinstance(sources, Array):
        sources = [sources]
        targets = [targets]
    if any(not isinstance(s, Array) for s in sources):
        raise ValueError("All sources must be dask array objects")
    if len(sources) != len(targets):
        raise ValueError(
            "Different number of sources [%d] and targets [%d]"
            % (len(sources), len(targets))
        )
    if isinstance(regions, tuple) or regions is None:
        regions = [regions]
    if len(sources) > 1 and len(regions) == 1:
        regions *= len(sources)
    if len(sources) != len(regions):
        raise ValueError(
            "Different number of sources [%d] and targets [%d] than regions [%d]"
            % (len(sources), len(targets), len(regions))
        )
    # Optimize all sources together
    sources_dsk = HighLevelGraph.merge(*[e.__dask_graph__() for e in sources])
    sources_dsk = Array.__dask_optimize__(
        sources_dsk, list(core.flatten([e.__dask_keys__() for e in sources]))
    )
    sources2 = [Array(sources_dsk, e.name, e.chunks, meta=e) for e in sources]
    # Optimize all targets together
    targets2 = []
    targets_keys = []
    targets_dsk = []
    for e in targets:
        if isinstance(e, Delayed):
            targets2.append(e.key)
            targets_keys.extend(e.__dask_keys__())
            targets_dsk.append(e.__dask_graph__())
        elif is_dask_collection(e):
            raise TypeError("Targets must be either Delayed objects or array-likes")
        else:
            targets2.append(e)
    targets_dsk = HighLevelGraph.merge(*targets_dsk)
    targets_dsk = Delayed.__dask_optimize__(targets_dsk, targets_keys)
    load_stored = return_stored and not compute
    toks = [str(uuid.uuid1()) for _ in range(len(sources))]
    store_dsk = HighLevelGraph.merge(
        *[
            insert_to_ooc(s, t, lock, r, return_stored, load_stored, tok)
            for s, t, r, tok in zip(sources2, targets2, regions, toks)
        ]
    )
    store_keys = list(store_dsk.keys())
    store_dsk = HighLevelGraph.merge(store_dsk, targets_dsk, sources_dsk)
    if return_stored:
        load_store_dsk = store_dsk
        if compute:
            store_dlyds = [Delayed(k, store_dsk) for k in store_keys]
            store_dlyds = persist(*store_dlyds, **kwargs)
            store_dsk_2 = HighLevelGraph.merge(*[e.dask for e in store_dlyds])
            load_store_dsk = retrieve_from_ooc(store_keys, store_dsk, store_dsk_2)
        result = tuple(
            Array(load_store_dsk, "load-store-%s" % t, s.chunks, meta=s)
            for s, t in zip(sources, toks)
        )
        return result
    else:
        name = "store-" + str(uuid.uuid1())
        dsk = HighLevelGraph.merge({name: store_keys}, store_dsk)
        result = Delayed(name, dsk)
        if compute:
            result.compute(**kwargs)
            return None
        else:
            return result
def blockdims_from_blockshape(shape, chunks):
    
    if chunks is None:
        raise TypeError("Must supply chunks= keyword argument")
    if shape is None:
        raise TypeError("Must supply shape= keyword argument")
    if np.isnan(sum(shape)) or np.isnan(sum(chunks)):
        raise ValueError(
            "Array chunk sizes are unknown. shape: %s, chunks: %s%s"
            % (shape, chunks, unknown_chunk_message)
        )
    if not all(map(is_integer, chunks)):
        raise ValueError("chunks can only contain integers.")
    if not all(map(is_integer, shape)):
        raise ValueError("shape can only contain integers.")
    shape = tuple(map(int, shape))
    chunks = tuple(map(int, chunks))
    return tuple(
        ((bd,) * (d // bd) + ((d % bd,) if d % bd else ()) if d else (0,))
        for d, bd in zip(shape, chunks)
    )
def finalize(results):
    if not results:
        return concatenate3(results)
    results2 = results
    while isinstance(results2, (tuple, list)):
        if len(results2) > 1:
            return concatenate3(results)
        else:
            results2 = results2[0]
    return unpack_singleton(results)
CHUNKS_NONE_ERROR_MESSAGE = 
.strip()
class Array(DaskMethodsMixin):
    
    __slots__ = "dask", "_name", "_cached_keys", "_chunks", "_meta"
    def __new__(cls, dask, name, chunks, dtype=None, meta=None, shape=None):
        self = super(Array, cls).__new__(cls)
        assert isinstance(dask, Mapping)
        if not isinstance(dask, HighLevelGraph):
            dask = HighLevelGraph.from_collections(name, dask, dependencies=())
        self.dask = dask
        self.name = name
        meta = meta_from_array(meta, dtype=dtype)
        if (
            isinstance(chunks, str)
            or isinstance(chunks, tuple)
            and chunks
            and any(isinstance(c, str) for c in chunks)
        ):
            dt = meta.dtype
        else:
            dt = None
        self._chunks = normalize_chunks(chunks, shape, dtype=dt)
        if self._chunks is None:
            raise ValueError(CHUNKS_NONE_ERROR_MESSAGE)
        self._meta = meta_from_array(meta, ndim=self.ndim, dtype=dtype)
        for plugin in config.get("array_plugins", ()):
            result = plugin(self)
            if result is not None:
                self = result
        return self
    def __reduce__(self):
        return (Array, (self.dask, self.name, self.chunks, self.dtype))
    def __dask_graph__(self):
        return self.dask
    def __dask_layers__(self):
        return (self.name,)
    def __dask_keys__(self):
        if self._cached_keys is not None:
            return self._cached_keys
        name, chunks, numblocks = self.name, self.chunks, self.numblocks
        def keys(*args):
            if not chunks:
                return [(name,)]
            ind = len(args)
            if ind + 1 == len(numblocks):
                result = [(name,) + args + (i,) for i in range(numblocks[ind])]
            else:
                result = [keys(*(args + (i,))) for i in range(numblocks[ind])]
            return result
        self._cached_keys = result = keys()
        return result
    def __dask_tokenize__(self):
        return self.name
    __dask_optimize__ = globalmethod(
        optimize, key="array_optimize", falsey=dont_optimize
    )
    __dask_scheduler__ = staticmethod(threaded.get)
    def __dask_postcompute__(self):
        return finalize, ()
    def __dask_postpersist__(self):
        return Array, (self.name, self.chunks, self.dtype, self._meta)
    @property
    def numblocks(self):
        return tuple(map(len, self.chunks))
    @property
    def npartitions(self):
        return reduce(mul, self.numblocks, 1)
    def compute_chunk_sizes(self):
        
        x = self
        chunk_shapes = x.map_blocks(
            _get_chunk_shape,
            dtype=int,
            chunks=tuple(len(c) * (1,) for c in x.chunks) + ((x.ndim,),),
            new_axis=x.ndim,
        )
        c = []
        for i in range(x.ndim):
            s = x.ndim * [0] + [i]
            s[i] = slice(None)
            s = tuple(s)
            c.append(tuple(chunk_shapes[s]))
        x._chunks = compute(tuple(c))[0]
        return x
    @property
    def shape(self):
        return tuple(map(sum, self.chunks))
    @property
    def chunksize(self):
        return tuple(max(c) for c in self.chunks)
    @property
    def dtype(self):
        return self._meta.dtype
    def _get_chunks(self):
        return self._chunks
    def _set_chunks(self, chunks):
        msg = (
            "Can not set chunks directly\n\n"
            "Please use the rechunk method instead:\n"
            "  x.rechunk({})\n\n"
            "If trying to avoid unknown chunks, use\n"
            "  x.compute_chunk_sizes()"
        )
        raise TypeError(msg.format(chunks))
    chunks = property(_get_chunks, _set_chunks, "chunks property")
    def __len__(self):
        if not self.chunks:
            raise TypeError("len() of unsized object")
        return sum(self.chunks[0])
    def __array_ufunc__(self, numpy_ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            if not isinstance(x, (np.ndarray, Number, Array)):
                return NotImplemented
        if method == "__call__":
            if numpy_ufunc is np.matmul:
                from .routines import matmul
                # special case until apply_gufunc handles optional dimensions
                return matmul(*inputs, **kwargs)
            if numpy_ufunc.signature is not None:
                from .gufunc import apply_gufunc
                return apply_gufunc(
                    numpy_ufunc, numpy_ufunc.signature, *inputs, **kwargs
                )
            if numpy_ufunc.nout > 1:
                from . import ufunc
                try:
                    da_ufunc = getattr(ufunc, numpy_ufunc.__name__)
                except AttributeError:
                    return NotImplemented
                return da_ufunc(*inputs, **kwargs)
            else:
                return elemwise(numpy_ufunc, *inputs, **kwargs)
        elif method == "outer":
            from . import ufunc
            try:
                da_ufunc = getattr(ufunc, numpy_ufunc.__name__)
            except AttributeError:
                return NotImplemented
            return da_ufunc.outer(*inputs, **kwargs)
        else:
            return NotImplemented
    def __repr__(self):
        
        chunksize = str(self.chunksize)
        name = self.name.rsplit("-", 1)[0]
        return "dask.array<%s, shape=%s, dtype=%s, chunksize=%s, chunktype=%s.%s>" % (
            name,
            self.shape,
            self.dtype,
            chunksize,
            type(self._meta).__module__.split(".")[0],
            type(self._meta).__name__,
        )
    def _repr_html_(self):
        table = self._repr_html_table()
        try:
            grid = self.to_svg(size=config.get("array.svg.size", 120))
        except NotImplementedError:
            grid = ""
        both = [
            "<table>",
            "<tr>",
            "<td>",
            table,
            "</td>",
            "<td>",
            grid,
            "</td>",
            "</tr>",
            "</table>",
        ]
        return "\n".join(both)
    def _repr_html_table(self):
        if "sparse" in typename(type(self._meta)):
            nbytes = None
            cbytes = None
        elif not math.isnan(self.nbytes):
            nbytes = format_bytes(self.nbytes)
            cbytes = format_bytes(np.prod(self.chunksize) * self.dtype.itemsize)
        else:
            nbytes = "unknown"
            cbytes = "unknown"
        table = [
            "<table>",
            "  <thead>",
            "    <tr><td> </td><th> Array </th><th> Chunk </th></tr>",
            "  </thead>",
            "  <tbody>",
            "    <tr><th> Bytes </th><td> %s </td> <td> %s </td></tr>"
            % (nbytes, cbytes)
            if nbytes is not None
            else "",
            "    <tr><th> Shape </th><td> %s </td> <td> %s </td></tr>"
            % (str(self.shape), str(self.chunksize)),
            "    <tr><th> Count </th><td> %d Tasks </td><td> %d Chunks </td></tr>"
            % (len(self.__dask_graph__()), self.npartitions),
            "    <tr><th> Type </th><td> %s </td><td> %s.%s </td></tr>"
            % (
                self.dtype,
                type(self._meta).__module__.split(".")[0],
                type(self._meta).__name__,
            ),
            "  </tbody>",
            "</table>",
        ]
        return "\n".join(table)
    @property
    def ndim(self):
        return len(self.shape)
    @property
    def size(self):
        
        return reduce(mul, self.shape, 1)
    @property
    def nbytes(self):
        
        return self.size * self.dtype.itemsize
    @property
    def itemsize(self):
        
        return self.dtype.itemsize
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, val):
        self._name = val
        # Clear the key cache when the name is reset
        self._cached_keys = None
    __array_priority__ = 11  # higher than numpy.ndarray and numpy.matrix
    def __array__(self, dtype=None, **kwargs):
        x = self.compute()
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        return x
    def __array_function__(self, func, types, args, kwargs):
        import dask.array as module
        def handle_nonmatching_names(func, args, kwargs):
            if func not in _HANDLED_FUNCTIONS:
                warnings.warn(
                    "The `{}` function is not implemented by Dask array. "
                    "You may want to use the da.map_blocks function "
                    "or something similar to silence this warning. "
                    "Your code may stop working in a future release.".format(
                        func.__module__ + "." + func.__name__
                    ),
                    FutureWarning,
                )
                # Need to convert to array object (e.g. numpy.ndarray or
                # cupy.ndarray) as needed, so we can call the NumPy function
                # again and it gets the chance to dispatch to the right
                # implementation.
                args, kwargs = compute(args, kwargs)
                return func(*args, **kwargs)
            return _HANDLED_FUNCTIONS[func](*args, **kwargs)
        # First try to find a matching function name.  If that doesn't work, we may
        # be dealing with an alias or a function that's simply not in the Dask API.
        # Handle aliases via the _HANDLED_FUNCTIONS dict mapping, and warn otherwise.
        for submodule in func.__module__.split(".")[1:]:
            try:
                module = getattr(module, submodule)
            except AttributeError:
                return handle_nonmatching_names(func, args, kwargs)
        if not hasattr(module, func.__name__):
            return handle_nonmatching_names(func, args, kwargs)
        da_func = getattr(module, func.__name__)
        if da_func is func:
            return handle_nonmatching_names(func, args, kwargs)
        return da_func(*args, **kwargs)
    @property
    def _elemwise(self):
        return elemwise
    @wraps(store)
    def store(self, target, **kwargs):
        r = store([self], [target], **kwargs)
        if kwargs.get("return_stored", False):
            r = r[0]
        return r
    def to_svg(self, size=500):
        
        from .svg import svg
        return svg(self.chunks, size=size)
    def to_hdf5(self, filename, datapath, **kwargs):
        
        return to_hdf5(filename, datapath, self, **kwargs)
    def to_dask_dataframe(self, columns=None, index=None):
        
        from ..dataframe import from_dask_array
        return from_dask_array(self, columns=columns, index=index)
    def __bool__(self):
        if self.size > 1:
            raise ValueError(
                "The truth value of a {0} is ambiguous. "
                "Use a.any() or a.all().".format(self.__class__.__name__)
            )
        else:
            return bool(self.compute())
    __nonzero__ = __bool__  # python 2
    def _scalarfunc(self, cast_type):
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted to Python scalars")
        else:
            return cast_type(self.compute())
    def __int__(self):
        return self._scalarfunc(int)
    __long__ = __int__  # python 2
    def __float__(self):
        return self._scalarfunc(float)
    def __complex__(self):
        return self._scalarfunc(complex)
    def __setitem__(self, key, value):
        from .routines import where
        if isinstance(key, Array):
            if isinstance(value, Array) and value.ndim > 1:
                raise ValueError("boolean index array should have 1 dimension")
            y = where(key, value, self)
            self._meta = y._meta
            self.dask = y.dask
            self.name = y.name
            self._chunks = y.chunks
            return self
        else:
            raise NotImplementedError(
                "Item assignment with %s not supported" % type(key)
            )
    def __getitem__(self, index):
        # Field access, e.g. x['a'] or x[['a', 'b']]
        if isinstance(index, str) or (
            isinstance(index, list) and index and all(isinstance(i, str) for i in index)
        ):
            if isinstance(index, str):
                dt = self.dtype[index]
            else:
                dt = _make_sliced_dtype(self.dtype, index)
            if dt.shape:
                new_axis = list(range(self.ndim, self.ndim + len(dt.shape)))
                chunks = self.chunks + tuple((i,) for i in dt.shape)
                return self.map_blocks(
                    getitem, index, dtype=dt.base, chunks=chunks, new_axis=new_axis
                )
            else:
                return self.map_blocks(getitem, index, dtype=dt)
        if not isinstance(index, tuple):
            index = (index,)
        from .slicing import (
            normalize_index,
            slice_with_int_dask_array,
            slice_with_bool_dask_array,
        )
        index2 = normalize_index(index, self.shape)
        dependencies = {self.name}
        for i in index2:
            if isinstance(i, Array):
                dependencies.add(i.name)
        if any(isinstance(i, Array) and i.dtype.kind in "iu" for i in index2):
            self, index2 = slice_with_int_dask_array(self, index2)
        if any(isinstance(i, Array) and i.dtype == bool for i in index2):
            self, index2 = slice_with_bool_dask_array(self, index2)
        if all(isinstance(i, slice) and i == slice(None) for i in index2):
            return self
        out = "getitem-" + tokenize(self, index2)
        dsk, chunks = slice_array(out, self.name, self.chunks, index2)
        graph = HighLevelGraph.from_collections(out, dsk, dependencies=[self])
        meta = meta_from_array(self._meta, ndim=len(chunks))
        if np.isscalar(meta):
            meta = np.array(meta)
        return Array(graph, out, chunks, meta=meta)
    def _vindex(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is None for k in key):
            raise IndexError(
                "vindex does not support indexing with None (np.newaxis), "
                "got {}".format(key)
            )
        if all(isinstance(k, slice) for k in key):
            if all(
                k.indices(d) == slice(0, d).indices(d) for k, d in zip(key, self.shape)
            ):
                return self
            raise IndexError(
                "vindex requires at least one non-slice to vectorize over "
                "when the slices are not over the entire array (i.e, x[:]). "
                "Use normal slicing instead when only using slices. Got: {}".format(key)
            )
        return _vindex(self, *key)
    @property
    def vindex(self):
        
        return IndexCallable(self._vindex)
    def _blocks(self, index):
        from .slicing import normalize_index
        if not isinstance(index, tuple):
            index = (index,)
        if sum(isinstance(ind, (np.ndarray, list)) for ind in index) > 1:
            raise ValueError("Can only slice with a single list")
        if any(ind is None for ind in index):
            raise ValueError("Slicing with np.newaxis or None is not supported")
        index = normalize_index(index, self.numblocks)
        index = tuple(slice(k, k + 1) if isinstance(k, Number) else k for k in index)
        name = "blocks-" + tokenize(self, index)
        new_keys = np.array(self.__dask_keys__(), dtype=object)[index]
        chunks = tuple(
            tuple(np.array(c)[i].tolist()) for c, i in zip(self.chunks, index)
        )
        keys = list(product(*[range(len(c)) for c in chunks]))
        layer = {(name,) + key: tuple(new_keys[key].tolist()) for key in keys}
        graph = HighLevelGraph.from_collections(name, layer, dependencies=[self])
        return Array(graph, name, chunks, meta=self)
    @property
    def blocks(self):
        
        return IndexCallable(self._blocks)
    @property
    def partitions(self):
        
        return self.blocks
    @derived_from(np.ndarray)
    def dot(self, other):
        from .routines import tensordot
        return tensordot(self, other, axes=((self.ndim - 1,), (other.ndim - 2,)))
    @property
    def A(self):
        return self
    @property
    def T(self):
        return self.transpose()
    @derived_from(np.ndarray)
    def transpose(self, *axes):
        from .routines import transpose
        if not axes:
            axes = None
        elif len(axes) == 1 and isinstance(axes[0], Iterable):
            axes = axes[0]
        if (axes == tuple(range(self.ndim))) or (axes == tuple(range(-self.ndim, 0))):
            # no transpose necessary
            return self
        else:
            return transpose(self, axes=axes)
    @derived_from(np.ndarray)
    def ravel(self):
        from .routines import ravel
        return ravel(self)
    flatten = ravel
    @derived_from(np.ndarray)
    def choose(self, choices):
        from .routines import choose
        return choose(self, choices)
    @derived_from(np.ndarray)
    def reshape(self, *shape):
        from .reshape import reshape
        if len(shape) == 1 and not isinstance(shape[0], Number):
            shape = shape[0]
        return reshape(self, shape)
    def topk(self, k, axis=-1, split_every=None):
        
        from .reductions import topk
        return topk(self, k, axis=axis, split_every=split_every)
    def argtopk(self, k, axis=-1, split_every=None):
        
        from .reductions import argtopk
        return argtopk(self, k, axis=axis, split_every=split_every)
    def astype(self, dtype, **kwargs):
        
        # Scalars don't take `casting` or `copy` kwargs - as such we only pass
        # them to `map_blocks` if specified by user (different than defaults).
        extra = set(kwargs) - {"casting", "copy"}
        if extra:
            raise TypeError(
                "astype does not take the following keyword "
                "arguments: {0!s}".format(list(extra))
            )
        casting = kwargs.get("casting", "unsafe")
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self
        elif not np.can_cast(self.dtype, dtype, casting=casting):
            raise TypeError(
                "Cannot cast array from {0!r} to {1!r}"
                " according to the rule "
                "{2!r}".format(self.dtype, dtype, casting)
            )
        return self.map_blocks(chunk.astype, dtype=dtype, astype_dtype=dtype, **kwargs)
    def __abs__(self):
        return elemwise(operator.abs, self)
    def __add__(self, other):
        return elemwise(operator.add, self, other)
    def __radd__(self, other):
        return elemwise(operator.add, other, self)
    def __and__(self, other):
        return elemwise(operator.and_, self, other)
    def __rand__(self, other):
        return elemwise(operator.and_, other, self)
    def __div__(self, other):
        return elemwise(operator.div, self, other)
    def __rdiv__(self, other):
        return elemwise(operator.div, other, self)
    def __eq__(self, other):
        return elemwise(operator.eq, self, other)
    def __gt__(self, other):
        return elemwise(operator.gt, self, other)
    def __ge__(self, other):
        return elemwise(operator.ge, self, other)
    def __invert__(self):
        return elemwise(operator.invert, self)
    def __lshift__(self, other):
        return elemwise(operator.lshift, self, other)
    def __rlshift__(self, other):
        return elemwise(operator.lshift, other, self)
    def __lt__(self, other):
        return elemwise(operator.lt, self, other)
    def __le__(self, other):
        return elemwise(operator.le, self, other)
    def __mod__(self, other):
        return elemwise(operator.mod, self, other)
    def __rmod__(self, other):
        return elemwise(operator.mod, other, self)
    def __mul__(self, other):
        return elemwise(operator.mul, self, other)
    def __rmul__(self, other):
        return elemwise(operator.mul, other, self)
    def __ne__(self, other):
        return elemwise(operator.ne, self, other)
    def __neg__(self):
        return elemwise(operator.neg, self)
    def __or__(self, other):
        return elemwise(operator.or_, self, other)
    def __pos__(self):
        return self
    def __ror__(self, other):
        return elemwise(operator.or_, other, self)
    def __pow__(self, other):
        return elemwise(operator.pow, self, other)
    def __rpow__(self, other):
        return elemwise(operator.pow, other, self)
    def __rshift__(self, other):
        return elemwise(operator.rshift, self, other)
    def __rrshift__(self, other):
        return elemwise(operator.rshift, other, self)
    def __sub__(self, other):
        return elemwise(operator.sub, self, other)
    def __rsub__(self, other):
        return elemwise(operator.sub, other, self)
    def __truediv__(self, other):
        return elemwise(operator.truediv, self, other)
    def __rtruediv__(self, other):
        return elemwise(operator.truediv, other, self)
    def __floordiv__(self, other):
        return elemwise(operator.floordiv, self, other)
    def __rfloordiv__(self, other):
        return elemwise(operator.floordiv, other, self)
    def __xor__(self, other):
        return elemwise(operator.xor, self, other)
    def __rxor__(self, other):
        return elemwise(operator.xor, other, self)
    def __matmul__(self, other):
        from .routines import matmul
        return matmul(self, other)
    def __rmatmul__(self, other):
        from .routines import matmul
        return matmul(other, self)
    def __divmod__(self, other):
        from .ufunc import divmod
        return divmod(self, other)
    def __rdivmod__(self, other):
        from .ufunc import divmod
        return divmod(other, self)
    @derived_from(np.ndarray)
    def any(self, axis=None, keepdims=False, split_every=None, out=None):
        from .reductions import any
        return any(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)
    @derived_from(np.ndarray)
    def all(self, axis=None, keepdims=False, split_every=None, out=None):
        from .reductions import all
        return all(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)
    @derived_from(np.ndarray)
    def min(self, axis=None, keepdims=False, split_every=None, out=None):
        from .reductions import min
        return min(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)
    @derived_from(np.ndarray)
    def max(self, axis=None, keepdims=False, split_every=None, out=None):
        from .reductions import max
        return max(self, axis=axis, keepdims=keepdims, split_every=split_every, out=out)
    @derived_from(np.ndarray)
    def argmin(self, axis=None, split_every=None, out=None):
        from .reductions import argmin
        return argmin(self, axis=axis, split_every=split_every, out=out)
    @derived_from(np.ndarray)
    def argmax(self, axis=None, split_every=None, out=None):
        from .reductions import argmax
        return argmax(self, axis=axis, split_every=split_every, out=out)
    @derived_from(np.ndarray)
    def sum(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        from .reductions import sum
        return sum(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            split_every=split_every,
            out=out,
        )
    @derived_from(np.ndarray)
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None):
        from .reductions import trace
        return trace(self, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)
    @derived_from(np.ndarray)
    def prod(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        from .reductions import prod
        return prod(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            split_every=split_every,
            out=out,
        )
    @derived_from(np.ndarray)
    def mean(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        from .reductions import mean
        return mean(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            split_every=split_every,
            out=out,
        )
    @derived_from(np.ndarray)
    def std(
        self, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None
    ):
        from .reductions import std
        return std(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            ddof=ddof,
            split_every=split_every,
            out=out,
        )
    @derived_from(np.ndarray)
    def var(
        self, axis=None, dtype=None, keepdims=False, ddof=0, split_every=None, out=None
    ):
        from .reductions import var
        return var(
            self,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            ddof=ddof,
            split_every=split_every,
            out=out,
        )
    def moment(
        self,
        order,
        axis=None,
        dtype=None,
        keepdims=False,
        ddof=0,
        split_every=None,
        out=None,
    ):
        
        from .reductions import moment
        return moment(
            self,
            order,
            axis=axis,
            dtype=dtype,
            keepdims=keepdims,
            ddof=ddof,
            split_every=split_every,
            out=out,
        )
    @wraps(map_blocks)
    def map_blocks(self, func, *args, **kwargs):
        return map_blocks(func, self, *args, **kwargs)
    def map_overlap(self, func, depth, boundary=None, trim=True, **kwargs):
        
        from .overlap import map_overlap
        return map_overlap(self, func, depth, boundary, trim, **kwargs)
    def cumsum(self, axis, dtype=None, out=None):
        
        from .reductions import cumsum
        return cumsum(self, axis, dtype, out=out)
    def cumprod(self, axis, dtype=None, out=None):
        
        from .reductions import cumprod
        return cumprod(self, axis, dtype, out=out)
    @derived_from(np.ndarray)
    def squeeze(self, axis=None):
        from .routines import squeeze
        return squeeze(self, axis)
    def rechunk(self, chunks="auto", threshold=None, block_size_limit=None):
        
        from . import rechunk  # avoid circular import
        return rechunk(self, chunks, threshold, block_size_limit)
    @property
    def real(self):
        from .ufunc import real
        return real(self)
    @property
    def imag(self):
        from .ufunc import imag
        return imag(self)
    def conj(self):
        from .ufunc import conj
        return conj(self)
    @derived_from(np.ndarray)
    def clip(self, min=None, max=None):
        from .ufunc import clip
        return clip(self, min, max)
    def view(self, dtype=None, order="C"):
        
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)
        mult = self.dtype.itemsize / dtype.itemsize
        if order == "C":
            chunks = self.chunks[:-1] + (
                tuple(ensure_int(c * mult) for c in self.chunks[-1]),
            )
        elif order == "F":
            chunks = (
                tuple(ensure_int(c * mult) for c in self.chunks[0]),
            ) + self.chunks[1:]
        else:
            raise ValueError("Order must be one of 'C' or 'F'")
        return self.map_blocks(
            chunk.view, dtype, order=order, dtype=dtype, chunks=chunks
        )
    @derived_from(np.ndarray)
    def swapaxes(self, axis1, axis2):
        from .routines import swapaxes
        return swapaxes(self, axis1, axis2)
    @derived_from(np.ndarray)
    def round(self, decimals=0):
        from .routines import round
        return round(self, decimals=decimals)
    def copy(self):
        
        if self.npartitions == 1:
            return self.map_blocks(M.copy)
        else:
            return Array(self.dask, self.name, self.chunks, meta=self)
    def __deepcopy__(self, memo):
        c = self.copy()
        memo[id(self)] = c
        return c
    def to_delayed(self, optimize_graph=True):
        
        keys = self.__dask_keys__()
        graph = self.__dask_graph__()
        if optimize_graph:
            graph = self.__dask_optimize__(graph, keys)  # TODO, don't collape graph
            name = "delayed-" + self.name
            graph = HighLevelGraph.from_collections(name, graph, dependencies=())
        L = ndeepmap(self.ndim, lambda k: Delayed(k, graph), keys)
        return np.array(L, dtype=object)
    @derived_from(np.ndarray)
    def repeat(self, repeats, axis=None):
        from .creation import repeat
        return repeat(self, repeats, axis=axis)
    @derived_from(np.ndarray)
    def nonzero(self):
        from .routines import nonzero
        return nonzero(self)
    def to_zarr(self, *args, **kwargs):
        
        return to_zarr(self, *args, **kwargs)
    def to_tiledb(self, uri, *args, **kwargs):
        
        from .tiledb_io import to_tiledb
        return to_tiledb(self, uri, *args, **kwargs)
def ensure_int(f):
    i = int(f)
    if i != f:
        raise ValueError("Could not coerce %f to integer" % f)
    return i
def normalize_chunks(chunks, shape=None, limit=None, dtype=None, previous_chunks=None):
    
    if dtype and not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    if chunks is None:
        raise ValueError(CHUNKS_NONE_ERROR_MESSAGE)
    if isinstance(chunks, list):
        chunks = tuple(chunks)
    if isinstance(chunks, (Number, str)):
        chunks = (chunks,) * len(shape)
    if isinstance(chunks, dict):
        chunks = tuple(chunks.get(i, None) for i in range(len(shape)))
    if isinstance(chunks, np.ndarray):
        chunks = chunks.tolist()
    if not chunks and shape and all(s == 0 for s in shape):
        chunks = ((0,),) * len(shape)
    if (
        shape
        and len(shape) == 1
        and len(chunks) > 1
        and all(isinstance(c, (Number, str)) for c in chunks)
    ):
        chunks = (chunks,)
    if shape and len(chunks) != len(shape):
        raise ValueError(
            "Chunks and shape must be of the same length/dimension. "
            "Got chunks=%s, shape=%s" % (chunks, shape)
        )
    if -1 in chunks or None in chunks:
        chunks = tuple(s if c == -1 or c is None else c for c, s in zip(chunks, shape))
    # If specifying chunk size in bytes, use that value to set the limit.
    # Verify there is only one consistent value of limit or chunk-bytes used.
    for c in chunks:
        if isinstance(c, str) and c != "auto":
            parsed = parse_bytes(c)
            if limit is None:
                limit = parsed
            elif parsed != limit:
                raise ValueError(
                    "Only one consistent value of limit or chunk is allowed."
                    "Used %s != %s" % (parsed, limit)
                )
    # Substitute byte limits with 'auto' now that limit is set.
    chunks = tuple("auto" if isinstance(c, str) and c != "auto" else c for c in chunks)
    if any(c == "auto" for c in chunks):
        chunks = auto_chunks(chunks, shape, limit, dtype, previous_chunks)
    if shape is not None:
        chunks = tuple(c if c not in {None, -1} else s for c, s in zip(chunks, shape))
    if chunks and shape is not None:
        chunks = sum(
            (
                blockdims_from_blockshape((s,), (c,))
                if not isinstance(c, (tuple, list))
                else (c,)
                for s, c in zip(shape, chunks)
            ),
            (),
        )
    for c in chunks:
        if not c:
            raise ValueError(
                "Empty tuples are not allowed in chunks. Express "
                "zero length dimensions with 0(s) in chunks"
            )
    if shape is not None:
        if len(chunks) != len(shape):
            raise ValueError(
                "Input array has %d dimensions but the supplied "
                "chunks has only %d dimensions" % (len(shape), len(chunks))
            )
        if not all(
            c == s or (math.isnan(c) or math.isnan(s))
            for c, s in zip(map(sum, chunks), shape)
        ):
            raise ValueError(
                "Chunks do not add up to shape. "
                "Got chunks=%s, shape=%s" % (chunks, shape)
            )
    return tuple(tuple(int(x) if not math.isnan(x) else x for x in c) for c in chunks)
def auto_chunks(chunks, shape, limit, dtype, previous_chunks=None):
    
    if previous_chunks is not None:
        previous_chunks = tuple(
            c if isinstance(c, tuple) else (c,) for c in previous_chunks
        )
    chunks = list(chunks)
    autos = {i for i, c in enumerate(chunks) if c == "auto"}
    if not autos:
        return tuple(chunks)
    if limit is None:
        limit = config.get("array.chunk-size")
    if isinstance(limit, str):
        limit = parse_bytes(limit)
    if dtype is None:
        raise TypeError("DType must be known for auto-chunking")
    if dtype.hasobject:
        raise NotImplementedError(
            "Can not use auto rechunking with object dtype. "
            "We are unable to estimate the size in bytes of object data"
        )
    for x in tuple(chunks) + tuple(shape):
        if (
            isinstance(x, Number)
            and np.isnan(x)
            or isinstance(x, tuple)
            and np.isnan(x).any()
        ):
            raise ValueError(
                "Can not perform automatic rechunking with unknown "
                "(nan) chunk sizes.%s" % unknown_chunk_message
            )
    limit = max(1, limit)
    largest_block = np.prod(
        [cs if isinstance(cs, Number) else max(cs) for cs in chunks if cs != "auto"]
    )
    if previous_chunks:
        # Base ideal ratio on the median chunk size of the previous chunks
        result = {a: np.median(previous_chunks[a]) for a in autos}
        ideal_shape = []
        for i, s in enumerate(shape):
            chunk_frequencies = frequencies(previous_chunks[i])
            mode, count = max(chunk_frequencies.items(), key=lambda kv: kv[1])
            if mode > 1 and count >= len(previous_chunks[i]) / 2:
                ideal_shape.append(mode)
            else:
                ideal_shape.append(s)
        # How much larger or smaller the ideal chunk size is relative to what we have now
        multiplier = (
            limit / dtype.itemsize / largest_block / np.prod(list(result.values()))
        )
        last_multiplier = 0
        last_autos = set()
        while (
            multiplier != last_multiplier or autos != last_autos
        ):  # while things change
            last_multiplier = multiplier  # record previous values
            last_autos = set(autos)  # record previous values
            # Expand or contract each of the dimensions appropriately
            for a in sorted(autos):
                proposed = result[a] * multiplier ** (1 / len(autos))
                if proposed > shape[a]:  # we've hit the shape boundary
                    autos.remove(a)
                    largest_block *= shape[a]
                    chunks[a] = shape[a]
                    del result[a]
                else:
                    result[a] = round_to(proposed, ideal_shape[a])
            # recompute how much multiplier we have left, repeat
            multiplier = (
                limit / dtype.itemsize / largest_block / np.prod(list(result.values()))
            )
        for k, v in result.items():
            chunks[k] = v
        return tuple(chunks)
    else:
        size = (limit / dtype.itemsize / largest_block) ** (1 / len(autos))
        small = [i for i in autos if shape[i] < size]
        if small:
            for i in small:
                chunks[i] = (shape[i],)
            return auto_chunks(chunks, shape, limit, dtype)
        for i in autos:
            chunks[i] = round_to(size, shape[i])
        return tuple(chunks)
def round_to(c, s):
    
    if c <= s:
        try:
            return max(f for f in factors(s) if c / 2 <= f <= c)
        except ValueError:  # no matching factors within factor of two
            return max(1, int(c))
    else:
        return c // s * s
def _get_chunk_shape(a):
    s = np.asarray(a.shape, dtype=int)
    return s[len(s) * (None,) + (slice(None),)]
def from_array(
    x,
    chunks="auto",
    name=None,
    lock=False,
    asarray=None,
    fancy=True,
    getitem=None,
    meta=None,
):
    
    if isinstance(x, Array):
        raise ValueError(
            "Array is already a dask array. Use 'asarray' or " "'rechunk' instead."
        )
    if isinstance(x, (list, tuple, memoryview) + np.ScalarType):
        x = np.array(x)
    if asarray is None:
        asarray = not hasattr(x, "__array_function__")
    previous_chunks = getattr(x, "chunks", None)
    chunks = normalize_chunks(
        chunks, x.shape, dtype=x.dtype, previous_chunks=previous_chunks
    )
    if name in (None, True):
        token = tokenize(x, chunks)
        original_name = "array-original-" + token
        name = name or "array-" + token
    elif name is False:
        original_name = name = "array-" + str(uuid.uuid1())
    else:
        original_name = name
    if lock is True:
        lock = SerializableLock()
    # Always use the getter for h5py etc. Not using isinstance(x, np.ndarray)
    # because np.matrix is a subclass of np.ndarray.
    if type(x) is np.ndarray and all(len(c) == 1 for c in chunks):
        # No slicing needed
        dsk = {(name,) + (0,) * x.ndim: x}
    else:
        if getitem is None:
            if type(x) is np.ndarray and not lock:
                # simpler and cleaner, but missing all the nuances of getter
                getitem = operator.getitem
            elif fancy:
                getitem = getter
            else:
                getitem = getter_nofancy
        dsk = getem(
            original_name,
            chunks,
            getitem=getitem,
            shape=x.shape,
            out_name=name,
            lock=lock,
            asarray=asarray,
            dtype=x.dtype,
        )
        dsk[original_name] = x
    # Workaround for TileDB, its indexing is 1-based,
    # and doesn't seems to support 0-length slicing
    if x.__class__.__module__.split(".")[0] == "tiledb" and hasattr(x, "_ctx_"):
        return Array(dsk, name, chunks, dtype=x.dtype)
    if meta is None:
        meta = x
    return Array(dsk, name, chunks, meta=meta, dtype=getattr(x, "dtype", None))
def from_zarr(
    url, component=None, storage_options=None, chunks=None, name=None, **kwargs
):
    
    import zarr
    storage_options = storage_options or {}
    if isinstance(url, zarr.Array):
        z = url
    elif isinstance(url, str):
        from ..bytes.core import get_mapper
        mapper = get_mapper(url, **storage_options)
        z = zarr.Array(mapper, read_only=True, path=component, **kwargs)
    else:
        mapper = url
        z = zarr.Array(mapper, read_only=True, path=component, **kwargs)
    chunks = chunks if chunks is not None else z.chunks
    if name is None:
        name = "from-zarr-" + tokenize(z, component, storage_options, chunks, **kwargs)
    return from_array(z, chunks, name=name)
def to_zarr(
    arr,
    url,
    component=None,
    storage_options=None,
    overwrite=False,
    compute=True,
    return_stored=False,
    **kwargs
):
    
    import zarr
    if np.isnan(arr.shape).any():
        raise ValueError(
            "Saving a dask array with unknown chunk sizes is not "
            "currently supported by Zarr.%s" % unknown_chunk_message
        )
    if isinstance(url, zarr.Array):
        z = url
        if isinstance(z.store, (dict, zarr.DictStore)) and "distributed" in config.get(
            "scheduler", ""
        ):
            raise RuntimeError(
                "Cannot store into in memory Zarr Array using "
                "the Distributed Scheduler."
            )
        arr = arr.rechunk(z.chunks)
        return arr.store(z, lock=False, compute=compute, return_stored=return_stored)
    if not _check_regular_chunks(arr.chunks):
        raise ValueError(
            "Attempt to save array to zarr with irregular "
            "chunking, please call `arr.rechunk(...)` first."
        )
    storage_options = storage_options or {}
    if isinstance(url, str):
        from ..bytes.core import get_mapper
        mapper = get_mapper(url, **storage_options)
    else:
        # assume the object passed is already a mapper
        mapper = url
    chunks = [c[0] for c in arr.chunks]
    z = zarr.create(
        shape=arr.shape,
        chunks=chunks,
        dtype=arr.dtype,
        store=mapper,
        path=component,
        overwrite=overwrite,
        **kwargs
    )
    return arr.store(z, lock=False, compute=compute, return_stored=return_stored)
def _check_regular_chunks(chunkset):
    
    for chunks in chunkset:
        if len(chunks) == 1:
            continue
        if len(set(chunks[:-1])) > 1:
            return False
        if chunks[-1] > chunks[0]:
            return False
    return True
def from_delayed(value, shape, dtype=None, meta=None, name=None):
    
    from ..delayed import delayed, Delayed
    if not isinstance(value, Delayed) and hasattr(value, "key"):
        value = delayed(value)
    name = name or "from-value-" + tokenize(value, shape, dtype, meta)
    dsk = {(name,) + (0,) * len(shape): value.key}
    chunks = tuple((d,) for d in shape)
    # TODO: value._key may not be the name of the layer in value.dask
    # This should be fixed after we build full expression graphs
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[value])
    return Array(graph, name, chunks, dtype=dtype, meta=meta)
def from_func(func, shape, dtype=None, name=None, args=(), kwargs={}):
    
    name = name or "from_func-" + tokenize(func, shape, dtype, args, kwargs)
    if args or kwargs:
        func = partial(func, *args, **kwargs)
    dsk = {(name,) + (0,) * len(shape): (func,)}
    chunks = tuple((i,) for i in shape)
    return Array(dsk, name, chunks, dtype)
def common_blockdim(blockdims):
    
    if not any(blockdims):
        return ()
    non_trivial_dims = set([d for d in blockdims if len(d) > 1])
    if len(non_trivial_dims) == 1:
        return first(non_trivial_dims)
    if len(non_trivial_dims) == 0:
        return max(blockdims, key=first)
    if np.isnan(sum(map(sum, blockdims))):
        raise ValueError(
            "Arrays chunk sizes (%s) are unknown.\n\n"
            "A possible solution:\n"
            "  x.compute_chunk_sizes()" % blockdims
        )
    if len(set(map(sum, non_trivial_dims))) > 1:
        raise ValueError("Chunks do not add up to same value", blockdims)
    # We have multiple non-trivial chunks on this axis
    # e.g. (5, 2) and (4, 3)
    # We create a single chunk tuple with the same total length
    # that evenly divides both, e.g. (4, 1, 2)
    # To accomplish this we walk down all chunk tuples together, finding the
    # smallest element, adding it to the output, and subtracting it from all
    # other elements and remove the element itself.  We stop once we have
    # burned through all of the chunk tuples.
    # For efficiency's sake we reverse the lists so that we can pop off the end
    rchunks = [list(ntd)[::-1] for ntd in non_trivial_dims]
    total = sum(first(non_trivial_dims))
    i = 0
    out = []
    while i < total:
        m = min(c[-1] for c in rchunks)
        out.append(m)
        for c in rchunks:
            c[-1] -= m
            if c[-1] == 0:
                c.pop()
        i += m
    return tuple(out)
def unify_chunks(*args, **kwargs):
    
    if not args:
        return {}, []
    arginds = [
        (asanyarray(a) if ind is not None else a, ind) for a, ind in partition(2, args)
    ]  # [x, ij, y, jk]
    args = list(concat(arginds))  # [(x, ij), (y, jk)]
    warn = kwargs.get("warn", True)
    arrays, inds = zip(*arginds)
    if all(ind is None for ind in inds):
        return {}, list(arrays)
    if all(ind == inds[0] for ind in inds) and all(
        a.chunks == arrays[0].chunks for a in arrays
    ):
        return dict(zip(inds[0], arrays[0].chunks)), arrays
    nameinds = [(a.name if i is not None else a, i) for a, i in arginds]
    blockdim_dict = {a.name: a.chunks for a, ind in arginds if ind is not None}
    chunkss = broadcast_dimensions(nameinds, blockdim_dict, consolidate=common_blockdim)
    max_parts = max(arg.npartitions for arg, ind in arginds if ind is not None)
    nparts = np.prod(list(map(len, chunkss.values())))
    if warn and nparts and nparts >= max_parts * 10:
        warnings.warn(
            "Increasing number of chunks by factor of %d" % (nparts / max_parts),
            PerformanceWarning,
            stacklevel=3,
        )
    arrays = []
    for a, i in arginds:
        if i is None:
            arrays.append(a)
        else:
            chunks = tuple(
                chunkss[j]
                if a.shape[n] > 1
                else a.shape[n]
                if not np.isnan(sum(chunkss[j]))
                else None
                for n, j in enumerate(i)
            )
            if chunks != a.chunks and all(a.chunks):
                arrays.append(a.rechunk(chunks))
            else:
                arrays.append(a)
    return chunkss, arrays
def unpack_singleton(x):
    
    while isinstance(x, (list, tuple)):
        try:
            x = x[0]
        except (IndexError, TypeError, KeyError):
            break
    return x
def block(arrays, allow_unknown_chunksizes=False):
    
    # This was copied almost verbatim from numpy.core.shape_base.block
    # See numpy license at https://github.com/numpy/numpy/blob/master/LICENSE.txt
    # or NUMPY_LICENSE.txt within this directory
    def atleast_nd(x, ndim):
        x = asanyarray(x)
        diff = max(ndim - x.ndim, 0)
        return x[(None,) * diff + (Ellipsis,)]
    def format_index(index):
        return "arrays" + "".join("[{}]".format(i) for i in index)
    rec = _Recurser(recurse_if=lambda x: type(x) is list)
    # ensure that the lists are all matched in depth
    list_ndim = None
    any_empty = False
    for index, value, entering in rec.walk(arrays):
        if type(value) is tuple:
            # not strictly necessary, but saves us from:
            #  - more than one way to do things - no point treating tuples like
            #    lists
            #  - horribly confusing behaviour that results when tuples are
            #    treated like ndarray
            raise TypeError(
                "{} is a tuple. "
                "Only lists can be used to arrange blocks, and np.block does "
                "not allow implicit conversion from tuple to ndarray.".format(
                    format_index(index)
                )
            )
        if not entering:
            curr_depth = len(index)
        elif len(value) == 0:
            curr_depth = len(index) + 1
            any_empty = True
        else:
            continue
        if list_ndim is not None and list_ndim != curr_depth:
            raise ValueError(
                "List depths are mismatched. First element was at depth {}, "
                "but there is an element at depth {} ({})".format(
                    list_ndim, curr_depth, format_index(index)
                )
            )
        list_ndim = curr_depth
    # do this here so we catch depth mismatches first
    if any_empty:
        raise ValueError("Lists cannot be empty")
    # convert all the arrays to ndarrays
    arrays = rec.map_reduce(arrays, f_map=asanyarray, f_reduce=list)
    # determine the maximum dimension of the elements
    elem_ndim = rec.map_reduce(arrays, f_map=lambda xi: xi.ndim, f_reduce=max)
    ndim = max(list_ndim, elem_ndim)
    # first axis to concatenate along
    first_axis = ndim - list_ndim
    # Make all the elements the same dimension
    arrays = rec.map_reduce(
        arrays, f_map=lambda xi: atleast_nd(xi, ndim), f_reduce=list
    )
    # concatenate innermost lists on the right, outermost on the left
    return rec.map_reduce(
        arrays,
        f_reduce=lambda xs, axis: concatenate(
            list(xs), axis=axis, allow_unknown_chunksizes=allow_unknown_chunksizes
        ),
        f_kwargs=lambda axis: dict(axis=(axis + 1)),
        axis=first_axis,
    )
def concatenate(seq, axis=0, allow_unknown_chunksizes=False):
    
    from . import wrap
    seq = [asarray(a) for a in seq]
    if not seq:
        raise ValueError("Need array(s) to concatenate")
    meta = np.concatenate([meta_from_array(s) for s in seq], axis=axis)
    # Promote types to match meta
    seq = [a.astype(meta.dtype) for a in seq]
    # Find output array shape
    ndim = len(seq[0].shape)
    shape = tuple(
        sum((a.shape[i] for a in seq)) if i == axis else seq[0].shape[i]
        for i in range(ndim)
    )
    # Drop empty arrays
    seq2 = [a for a in seq if a.size]
    if not seq2:
        seq2 = seq
    if axis < 0:
        axis = ndim + axis
    if axis >= ndim:
        msg = (
            "Axis must be less than than number of dimensions"
            "\nData has %d dimensions, but got axis=%d"
        )
        raise ValueError(msg % (ndim, axis))
    n = len(seq2)
    if n == 0:
        try:
            return wrap.empty_like(meta, shape=shape, chunks=shape, dtype=meta.dtype)
        except TypeError:
            return wrap.empty(shape, chunks=shape, dtype=meta.dtype)
    elif n == 1:
        return seq2[0]
    if not allow_unknown_chunksizes and not all(
        i == axis or all(x.shape[i] == seq2[0].shape[i] for x in seq2)
        for i in range(ndim)
    ):
        if any(map(np.isnan, seq2[0].shape)):
            raise ValueError(
                "Tried to concatenate arrays with unknown"
                " shape %s.\n\nTwo solutions:\n"
                "  1. Force concatenation pass"
                " allow_unknown_chunksizes=True.\n"
                "  2. Compute shapes with "
                "[x.compute_chunk_sizes() for x in seq]" % str(seq2[0].shape)
            )
        raise ValueError("Shapes do not align: %s", [x.shape for x in seq2])
    inds = [list(range(ndim)) for i in range(n)]
    for i, ind in enumerate(inds):
        ind[axis] = -(i + 1)
    uc_args = list(concat(zip(seq2, inds)))
    _, seq2 = unify_chunks(*uc_args, warn=False)
    bds = [a.chunks for a in seq2]
    chunks = (
        seq2[0].chunks[:axis]
        + (sum([bd[axis] for bd in bds], ()),)
        + seq2[0].chunks[axis + 1 :]
    )
    cum_dims = [0] + list(accumulate(add, [len(a.chunks[axis]) for a in seq2]))
    names = [a.name for a in seq2]
    name = "concatenate-" + tokenize(names, axis)
    keys = list(product([name], *[range(len(bd)) for bd in chunks]))
    values = [
        (names[bisect(cum_dims, key[axis + 1]) - 1],)
        + key[1 : axis + 1]
        + (key[axis + 1] - cum_dims[bisect(cum_dims, key[axis + 1]) - 1],)
        + key[axis + 2 :]
        for key in keys
    ]
    dsk = dict(zip(keys, values))
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=seq2)
    return Array(graph, name, chunks, meta=meta)
def load_store_chunk(x, out, index, lock, return_stored, load_stored):
    
    result = None
    if return_stored and not load_stored:
        result = out
    if lock:
        lock.acquire()
    try:
        if x is not None:
            out[index] = np.asanyarray(x)
        if return_stored and load_stored:
            result = out[index]
    finally:
        if lock:
            lock.release()
    return result
def store_chunk(x, out, index, lock, return_stored):
    return load_store_chunk(x, out, index, lock, return_stored, False)
def load_chunk(out, index, lock):
    return load_store_chunk(None, out, index, lock, True, True)
def insert_to_ooc(
    arr, out, lock=True, region=None, return_stored=False, load_stored=False, tok=None
):
    
    if lock is True:
        lock = Lock()
    slices = slices_from_chunks(arr.chunks)
    if region:
        slices = [fuse_slice(region, slc) for slc in slices]
    name = "store-%s" % (tok or str(uuid.uuid1()))
    func = store_chunk
    args = ()
    if return_stored and load_stored:
        name = "load-%s" % name
        func = load_store_chunk
        args = args + (load_stored,)
    dsk = {
        (name,) + t[1:]: (func, t, out, slc, lock, return_stored) + args
        for t, slc in zip(core.flatten(arr.__dask_keys__()), slices)
    }
    return dsk
def retrieve_from_ooc(keys, dsk_pre, dsk_post=None):
    
    if not dsk_post:
        dsk_post = {k: k for k in keys}
    load_dsk = {
        ("load-" + k[0],) + k[1:]: (load_chunk, dsk_post[k]) + dsk_pre[k][3:-1]
        for k in keys
    }
    return load_dsk
def asarray(a, **kwargs):
    
    if isinstance(a, Array):
        return a
    elif hasattr(a, "to_dask_array"):
        return a.to_dask_array()
    elif type(a).__module__.startswith("xarray.") and hasattr(a, "data"):
        return asarray(a.data)
    elif isinstance(a, (list, tuple)) and any(isinstance(i, Array) for i in a):
        return stack(a)
    elif not isinstance(getattr(a, "shape", None), Iterable):
        a = np.asarray(a)
    return from_array(a, getitem=getter_inline, **kwargs)
def asanyarray(a):
    
    if isinstance(a, Array):
        return a
    elif hasattr(a, "to_dask_array"):
        return a.to_dask_array()
    elif type(a).__module__.startswith("xarray.") and hasattr(a, "data"):
        return asanyarray(a.data)
    elif isinstance(a, (list, tuple)) and any(isinstance(i, Array) for i in a):
        a = stack(a)
    elif not isinstance(getattr(a, "shape", None), Iterable):
        a = np.asanyarray(a)
    return from_array(a, chunks=a.shape, getitem=getter_inline, asarray=False)
def is_scalar_for_elemwise(arg):
    
    # the second half of shape_condition is essentially just to ensure that
    # dask series / frame are treated as scalars in elemwise.
    maybe_shape = getattr(arg, "shape", None)
    shape_condition = not isinstance(maybe_shape, Iterable) or any(
        is_dask_collection(x) for x in maybe_shape
    )
    return (
        np.isscalar(arg)
        or shape_condition
        or isinstance(arg, np.dtype)
        or (isinstance(arg, np.ndarray) and arg.ndim == 0)
    )
def broadcast_shapes(*shapes):
    
    if len(shapes) == 1:
        return shapes[0]
    out = []
    for sizes in zip_longest(*map(reversed, shapes), fillvalue=-1):
        if np.isnan(sizes).any():
            dim = np.nan
        else:
            dim = 0 if 0 in sizes else np.max(sizes)
        if any(i not in [-1, 0, 1, dim] and not np.isnan(i) for i in sizes):
            raise ValueError(
                "operands could not be broadcast together with "
                "shapes {0}".format(" ".join(map(str, shapes)))
            )
        out.append(dim)
    return tuple(reversed(out))
def elemwise(op, *args, **kwargs):
    
    out = kwargs.pop("out", None)
    if not set(["name", "dtype"]).issuperset(kwargs):
        msg = "%s does not take the following keyword arguments %s"
        raise TypeError(
            msg % (op.__name__, str(sorted(set(kwargs) - set(["name", "dtype"]))))
        )
    args = [np.asarray(a) if isinstance(a, (list, tuple)) else a for a in args]
    shapes = []
    for arg in args:
        shape = getattr(arg, "shape", ())
        if any(is_dask_collection(x) for x in shape):
            # Want to excluded Delayed shapes and dd.Scalar
            shape = ()
        shapes.append(shape)
    shapes = [s if isinstance(s, Iterable) else () for s in shapes]
    out_ndim = len(
        broadcast_shapes(*shapes)
    )  # Raises ValueError if dimensions mismatch
    expr_inds = tuple(range(out_ndim))[::-1]
    need_enforce_dtype = False
    if "dtype" in kwargs:
        dt = kwargs["dtype"]
    else:
        # We follow NumPy's rules for dtype promotion, which special cases
        # scalars and 0d ndarrays (which it considers equivalent) by using
        # their values to compute the result dtype:
        # https://github.com/numpy/numpy/issues/6240
        # We don't inspect the values of 0d dask arrays, because these could
        # hold potentially very expensive calculations. Instead, we treat
        # them just like other arrays, and if necessary cast the result of op
        # to match.
        vals = [
            np.empty((1,) * max(1, a.ndim), dtype=a.dtype)
            if not is_scalar_for_elemwise(a)
            else a
            for a in args
        ]
        try:
            dt = apply_infer_dtype(op, vals, {}, "elemwise", suggest_dtype=False)
        except Exception:
            return NotImplemented
        need_enforce_dtype = any(
            not is_scalar_for_elemwise(a) and a.ndim == 0 for a in args
        )
    name = kwargs.get("name", None) or "%s-%s" % (funcname(op), tokenize(op, dt, *args))
    blockwise_kwargs = dict(dtype=dt, name=name, token=funcname(op).strip("_"))
    if need_enforce_dtype:
        blockwise_kwargs["enforce_dtype"] = dt
        blockwise_kwargs["enforce_dtype_function"] = op
        op = _enforce_dtype
    result = blockwise(
        op,
        expr_inds,
        *concat(
            (a, tuple(range(a.ndim)[::-1]) if not is_scalar_for_elemwise(a) else None)
            for a in args
        ),
        **blockwise_kwargs
    )
    return handle_out(out, result)
def handle_out(out, result):
    
    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        elif len(out) > 1:
            raise NotImplementedError("The out parameter is not fully supported")
        else:
            out = None
    if isinstance(out, Array):
        if out.shape != result.shape:
            raise ValueError(
                "Mismatched shapes between result and out parameter. "
                "out=%s, result=%s" % (str(out.shape), str(result.shape))
            )
        out._chunks = result.chunks
        out.dask = result.dask
        out._meta = result._meta
        out.name = result.name
    elif out is not None:
        msg = (
            "The out parameter is not fully supported."
            " Received type %s, expected Dask Array" % type(out).__name__
        )
        raise NotImplementedError(msg)
    else:
        return result
def _enforce_dtype(*args, **kwargs):
    
    dtype = kwargs.pop("enforce_dtype")
    function = kwargs.pop("enforce_dtype_function")
    result = function(*args, **kwargs)
    if hasattr(result, "dtype") and dtype != result.dtype and dtype != object:
        if not np.can_cast(result, dtype, casting="same_kind"):
            raise ValueError(
                "Inferred dtype from function %r was %r "
                "but got %r, which can't be cast using "
                "casting='same_kind'"
                % (funcname(function), str(dtype), str(result.dtype))
            )
        if np.isscalar(result):
            # scalar astype method doesn't take the keyword arguments, so
            # have to convert via 0-dimensional array and back.
            result = result.astype(dtype)
        else:
            try:
                result = result.astype(dtype, copy=False)
            except TypeError:
                # Missing copy kwarg
                result = result.astype(dtype)
    return result
def broadcast_to(x, shape, chunks=None):
    
    x = asarray(x)
    shape = tuple(shape)
    if x.shape == shape and (chunks is None or chunks == x.chunks):
        return x
    ndim_new = len(shape) - x.ndim
    if ndim_new < 0 or any(
        new != old for new, old in zip(shape[ndim_new:], x.shape) if old != 1
    ):
        raise ValueError("cannot broadcast shape %s to shape %s" % (x.shape, shape))
    if chunks is None:
        chunks = tuple((s,) for s in shape[:ndim_new]) + tuple(
            bd if old > 1 else (new,)
            for bd, old, new in zip(x.chunks, x.shape, shape[ndim_new:])
        )
    else:
        chunks = normalize_chunks(
            chunks, shape, dtype=x.dtype, previous_chunks=x.chunks
        )
        for old_bd, new_bd in zip(x.chunks, chunks[ndim_new:]):
            if old_bd != new_bd and old_bd != (1,):
                raise ValueError(
                    "cannot broadcast chunks %s to chunks %s: "
                    "new chunks must either be along a new "
                    "dimension or a dimension of size 1" % (x.chunks, chunks)
                )
    name = "broadcast_to-" + tokenize(x, shape, chunks)
    dsk = {}
    enumerated_chunks = product(*(enumerate(bds) for bds in chunks))
    for new_index, chunk_shape in (zip(*ec) for ec in enumerated_chunks):
        old_index = tuple(
            0 if bd == (1,) else i for bd, i in zip(x.chunks, new_index[ndim_new:])
        )
        old_key = (x.name,) + old_index
        new_key = (name,) + new_index
        dsk[new_key] = (np.broadcast_to, old_key, quote(chunk_shape))
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[x])
    return Array(graph, name, chunks, dtype=x.dtype)
@derived_from(np)
def broadcast_arrays(*args, **kwargs):
    subok = bool(kwargs.pop("subok", False))
    to_array = asanyarray if subok else asarray
    args = tuple(to_array(e) for e in args)
    if kwargs:
        raise TypeError("unsupported keyword argument(s) provided")
    shape = broadcast_shapes(*(e.shape for e in args))
    chunks = broadcast_chunks(*(e.chunks for e in args))
    result = [broadcast_to(e, shape=shape, chunks=chunks) for e in args]
    return result
def offset_func(func, offset, *args):
    
    def _offset(*args):
        args2 = list(map(add, args, offset))
        return func(*args2)
    with ignoring(Exception):
        _offset.__name__ = "offset_" + func.__name__
    return _offset
def chunks_from_arrays(arrays):
    
    if not arrays:
        return ()
    result = []
    dim = 0
    def shape(x):
        try:
            return x.shape
        except AttributeError:
            return (1,)
    while isinstance(arrays, (list, tuple)):
        result.append(tuple([shape(deepfirst(a))[dim] for a in arrays]))
        arrays = arrays[0]
        dim += 1
    return tuple(result)
def deepfirst(seq):
    
    if not isinstance(seq, (list, tuple)):
        return seq
    else:
        return deepfirst(seq[0])
def shapelist(a):
    
    if type(a) is list:
        return tuple([len(a)] + list(shapelist(a[0])))
    else:
        return ()
def reshapelist(shape, seq):
    
    if len(shape) == 1:
        return list(seq)
    else:
        n = int(len(seq) / shape[0])
        return [reshapelist(shape[1:], part) for part in partition(n, seq)]
def transposelist(arrays, axes, extradims=0):
    
    if len(axes) != ndimlist(arrays):
        raise ValueError("Length of axes should equal depth of nested arrays")
    if extradims < 0:
        raise ValueError("`newdims` should be positive")
    if len(axes) > len(set(axes)):
        raise ValueError("`axes` should be unique")
    ndim = max(axes) + 1
    shape = shapelist(arrays)
    newshape = [
        shape[axes.index(i)] if i in axes else 1 for i in range(ndim + extradims)
    ]
    result = list(core.flatten(arrays))
    return reshapelist(newshape, result)
def stack(seq, axis=0):
    
    from . import wrap
    seq = [asarray(a) for a in seq]
    if not seq:
        raise ValueError("Need array(s) to stack")
    if not all(x.shape == seq[0].shape for x in seq):
        idx = np.where(np.asanyarray([x.shape for x in seq]) != seq[0].shape)[0]
        raise ValueError(
            "Stacked arrays must have the same shape. "
            "The first {0} had shape {1}, while array "
            "{2} has shape {3}".format(
                idx[0], seq[0].shape, idx[0] + 1, seq[idx[0]].shape
            )
        )
    meta = np.stack([meta_from_array(a) for a in seq], axis=axis)
    seq = [x.astype(meta.dtype) for x in seq]
    ndim = meta.ndim - 1
    if axis < 0:
        axis = ndim + axis + 1
    shape = tuple(
        len(seq)
        if i == axis
        else (seq[0].shape[i] if i < axis else seq[0].shape[i - 1])
        for i in range(meta.ndim)
    )
    seq2 = [a for a in seq if a.size]
    if not seq2:
        seq2 = seq
    n = len(seq2)
    if n == 0:
        try:
            return wrap.empty_like(meta, shape=shape, chunks=shape, dtype=meta.dtype)
        except TypeError:
            return wrap.empty(shape, chunks=shape, dtype=meta.dtype)
    ind = list(range(ndim))
    uc_args = list(concat((x, ind) for x in seq2))
    _, seq2 = unify_chunks(*uc_args)
    assert len(set(a.chunks for a in seq2)) == 1  # same chunks
    chunks = seq2[0].chunks[:axis] + ((1,) * n,) + seq2[0].chunks[axis:]
    names = [a.name for a in seq2]
    name = "stack-" + tokenize(names, axis)
    keys = list(product([name], *[range(len(bd)) for bd in chunks]))
    inputs = [
        (names[key[axis + 1]],) + key[1 : axis + 1] + key[axis + 2 :] for key in keys
    ]
    values = [
        (
            getitem,
            inp,
            (slice(None, None, None),) * axis
            + (None,)
            + (slice(None, None, None),) * (ndim - axis),
        )
        for inp in inputs
    ]
    layer = dict(zip(keys, values))
    graph = HighLevelGraph.from_collections(name, layer, dependencies=seq2)
    return Array(graph, name, chunks, meta=meta)
def concatenate3(arrays):
    
    from .utils import IS_NEP18_ACTIVE
    # We need this as __array_function__ may not exist on older NumPy versions.
    # And to reduce verbosity.
    NDARRAY_ARRAY_FUNCTION = getattr(np.ndarray, "__array_function__", None)
    arrays = concrete(arrays)
    if not arrays:
        return np.empty(0)
    advanced = max(
        core.flatten(arrays, container=(list, tuple)),
        key=lambda x: getattr(x, "__array_priority__", 0),
    )
    if IS_NEP18_ACTIVE and not all(
        NDARRAY_ARRAY_FUNCTION
        is getattr(arr, "__array_function__", NDARRAY_ARRAY_FUNCTION)
        for arr in arrays
    ):
        try:
            x = unpack_singleton(arrays)
            return _concatenate2(arrays, axes=tuple(range(x.ndim)))
        except TypeError:
            pass
    if concatenate_lookup.dispatch(type(advanced)) is not np.concatenate:
        x = unpack_singleton(arrays)
        return _concatenate2(arrays, axes=list(range(x.ndim)))
    ndim = ndimlist(arrays)
    if not ndim:
        return arrays
    chunks = chunks_from_arrays(arrays)
    shape = tuple(map(sum, chunks))
    def dtype(x):
        try:
            return x.dtype
        except AttributeError:
            return type(x)
    result = np.empty(shape=shape, dtype=dtype(deepfirst(arrays)))
    for (idx, arr) in zip(slices_from_chunks(chunks), core.flatten(arrays)):
        if hasattr(arr, "ndim"):
            while arr.ndim < ndim:
                arr = arr[None, ...]
        result[idx] = arr
    return result
def concatenate_axes(arrays, axes):
    
    if len(axes) != ndimlist(arrays):
        raise ValueError("Length of axes should equal depth of nested arrays")
    extradims = max(0, deepfirst(arrays).ndim - (max(axes) + 1))
    return concatenate3(transposelist(arrays, axes, extradims=extradims))
def to_hdf5(filename, *args, **kwargs):
    
    if len(args) == 1 and isinstance(args[0], dict):
        data = args[0]
    elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], Array):
        data = {args[0]: args[1]}
    else:
        raise ValueError("Please provide {'/data/path': array} dictionary")
    chunks = kwargs.pop("chunks", True)
    import h5py
    with h5py.File(filename, mode="a") as f:
        dsets = [
            f.require_dataset(
                dp,
                shape=x.shape,
                dtype=x.dtype,
                chunks=tuple([c[0] for c in x.chunks]) if chunks is True else chunks,
                **kwargs
            )
            for dp, x in data.items()
        ]
        store(list(data.values()), dsets)
def interleave_none(a, b):
    
    result = []
    i = j = 0
    n = len(a) + len(b)
    while i + j < n:
        if a[i] is not None:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            i += 1
            j += 1
    return tuple(result)
def keyname(name, i, okey):
    
    return (name, i) + tuple(k for k in okey if k is not None)
def _vindex(x, *indexes):
    
    indexes = replace_ellipsis(x.ndim, indexes)
    nonfancy_indexes = []
    reduced_indexes = []
    for i, ind in enumerate(indexes):
        if isinstance(ind, Number):
            nonfancy_indexes.append(ind)
        elif isinstance(ind, slice):
            nonfancy_indexes.append(ind)
            reduced_indexes.append(slice(None))
        else:
            nonfancy_indexes.append(slice(None))
            reduced_indexes.append(ind)
    nonfancy_indexes = tuple(nonfancy_indexes)
    reduced_indexes = tuple(reduced_indexes)
    x = x[nonfancy_indexes]
    array_indexes = {}
    for i, (ind, size) in enumerate(zip(reduced_indexes, x.shape)):
        if not isinstance(ind, slice):
            ind = np.array(ind, copy=True)
            if ind.dtype.kind == "b":
                raise IndexError("vindex does not support indexing with boolean arrays")
            if ((ind >= size) | (ind < -size)).any():
                raise IndexError(
                    "vindex key has entries out of bounds for "
                    "indexing along axis %s of size %s: %r" % (i, size, ind)
                )
            ind %= size
            array_indexes[i] = ind
    if array_indexes:
        x = _vindex_array(x, array_indexes)
    return x
def _vindex_array(x, dict_indexes):
    
    try:
        broadcast_indexes = np.broadcast_arrays(*dict_indexes.values())
    except ValueError:
        # note: error message exactly matches numpy
        shapes_str = " ".join(str(a.shape) for a in dict_indexes.values())
        raise IndexError(
            "shape mismatch: indexing arrays could not be "
            "broadcast together with shapes " + shapes_str
        )
    broadcast_shape = broadcast_indexes[0].shape
    lookup = dict(zip(dict_indexes, broadcast_indexes))
    flat_indexes = [
        lookup[i].ravel().tolist() if i in lookup else None for i in range(x.ndim)
    ]
    flat_indexes.extend([None] * (x.ndim - len(flat_indexes)))
    flat_indexes = [
        list(index) if index is not None else index for index in flat_indexes
    ]
    bounds = [list(accumulate(add, (0,) + c)) for c in x.chunks]
    bounds2 = [b for i, b in zip(flat_indexes, bounds) if i is not None]
    axis = _get_axis(flat_indexes)
    token = tokenize(x, flat_indexes)
    out_name = "vindex-merge-" + token
    points = list()
    for i, idx in enumerate(zip(*[i for i in flat_indexes if i is not None])):
        block_idx = [
            np.searchsorted(b, ind, "right") - 1 for b, ind in zip(bounds2, idx)
        ]
        inblock_idx = [
            ind - bounds2[k][j] for k, (ind, j) in enumerate(zip(idx, block_idx))
        ]
        points.append((i, tuple(block_idx), tuple(inblock_idx)))
    chunks = [c for i, c in zip(flat_indexes, x.chunks) if i is None]
    chunks.insert(0, (len(points),) if points else (0,))
    chunks = tuple(chunks)
    if points:
        per_block = groupby(1, points)
        per_block = dict((k, v) for k, v in per_block.items() if v)
        other_blocks = list(
            product(
                *[
                    list(range(len(c))) if i is None else [None]
                    for i, c in zip(flat_indexes, x.chunks)
                ]
            )
        )
        full_slices = [slice(None, None) if i is None else None for i in flat_indexes]
        name = "vindex-slice-" + token
        dsk = dict(
            (
                keyname(name, i, okey),
                (
                    _vindex_transpose,
                    (
                        _vindex_slice,
                        (x.name,) + interleave_none(okey, key),
                        interleave_none(
                            full_slices, list(zip(*pluck(2, per_block[key])))
                        ),
                    ),
                    axis,
                ),
            )
            for i, key in enumerate(per_block)
            for okey in other_blocks
        )
        dsk.update(
            (
                keyname("vindex-merge-" + token, 0, okey),
                (
                    _vindex_merge,
                    [list(pluck(0, per_block[key])) for key in per_block],
                    [keyname(name, i, okey) for i in range(len(per_block))],
                ),
            )
            for okey in other_blocks
        )
        result_1d = Array(
            HighLevelGraph.from_collections(out_name, dsk, dependencies=[x]),
            out_name,
            chunks,
            x.dtype,
        )
        return result_1d.reshape(broadcast_shape + result_1d.shape[1:])
    # output has a zero dimension, just create a new zero-shape array with the
    # same dtype
    from .wrap import empty
    result_1d = empty(
        tuple(map(sum, chunks)), chunks=chunks, dtype=x.dtype, name=out_name
    )
    return result_1d.reshape(broadcast_shape + result_1d.shape[1:])
def _get_axis(indexes):
    
    ndim = len(indexes)
    indexes = [slice(None, None) if i is None else [0] for i in indexes]
    x = np.empty((2,) * ndim)
    x2 = x[tuple(indexes)]
    return x2.shape.index(1)
def _vindex_slice(block, points):
    
    points = [p if isinstance(p, slice) else list(p) for p in points]
    return block[tuple(points)]
def _vindex_transpose(block, axis):
    
    axes = [axis] + list(range(axis)) + list(range(axis + 1, block.ndim))
    return block.transpose(axes)
def _vindex_merge(locations, values):
    
    locations = list(map(list, locations))
    values = list(values)
    n = sum(map(len, locations))
    shape = list(values[0].shape)
    shape[0] = n
    shape = tuple(shape)
    dtype = values[0].dtype
    x = np.empty(shape, dtype=dtype)
    ind = [slice(None, None) for i in range(x.ndim)]
    for loc, val in zip(locations, values):
        ind[0] = loc
        x[tuple(ind)] = val
    return x
def to_npy_stack(dirname, x, axis=0):
    
    chunks = tuple((c if i == axis else (sum(c),)) for i, c in enumerate(x.chunks))
    xx = x.rechunk(chunks)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    meta = {"chunks": chunks, "dtype": x.dtype, "axis": axis}
    with open(os.path.join(dirname, "info"), "wb") as f:
        pickle.dump(meta, f)
    name = "to-npy-stack-" + str(uuid.uuid1())
    dsk = {
        (name, i): (np.save, os.path.join(dirname, "%d.npy" % i), key)
        for i, key in enumerate(core.flatten(xx.__dask_keys__()))
    }
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[xx])
    compute_as_if_collection(Array, graph, list(dsk))
def from_npy_stack(dirname, mmap_mode="r"):
    
    with open(os.path.join(dirname, "info"), "rb") as f:
        info = pickle.load(f)
    dtype = info["dtype"]
    chunks = info["chunks"]
    axis = info["axis"]
    name = "from-npy-stack-%s" % dirname
    keys = list(product([name], *[range(len(c)) for c in chunks]))
    values = [
        (np.load, os.path.join(dirname, "%d.npy" % i), mmap_mode)
        for i in range(len(chunks[axis]))
    ]
    dsk = dict(zip(keys, values))
    return Array(dsk, name, chunks, dtype)
from .utils import meta_from_array
