from collections import OrderedDict
from collections.abc import Mapping, Iterator
from functools import partial
from hashlib import md5
from operator import getitem
import inspect
import pickle
import os
import threading
import uuid
from toolz import merge, groupby, curry, identity
from toolz.functoolz import Compose
from .compatibility import is_dataclass, dataclass_fields
from .context import thread_state
from .core import flatten, quote, get as simple_get
from .hashing import hash_buffer_hex
from .utils import Dispatch, ensure_dict, apply
from . import config, local, threaded
__all__ = (
    "DaskMethodsMixin",
    "is_dask_collection",
    "compute",
    "persist",
    "optimize",
    "visualize",
    "tokenize",
    "normalize_token",
)
def is_dask_collection(x):
    
    try:
        return x.__dask_graph__() is not None
    except (AttributeError, TypeError):
        return False
class DaskMethodsMixin(object):
    
    __slots__ = ()
    def visualize(self, filename="mydask", format=None, optimize_graph=False, **kwargs):
        
        return visualize(
            self,
            filename=filename,
            format=format,
            optimize_graph=optimize_graph,
            **kwargs
        )
    def persist(self, **kwargs):
        
        (result,) = persist(self, traverse=False, **kwargs)
        return result
    def compute(self, **kwargs):
        
        (result,) = compute(self, traverse=False, **kwargs)
        return result
    def __await__(self):
        try:
            from distributed import wait, futures_of
        except ImportError:
            raise ImportError(
                "Using async/await with dask requires the `distributed` package"
            )
        from tornado import gen
        @gen.coroutine
        def f():
            if futures_of(self):
                yield wait(self)
            raise gen.Return(self)
        return f().__await__()
def compute_as_if_collection(cls, dsk, keys, scheduler=None, get=None, **kwargs):
    
    schedule = get_scheduler(scheduler=scheduler, cls=cls, get=get)
    dsk2 = optimization_function(cls)(ensure_dict(dsk), keys, **kwargs)
    return schedule(dsk2, keys, **kwargs)
def dont_optimize(dsk, keys, **kwargs):
    return dsk
def optimization_function(x):
    return getattr(x, "__dask_optimize__", dont_optimize)
def collections_to_dsk(collections, optimize_graph=True, **kwargs):
    
    optimizations = kwargs.pop("optimizations", None) or config.get("optimizations", [])
    if optimize_graph:
        groups = groupby(optimization_function, collections)
        groups = {opt: _extract_graph_and_keys(val) for opt, val in groups.items()}
        for opt in optimizations:
            groups = {k: (opt(dsk, keys), keys) for k, (dsk, keys) in groups.items()}
        dsk = merge(
            *map(
                ensure_dict,
                [opt(dsk, keys, **kwargs) for opt, (dsk, keys) in groups.items()],
            )
        )
    else:
        dsk, _ = _extract_graph_and_keys(collections)
    return dsk
def _extract_graph_and_keys(vals):
    
    from .highlevelgraph import HighLevelGraph
    graphs = [v.__dask_graph__() for v in vals]
    keys = [v.__dask_keys__() for v in vals]
    if any(isinstance(graph, HighLevelGraph) for graph in graphs):
        graph = HighLevelGraph.merge(*graphs)
    else:
        graph = merge(*map(ensure_dict, graphs))
    return graph, keys
def unpack_collections(*args, **kwargs):
    
    traverse = kwargs.pop("traverse", True)
    collections = []
    repack_dsk = {}
    collections_token = uuid.uuid4().hex
    def _unpack(expr):
        if is_dask_collection(expr):
            tok = tokenize(expr)
            if tok not in repack_dsk:
                repack_dsk[tok] = (getitem, collections_token, len(collections))
                collections.append(expr)
            return tok
        tok = uuid.uuid4().hex
        if not traverse:
            tsk = quote(expr)
        else:
            # Treat iterators like lists
            typ = list if isinstance(expr, Iterator) else type(expr)
            if typ in (list, tuple, set):
                tsk = (typ, [_unpack(i) for i in expr])
            elif typ in (dict, OrderedDict):
                tsk = (typ, [[_unpack(k), _unpack(v)] for k, v in expr.items()])
            elif is_dataclass(expr):
                tsk = (
                    apply,
                    typ,
                    (),
                    (
                        dict,
                        [
                            [f.name, _unpack(getattr(expr, f.name))]
                            for f in dataclass_fields(expr)
                        ],
                    ),
                )
            else:
                return expr
        repack_dsk[tok] = tsk
        return tok
    out = uuid.uuid4().hex
    repack_dsk[out] = (tuple, [_unpack(i) for i in args])
    def repack(results):
        dsk = repack_dsk.copy()
        dsk[collections_token] = quote(results)
        return simple_get(dsk, out)
    return collections, repack
def optimize(*args, **kwargs):
    
    collections, repack = unpack_collections(*args, **kwargs)
    if not collections:
        return args
    dsk = collections_to_dsk(collections, **kwargs)
    postpersists = [
        a.__dask_postpersist__() if is_dask_collection(a) else (None, a) for a in args
    ]
    keys, postpersists = [], []
    for a in collections:
        keys.extend(flatten(a.__dask_keys__()))
        postpersists.append(a.__dask_postpersist__())
    return repack([r(dsk, *s) for r, s in postpersists])
def compute(*args, **kwargs):
    
    traverse = kwargs.pop("traverse", True)
    optimize_graph = kwargs.pop("optimize_graph", True)
    collections, repack = unpack_collections(*args, traverse=traverse)
    if not collections:
        return args
    schedule = get_scheduler(
        scheduler=kwargs.pop("scheduler", None),
        collections=collections,
        get=kwargs.pop("get", None),
    )
    dsk = collections_to_dsk(collections, optimize_graph, **kwargs)
    keys = [x.__dask_keys__() for x in collections]
    postcomputes = [x.__dask_postcompute__() for x in collections]
    results = schedule(dsk, keys, **kwargs)
    return repack([f(r, *a) for r, (f, a) in zip(results, postcomputes)])
def visualize(*args, **kwargs):
    
    from dask.dot import dot_graph
    filename = kwargs.pop("filename", "mydask")
    optimize_graph = kwargs.pop("optimize_graph", False)
    args2 = []
    for arg in args:
        if isinstance(arg, (list, tuple, set)):
            args2.extend(arg)
        else:
            args2.append(arg)
    dsks = [arg for arg in args2 if isinstance(arg, Mapping)]
    args3 = [arg for arg in args2 if is_dask_collection(arg)]
    dsk = dict(collections_to_dsk(args3, optimize_graph=optimize_graph))
    for d in dsks:
        dsk.update(d)
    color = kwargs.get("color")
    if color == "order":
        from .order import order
        import matplotlib.pyplot as plt
        o = order(dsk)
        try:
            cmap = kwargs.pop("cmap")
        except KeyError:
            cmap = plt.cm.RdBu
        if isinstance(cmap, str):
            import matplotlib.pyplot as plt
            cmap = getattr(plt.cm, cmap)
        mx = max(o.values()) + 1
        colors = {k: _colorize(cmap(v / mx, bytes=True)) for k, v in o.items()}
        kwargs["function_attributes"] = {
            k: {"color": v, "label": str(o[k])} for k, v in colors.items()
        }
        kwargs["data_attributes"] = {k: {"color": v} for k, v in colors.items()}
    elif color:
        raise NotImplementedError("Unknown value color=%s" % color)
    return dot_graph(dsk, filename=filename, **kwargs)
def persist(*args, **kwargs):
    
    traverse = kwargs.pop("traverse", True)
    optimize_graph = kwargs.pop("optimize_graph", True)
    collections, repack = unpack_collections(*args, traverse=traverse)
    if not collections:
        return args
    schedule = get_scheduler(
        scheduler=kwargs.pop("scheduler", None), collections=collections
    )
    if inspect.ismethod(schedule):
        try:
            from distributed.client import default_client
        except ImportError:
            pass
        else:
            try:
                client = default_client()
            except ValueError:
                pass
            else:
                if client.get == schedule:
                    results = client.persist(
                        collections, optimize_graph=optimize_graph, **kwargs
                    )
                    return repack(results)
    dsk = collections_to_dsk(collections, optimize_graph, **kwargs)
    keys, postpersists = [], []
    for a in collections:
        a_keys = list(flatten(a.__dask_keys__()))
        rebuild, state = a.__dask_postpersist__()
        keys.extend(a_keys)
        postpersists.append((rebuild, a_keys, state))
    results = schedule(dsk, keys, **kwargs)
    d = dict(zip(keys, results))
    results2 = [r({k: d[k] for k in ks}, *s) for r, ks, s in postpersists]
    return repack(results2)
############
# Tokenize #
############
def tokenize(*args, **kwargs):
    
    if kwargs:
        args = args + (kwargs,)
    return md5(str(tuple(map(normalize_token, args))).encode()).hexdigest()
normalize_token = Dispatch()
normalize_token.register(
    (int, float, str, bytes, type(None), type, slice, complex, type(Ellipsis)), identity
)
@normalize_token.register(dict)
def normalize_dict(d):
    return normalize_token(sorted(d.items(), key=str))
@normalize_token.register(OrderedDict)
def normalize_ordered_dict(d):
    return type(d).__name__, normalize_token(list(d.items()))
@normalize_token.register(set)
def normalize_set(s):
    return normalize_token(sorted(s, key=str))
@normalize_token.register((tuple, list))
def normalize_seq(seq):
    return type(seq).__name__, list(map(normalize_token, seq))
@normalize_token.register(object)
def normalize_object(o):
    method = getattr(o, "__dask_tokenize__", None)
    if method is not None:
        return method()
    return normalize_function(o) if callable(o) else uuid.uuid4().hex
function_cache = {}
function_cache_lock = threading.Lock()
def normalize_function(func):
    try:
        return function_cache[func]
    except KeyError:
        result = _normalize_function(func)
        if len(function_cache) >= 500:  # clear half of cache if full
            with function_cache_lock:
                if len(function_cache) >= 500:
                    for k in list(function_cache)[::2]:
                        del function_cache[k]
        function_cache[func] = result
        return result
    except TypeError:  # not hashable
        return _normalize_function(func)
def _normalize_function(func):
    if isinstance(func, curry):
        func = func._partial
    if isinstance(func, Compose):
        first = getattr(func, "first", None)
        funcs = reversed((first,) + func.funcs) if first else func.funcs
        return tuple(normalize_function(f) for f in funcs)
    elif isinstance(func, partial):
        args = tuple(normalize_token(i) for i in func.args)
        if func.keywords:
            kws = tuple(
                (k, normalize_token(v)) for k, v in sorted(func.keywords.items())
            )
        else:
            kws = None
        return (normalize_function(func.func), args, kws)
    else:
        try:
            result = pickle.dumps(func, protocol=0)
            if b"__main__" not in result:  # abort on dynamic functions
                return result
        except Exception:
            pass
        try:
            import cloudpickle
            return cloudpickle.dumps(func, protocol=0)
        except Exception:
            return str(func)
@normalize_token.register_lazy("pandas")
def register_pandas():
    import pandas as pd
    @normalize_token.register(pd.Index)
    def normalize_index(ind):
        return [ind.name, normalize_token(ind.values)]
    @normalize_token.register(pd.Categorical)
    def normalize_categorical(cat):
        return [
            normalize_token(cat.codes),
            normalize_token(cat.categories),
            cat.ordered,
        ]
    @normalize_token.register(pd.Series)
    def normalize_series(s):
        return [
            s.name,
            s.dtype,
            normalize_token(s._data.blocks[0].values),
            normalize_token(s.index),
        ]
    @normalize_token.register(pd.DataFrame)
    def normalize_dataframe(df):
        data = [block.values for block in df._data.blocks]
        data += [df.columns, df.index]
        return list(map(normalize_token, data))
@normalize_token.register_lazy("numpy")
def register_numpy():
    import numpy as np
    @normalize_token.register(np.ndarray)
    def normalize_array(x):
        if not x.shape:
            return (x.item(), x.dtype)
        if hasattr(x, "mode") and getattr(x, "filename", None):
            if hasattr(x.base, "ctypes"):
                offset = (
                    x.ctypes.get_as_parameter().value
                    - x.base.ctypes.get_as_parameter().value
                )
            else:
                offset = 0  # root memmap's have mmap object as base
            if hasattr(
                x, "offset"
            ):  # offset numpy used while opening, and not the offset to the beginning of the file
                offset += getattr(x, "offset")
            return (
                x.filename,
                os.path.getmtime(x.filename),
                x.dtype,
                x.shape,
                x.strides,
                offset,
            )
        if x.dtype.hasobject:
            try:
                try:
                    # string fast-path
                    data = hash_buffer_hex(
                        "-".join(x.flat).encode(
                            encoding="utf-8", errors="surrogatepass"
                        )
                    )
                except UnicodeDecodeError:
                    # bytes fast-path
                    data = hash_buffer_hex(b"-".join(x.flat))
            except (TypeError, UnicodeDecodeError):
                try:
                    data = hash_buffer_hex(pickle.dumps(x, pickle.HIGHEST_PROTOCOL))
                except Exception:
                    # pickling not supported, use UUID4-based fallback
                    data = uuid.uuid4().hex
        else:
            try:
                data = hash_buffer_hex(x.ravel(order="K").view("i1"))
            except (BufferError, AttributeError, ValueError):
                data = hash_buffer_hex(x.copy().ravel(order="K").view("i1"))
        return (data, x.dtype, x.shape, x.strides)
    @normalize_token.register(np.matrix)
    def normalize_matrix(x):
        return type(x).__name__, normalize_array(x.view(type=np.ndarray))
    normalize_token.register(np.dtype, repr)
    normalize_token.register(np.generic, repr)
    @normalize_token.register(np.ufunc)
    def normalize_ufunc(x):
        try:
            name = x.__name__
            if getattr(np, name) is x:
                return "np." + name
        except AttributeError:
            return normalize_function(x)
@normalize_token.register_lazy("scipy")
def register_scipy():
    import scipy.sparse as sp
    def normalize_sparse_matrix(x, attrs):
        return (
            type(x).__name__,
            normalize_seq((normalize_token(getattr(x, key)) for key in attrs)),
        )
    for cls, attrs in [
        (sp.dia_matrix, ("data", "offsets", "shape")),
        (sp.bsr_matrix, ("data", "indices", "indptr", "blocksize", "shape")),
        (sp.coo_matrix, ("data", "row", "col", "shape")),
        (sp.csr_matrix, ("data", "indices", "indptr", "shape")),
        (sp.csc_matrix, ("data", "indices", "indptr", "shape")),
        (sp.lil_matrix, ("data", "rows", "shape")),
    ]:
        normalize_token.register(cls, partial(normalize_sparse_matrix, attrs=attrs))
    @normalize_token.register(sp.dok_matrix)
    def normalize_dok_matrix(x):
        return type(x).__name__, normalize_token(sorted(x.items()))
def _colorize(t):
    
    t = t[:3]
    i = sum(v * 256 ** (len(t) - i - 1) for i, v in enumerate(t))
    h = hex(int(i))[2:].upper()
    h = "0" * (6 - len(h)) + h
    return "#" + h
named_schedulers = {
    "sync": local.get_sync,
    "synchronous": local.get_sync,
    "single-threaded": local.get_sync,
    "threads": threaded.get,
    "threading": threaded.get,
}
try:
    from dask import multiprocessing as dask_multiprocessing
except ImportError:
    pass
else:
    named_schedulers.update(
        {
            "processes": dask_multiprocessing.get,
            "multiprocessing": dask_multiprocessing.get,
        }
    )
get_err_msg = 
.strip()
def get_scheduler(get=None, scheduler=None, collections=None, cls=None):
    
    if get:
        raise TypeError(get_err_msg)
    if scheduler is not None:
        if callable(scheduler):
            return scheduler
        elif "Client" in type(scheduler).__name__ and hasattr(scheduler, "get"):
            return scheduler.get
        elif scheduler.lower() in named_schedulers:
            return named_schedulers[scheduler.lower()]
        elif scheduler.lower() in ("dask.distributed", "distributed"):
            from distributed.worker import get_client
            return get_client().get
        else:
            raise ValueError(
                "Expected one of [distributed, %s]"
                % ", ".join(sorted(named_schedulers))
            )
        # else:  # try to connect to remote scheduler with this name
        #     return get_client(scheduler).get
    if config.get("scheduler", None):
        return get_scheduler(scheduler=config.get("scheduler", None))
    if config.get("get", None):
        raise ValueError(get_err_msg)
    if getattr(thread_state, "key", False):
        from distributed.worker import get_worker
        return get_worker().client.get
    if cls is not None:
        return cls.__dask_scheduler__
    if collections:
        collections = [c for c in collections if c is not None]
    if collections:
        get = collections[0].__dask_scheduler__
        if not all(c.__dask_scheduler__ == get for c in collections):
            raise ValueError(
                "Compute called on multiple collections with "
                "differing default schedulers. Please specify a "
                "scheduler=` parameter explicitly in compute or "
                "globally with `dask.config.set`."
            )
        return get
    return None
def wait(x, timeout=None, return_when="ALL_COMPLETED"):
    
    try:
        from distributed import wait
        return wait(x, timeout=timeout, return_when=return_when)
    except (ImportError, ValueError):
        return x
