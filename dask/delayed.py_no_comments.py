import operator
import types
import uuid
import warnings
from collections.abc import Iterator
try:
    from cytoolz import curry, concat, unique, merge
except ImportError:
    from toolz import curry, concat, unique, merge
from . import config, threaded
from .base import is_dask_collection, dont_optimize, DaskMethodsMixin
from .base import tokenize as _tokenize
from .compatibility import is_dataclass, dataclass_fields
from .core import quote
from .context import globalmethod
from .optimization import cull
from .utils import funcname, methodcaller, OperatorMethodMixin, ensure_dict, apply
from .highlevelgraph import HighLevelGraph
__all__ = ["Delayed", "delayed"]
def unzip(ls, nout):
    
    out = list(zip(*ls))
    if not out:
        out = [()] * nout
    return out
def finalize(collection):
    assert is_dask_collection(collection)
    name = "finalize-" + tokenize(collection)
    keys = collection.__dask_keys__()
    finalize, args = collection.__dask_postcompute__()
    layer = {name: (finalize, keys) + args}
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[collection])
    return Delayed(name, graph)
def unpack_collections(expr):
    
    if isinstance(expr, Delayed):
        return expr._key, (expr,)
    if is_dask_collection(expr):
        finalized = finalize(expr)
        return finalized._key, (finalized,)
    if isinstance(expr, Iterator):
        expr = tuple(expr)
    typ = type(expr)
    if typ in (list, tuple, set):
        args, collections = unzip((unpack_collections(e) for e in expr), 2)
        args = list(args)
        collections = tuple(unique(concat(collections), key=id))
        # Ensure output type matches input type
        if typ is not list:
            args = (typ, args)
        return args, collections
    if typ is dict:
        args, collections = unpack_collections([[k, v] for k, v in expr.items()])
        return (dict, args), collections
    if typ is slice:
        args, collections = unpack_collections([expr.start, expr.stop, expr.step])
        return (slice,) + tuple(args), collections
    if is_dataclass(expr):
        args, collections = unpack_collections(
            [[f.name, getattr(expr, f.name)] for f in dataclass_fields(expr)]
        )
        return (apply, typ, (), (dict, args)), collections
    return expr, ()
def to_task_dask(expr):
    
    warnings.warn(
        "The dask.delayed.to_dask_dask function has been "
        "Deprecated in favor of unpack_collections",
        stacklevel=2,
    )
    if isinstance(expr, Delayed):
        return expr.key, expr.dask
    if is_dask_collection(expr):
        name = "finalize-" + tokenize(expr, pure=True)
        keys = expr.__dask_keys__()
        opt = getattr(expr, "__dask_optimize__", dont_optimize)
        finalize, args = expr.__dask_postcompute__()
        dsk = {name: (finalize, keys) + args}
        dsk.update(opt(expr.__dask_graph__(), keys))
        return name, dsk
    if isinstance(expr, Iterator):
        expr = list(expr)
    typ = type(expr)
    if typ in (list, tuple, set):
        args, dasks = unzip((to_task_dask(e) for e in expr), 2)
        args = list(args)
        dsk = merge(dasks)
        # Ensure output type matches input type
        return (args, dsk) if typ is list else ((typ, args), dsk)
    if typ is dict:
        args, dsk = to_task_dask([[k, v] for k, v in expr.items()])
        return (dict, args), dsk
    if is_dataclass(expr):
        args, dsk = to_task_dask(
            [[f.name, getattr(expr, f.name)] for f in dataclass_fields(expr)]
        )
        return (apply, typ, (), (dict, args)), dsk
    if typ is slice:
        args, dsk = to_task_dask([expr.start, expr.stop, expr.step])
        return (slice,) + tuple(args), dsk
    return expr, {}
def tokenize(*args, **kwargs):
    
    pure = kwargs.pop("pure", None)
    if pure is None:
        pure = config.get("delayed_pure", False)
    if pure:
        return _tokenize(*args, **kwargs)
    else:
        return str(uuid.uuid4())
@curry
def delayed(obj, name=None, pure=None, nout=None, traverse=True):
    
    if isinstance(obj, Delayed):
        return obj
    if is_dask_collection(obj) or traverse:
        task, collections = unpack_collections(obj)
    else:
        task = quote(obj)
        collections = set()
    if task is obj:
        if not (nout is None or (type(nout) is int and nout >= 0)):
            raise ValueError(
                "nout must be None or a non-negative integer, got %s" % nout
            )
        if not name:
            try:
                prefix = obj.__name__
            except AttributeError:
                prefix = type(obj).__name__
            token = tokenize(obj, nout, pure=pure)
            name = "%s-%s" % (prefix, token)
        return DelayedLeaf(obj, name, pure=pure, nout=nout)
    else:
        if not name:
            name = "%s-%s" % (type(obj).__name__, tokenize(task, pure=pure))
        layer = {name: task}
        graph = HighLevelGraph.from_collections(name, layer, dependencies=collections)
        return Delayed(name, graph)
def right(method):
    
    def _inner(self, other):
        return method(other, self)
    return _inner
def optimize(dsk, keys, **kwargs):
    dsk = ensure_dict(dsk)
    dsk2, _ = cull(dsk, keys)
    return dsk2
def rebuild(dsk, key, length):
    return Delayed(key, dsk, length)
class Delayed(DaskMethodsMixin, OperatorMethodMixin):
    
    __slots__ = ("_key", "dask", "_length")
    def __init__(self, key, dsk, length=None):
        self._key = key
        self.dask = dsk
        self._length = length
    def __dask_graph__(self):
        return self.dask
    def __dask_keys__(self):
        return [self.key]
    def __dask_layers__(self):
        return (self.key,)
    def __dask_tokenize__(self):
        return self.key
    __dask_scheduler__ = staticmethod(threaded.get)
    __dask_optimize__ = globalmethod(optimize, key="delayed_optimize")
    def __dask_postcompute__(self):
        return single_key, ()
    def __dask_postpersist__(self):
        return rebuild, (self._key, getattr(self, "_length", None))
    def __getstate__(self):
        return tuple(getattr(self, i) for i in self.__slots__)
    def __setstate__(self, state):
        for k, v in zip(self.__slots__, state):
            setattr(self, k, v)
    @property
    def key(self):
        return self._key
    def __repr__(self):
        return "Delayed({0})".format(repr(self.key))
    def __hash__(self):
        return hash(self.key)
    def __dir__(self):
        return dir(type(self))
    def __getattr__(self, attr):
        if attr.startswith("_"):
            raise AttributeError("Attribute {0} not found".format(attr))
        return DelayedAttr(self, attr)
    def __setattr__(self, attr, val):
        if attr in self.__slots__:
            object.__setattr__(self, attr, val)
        else:
            raise TypeError("Delayed objects are immutable")
    def __setitem__(self, index, val):
        raise TypeError("Delayed objects are immutable")
    def __iter__(self):
        if getattr(self, "_length", None) is None:
            raise TypeError("Delayed objects of unspecified length are not iterable")
        for i in range(self._length):
            yield self[i]
    def __len__(self):
        if getattr(self, "_length", None) is None:
            raise TypeError("Delayed objects of unspecified length have no len()")
        return self._length
    def __call__(self, *args, **kwargs):
        pure = kwargs.pop("pure", None)
        name = kwargs.pop("dask_key_name", None)
        func = delayed(apply, pure=pure)
        if name is not None:
            return func(self, args, kwargs, dask_key_name=name)
        return func(self, args, kwargs)
    def __bool__(self):
        raise TypeError("Truth of Delayed objects is not supported")
    __nonzero__ = __bool__
    def __get__(self, instance, cls):
        if instance is None:
            return self
        return types.MethodType(self, instance)
    @classmethod
    def _get_binary_operator(cls, op, inv=False):
        method = delayed(right(op) if inv else op, pure=True)
        return lambda *args, **kwargs: method(*args, **kwargs)
    _get_unary_operator = _get_binary_operator
def call_function(func, func_token, args, kwargs, pure=None, nout=None):
    dask_key_name = kwargs.pop("dask_key_name", None)
    pure = kwargs.pop("pure", pure)
    if dask_key_name is None:
        name = "%s-%s" % (
            funcname(func),
            tokenize(func_token, *args, pure=pure, **kwargs),
        )
    else:
        name = dask_key_name
    args2, collections = unzip(map(unpack_collections, args), 2)
    collections = list(concat(collections))
    if kwargs:
        dask_kwargs, collections2 = unpack_collections(kwargs)
        collections.extend(collections2)
        task = (apply, func, list(args2), dask_kwargs)
    else:
        task = (func,) + args2
    graph = HighLevelGraph.from_collections(
        name, {name: task}, dependencies=collections
    )
    nout = nout if nout is not None else None
    return Delayed(name, graph, length=nout)
class DelayedLeaf(Delayed):
    __slots__ = ("_obj", "_key", "_pure", "_nout")
    def __init__(self, obj, key, pure=None, nout=None):
        self._obj = obj
        self._key = key
        self._pure = pure
        self._nout = nout
    @property
    def dask(self):
        return HighLevelGraph.from_collections(
            self._key, {self._key: self._obj}, dependencies=()
        )
    def __call__(self, *args, **kwargs):
        return call_function(
            self._obj, self._key, args, kwargs, pure=self._pure, nout=self._nout
        )
class DelayedAttr(Delayed):
    __slots__ = ("_obj", "_attr", "_key")
    def __init__(self, obj, attr):
        self._obj = obj
        self._attr = attr
        self._key = "getattr-%s" % tokenize(obj, attr, pure=True)
    def __getattr__(self, attr):
        # Calling np.dtype(dask.delayed(...)) used to result in a segfault, as
        # numpy recursively tries to get `dtype` from the object. This is
        # likely a bug in numpy. For now, we can do a dumb for if
        # `x.dtype().dtype()` is called (which shouldn't ever show up in real
        # code). See https://github.com/dask/dask/pull/4374#issuecomment-454381465
        if attr == "dtype" and self._attr == "dtype":
            raise AttributeError("Attribute %s not found" % attr)
        return super(DelayedAttr, self).__getattr__(attr)
    @property
    def dask(self):
        layer = {self._key: (getattr, self._obj._key, self._attr)}
        return HighLevelGraph.from_collections(
            self._key, layer, dependencies=[self._obj]
        )
    def __call__(self, *args, **kwargs):
        return call_function(
            methodcaller(self._attr), self._attr, (self._obj,) + args, kwargs
        )
for op in [
    operator.abs,
    operator.neg,
    operator.pos,
    operator.invert,
    operator.add,
    operator.sub,
    operator.mul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.pow,
    operator.and_,
    operator.or_,
    operator.xor,
    operator.lshift,
    operator.rshift,
    operator.eq,
    operator.ge,
    operator.gt,
    operator.ne,
    operator.le,
    operator.lt,
    operator.getitem,
]:
    Delayed._bind_operator(op)
try:
    Delayed._bind_operator(operator.matmul)
except AttributeError:
    pass
def single_key(seq):
    
    return seq[0]
