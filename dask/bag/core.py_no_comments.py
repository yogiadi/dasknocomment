import io
import itertools
import math
import operator
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator
from distutils.version import LooseVersion
from functools import wraps, partial
from random import Random
from urllib.request import urlopen
import toolz
from toolz import (
    merge,
    take,
    reduce,
    valmap,
    map,
    partition_all,
    filter,
    remove,
    compose,
    curry,
    first,
    second,
    accumulate,
    peek,
)
_implement_accumulate = LooseVersion(toolz.__version__) > "0.7.4"
try:
    import cytoolz
    from cytoolz import (
        frequencies,
        merge_with,
        join,
        reduceby,
        count,
        pluck,
        groupby,
        topk,
        unique,
    )
    if LooseVersion(cytoolz.__version__) > "0.7.3":
        from cytoolz import accumulate  # noqa: F811
        _implement_accumulate = True
except ImportError:
    from toolz import (
        frequencies,
        merge_with,
        join,
        reduceby,
        count,
        pluck,
        groupby,
        topk,
        unique,
    )
from .. import config
from .avro import to_avro
from ..base import tokenize, dont_optimize, DaskMethodsMixin
from ..bytes import open_files
from ..context import globalmethod
from ..core import quote, istask, get_dependencies, reverse_dict, flatten
from ..delayed import Delayed, unpack_collections
from ..highlevelgraph import HighLevelGraph
from ..multiprocessing import get as mpget
from ..optimization import fuse, cull, inline
from ..utils import (
    apply,
    system_encoding,
    takes_multiple_arguments,
    funcname,
    digit,
    insert,
    ensure_dict,
    ensure_bytes,
    ensure_unicode,
    key_split,
)
from . import chunk
no_default = "__no__default__"
no_result = type(
    "no_result", (object,), {"__slots__": (), "__reduce__": lambda self: "no_result"}
)
def lazify_task(task, start=True):
    
    if type(task) is list and len(task) < 50:
        return [lazify_task(arg, False) for arg in task]
    if not istask(task):
        return task
    head, tail = task[0], task[1:]
    if not start and head in (list, reify):
        task = task[1]
        return lazify_task(*tail, start=False)
    else:
        return (head,) + tuple([lazify_task(arg, False) for arg in tail])
def lazify(dsk):
    
    return valmap(lazify_task, dsk)
def inline_singleton_lists(dsk, keys, dependencies=None):
    
    if dependencies is None:
        dependencies = {k: get_dependencies(dsk, task=v) for k, v in dsk.items()}
    dependents = reverse_dict(dependencies)
    inline_keys = {
        k
        for k, v in dsk.items()
        if istask(v) and v and v[0] is list and len(dependents[k]) == 1
    }
    inline_keys.difference_update(flatten(keys))
    dsk = inline(dsk, inline_keys, inline_constants=False)
    for k in inline_keys:
        del dsk[k]
    return dsk
def optimize(dsk, keys, fuse_keys=None, rename_fused_keys=True, **kwargs):
    
    dsk = ensure_dict(dsk)
    dsk2, dependencies = cull(dsk, keys)
    dsk3, dependencies = fuse(
        dsk2, keys + (fuse_keys or []), dependencies, rename_keys=rename_fused_keys
    )
    dsk4 = inline_singleton_lists(dsk3, keys, dependencies)
    dsk5 = lazify(dsk4)
    return dsk5
def _to_textfiles_chunk(data, lazy_file, last_endline):
    with lazy_file as f:
        if isinstance(f, io.TextIOWrapper):
            endline = u"\n"
            ensure = ensure_unicode
        else:
            endline = b"\n"
            ensure = ensure_bytes
        started = False
        for d in data:
            if started:
                f.write(endline)
            else:
                started = True
            f.write(ensure(d))
        if last_endline:
            f.write(endline)
def to_textfiles(
    b,
    path,
    name_function=None,
    compression="infer",
    encoding=system_encoding,
    compute=True,
    storage_options=None,
    last_endline=False,
    **kwargs
):
    
    mode = "wb" if encoding is None else "wt"
    files = open_files(
        path,
        compression=compression,
        mode=mode,
        encoding=encoding,
        name_function=name_function,
        num=b.npartitions,
        **(storage_options or {})
    )
    name = "to-textfiles-" + uuid.uuid4().hex
    dsk = {
        (name, i): (_to_textfiles_chunk, (b.name, i), f, last_endline)
        for i, f in enumerate(files)
    }
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b])
    out = type(b)(graph, name, b.npartitions)
    if compute:
        out.compute(**kwargs)
        return [f.path for f in files]
    else:
        return out.to_delayed()
def finalize(results):
    if not results:
        return results
    if isinstance(results, Iterator):
        results = list(results)
    if isinstance(results[0], Iterable) and not isinstance(results[0], str):
        results = toolz.concat(results)
    if isinstance(results, Iterator):
        results = list(results)
    return results
def finalize_item(results):
    return results[0]
class StringAccessor(object):
    
    def __init__(self, bag):
        self._bag = bag
    def __dir__(self):
        return sorted(set(dir(type(self)) + dir(str)))
    def _strmap(self, key, *args, **kwargs):
        return self._bag.map(operator.methodcaller(key, *args, **kwargs))
    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if key in dir(str):
                func = getattr(str, key)
                return robust_wraps(func)(partial(self._strmap, key))
            else:
                raise
    def match(self, pattern):
        
        from fnmatch import fnmatch
        return self._bag.filter(partial(fnmatch, pat=pattern))
def robust_wraps(wrapper):
    
    def _(wrapped):
        wrapped.__doc__ = wrapper.__doc__
        return wrapped
    return _
class Item(DaskMethodsMixin):
    def __init__(self, dsk, key):
        self.dask = dsk
        self.key = key
        self.name = key
    def __dask_graph__(self):
        return self.dask
    def __dask_keys__(self):
        return [self.key]
    def __dask_tokenize__(self):
        return self.key
    __dask_optimize__ = globalmethod(optimize, key="bag_optimize", falsey=dont_optimize)
    __dask_scheduler__ = staticmethod(mpget)
    def __dask_postcompute__(self):
        return finalize_item, ()
    def __dask_postpersist__(self):
        return Item, (self.key,)
    @staticmethod
    def from_delayed(value):
        
        from dask.delayed import Delayed, delayed
        if not isinstance(value, Delayed) and hasattr(value, "key"):
            value = delayed(value)
        assert isinstance(value, Delayed)
        return Item(ensure_dict(value.dask), value.key)
    @property
    def _args(self):
        return (self.dask, self.key)
    def __getstate__(self):
        return self._args
    def __setstate__(self, state):
        self.dask, self.key = state
    def apply(self, func):
        name = "{0}-{1}".format(funcname(func), tokenize(self, func, "apply"))
        dsk = {name: (func, self.key)}
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return Item(graph, name)
    __int__ = __float__ = __complex__ = __bool__ = DaskMethodsMixin.compute
    def to_delayed(self, optimize_graph=True):
        
        from dask.delayed import Delayed
        dsk = self.__dask_graph__()
        if optimize_graph:
            dsk = self.__dask_optimize__(dsk, self.__dask_keys__())
        return Delayed(self.key, dsk)
class Bag(DaskMethodsMixin):
    
    def __init__(self, dsk, name, npartitions):
        if not isinstance(dsk, HighLevelGraph):
            dsk = HighLevelGraph.from_collections(name, dsk, dependencies=[])
        self.dask = dsk
        self.name = name
        self.npartitions = npartitions
    def __dask_graph__(self):
        return self.dask
    def __dask_keys__(self):
        return [(self.name, i) for i in range(self.npartitions)]
    def __dask_layers__(self):
        return (self.name,)
    def __dask_tokenize__(self):
        return self.name
    __dask_optimize__ = globalmethod(optimize, key="bag_optimize", falsey=dont_optimize)
    __dask_scheduler__ = staticmethod(mpget)
    def __dask_postcompute__(self):
        return finalize, ()
    def __dask_postpersist__(self):
        return type(self), (self.name, self.npartitions)
    def __str__(self):
        return "dask.bag<%s, npartitions=%d>" % (key_split(self.name), self.npartitions)
    __repr__ = __str__
    str = property(fget=StringAccessor)
    def map(self, func, *args, **kwargs):
        
        return bag_map(func, self, *args, **kwargs)
    def starmap(self, func, **kwargs):
        
        name = "{0}-{1}".format(
            funcname(func), tokenize(self, func, "starmap", **kwargs)
        )
        dependencies = [self]
        if kwargs:
            kwargs, collections = unpack_scalar_dask_kwargs(kwargs)
            dependencies.extend(collections)
        dsk = {
            (name, i): (reify, (starmap_chunk, func, (self.name, i), kwargs))
            for i in range(self.npartitions)
        }
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=dependencies)
        return type(self)(graph, name, self.npartitions)
    @property
    def _args(self):
        return (self.dask, self.name, self.npartitions)
    def __getstate__(self):
        return self._args
    def __setstate__(self, state):
        self.dask, self.name, self.npartitions = state
    def filter(self, predicate):
        
        name = "filter-{0}-{1}".format(funcname(predicate), tokenize(self, predicate))
        dsk = dict(
            ((name, i), (reify, (filter, predicate, (self.name, i))))
            for i in range(self.npartitions)
        )
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return type(self)(graph, name, self.npartitions)
    def random_sample(self, prob, random_state=None):
        
        if not 0 <= prob <= 1:
            raise ValueError("prob must be a number in the interval [0, 1]")
        if not isinstance(random_state, Random):
            random_state = Random(random_state)
        name = "random-sample-%s" % tokenize(self, prob, random_state.getstate())
        state_data = random_state_data_python(self.npartitions, random_state)
        dsk = {
            (name, i): (reify, (random_sample, (self.name, i), state, prob))
            for i, state in zip(range(self.npartitions), state_data)
        }
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return type(self)(graph, name, self.npartitions)
    def remove(self, predicate):
        
        name = "remove-{0}-{1}".format(funcname(predicate), tokenize(self, predicate))
        dsk = dict(
            ((name, i), (reify, (remove, predicate, (self.name, i))))
            for i in range(self.npartitions)
        )
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return type(self)(graph, name, self.npartitions)
    def map_partitions(self, func, *args, **kwargs):
        
        return map_partitions(func, self, *args, **kwargs)
    def pluck(self, key, default=no_default):
        
        name = "pluck-" + tokenize(self, key, default)
        key = quote(key)
        if default == no_default:
            dsk = dict(
                ((name, i), (list, (pluck, key, (self.name, i))))
                for i in range(self.npartitions)
            )
        else:
            dsk = dict(
                ((name, i), (list, (pluck, key, (self.name, i), default)))
                for i in range(self.npartitions)
            )
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return type(self)(graph, name, self.npartitions)
    def unzip(self, n):
        
        return tuple(self.pluck(i) for i in range(n))
    @wraps(to_textfiles)
    def to_textfiles(
        self,
        path,
        name_function=None,
        compression="infer",
        encoding=system_encoding,
        compute=True,
        storage_options=None,
        last_endline=False,
        **kwargs
    ):
        return to_textfiles(
            self,
            path,
            name_function,
            compression,
            encoding,
            compute,
            storage_options=storage_options,
            last_endline=last_endline,
            **kwargs
        )
    @wraps(to_avro)
    def to_avro(
        self,
        filename,
        schema,
        name_function=None,
        storage_options=None,
        codec="null",
        sync_interval=16000,
        metadata=None,
        compute=True,
        **kwargs
    ):
        return to_avro(
            self,
            filename,
            schema,
            name_function,
            storage_options,
            codec,
            sync_interval,
            metadata,
            compute,
            **kwargs
        )
    def fold(
        self, binop, combine=None, initial=no_default, split_every=None, out_type=Item
    ):
        
        combine = combine or binop
        if initial is not no_default:
            return self.reduction(
                curry(_reduce, binop, initial=initial),
                curry(_reduce, combine),
                split_every=split_every,
                out_type=out_type,
            )
        else:
            from toolz.curried import reduce
            return self.reduction(
                reduce(binop),
                reduce(combine),
                split_every=split_every,
                out_type=out_type,
            )
    def frequencies(self, split_every=None, sort=False):
        
        result = self.reduction(
            frequencies,
            merge_frequencies,
            out_type=Bag,
            split_every=split_every,
            name="frequencies",
        ).map_partitions(dictitems)
        if sort:
            result = result.map_partitions(sorted, key=second, reverse=True)
        return result
    def topk(self, k, key=None, split_every=None):
        
        if key:
            if callable(key) and takes_multiple_arguments(key):
                key = partial(apply, key)
            func = partial(topk, k, key=key)
        else:
            func = partial(topk, k)
        return self.reduction(
            func,
            compose(func, toolz.concat),
            out_type=Bag,
            split_every=split_every,
            name="topk",
        )
    def distinct(self, key=None):
        
        func = chunk_distinct if key is None else partial(chunk_distinct, key=key)
        agg = merge_distinct if key is None else partial(merge_distinct, key=key)
        return self.reduction(func, agg, out_type=Bag, name="distinct")
    def reduction(
        self, perpartition, aggregate, split_every=None, out_type=Item, name=None
    ):
        
        if split_every is None:
            split_every = 8
        if split_every is False:
            split_every = self.npartitions
        token = tokenize(self, perpartition, aggregate, split_every)
        a = "%s-part-%s" % (name or funcname(perpartition), token)
        is_last = self.npartitions == 1
        dsk = {
            (a, i): (empty_safe_apply, perpartition, (self.name, i), is_last)
            for i in range(self.npartitions)
        }
        k = self.npartitions
        b = a
        fmt = "%s-aggregate-%s" % (name or funcname(aggregate), token)
        depth = 0
        while k > split_every:
            c = fmt + str(depth)
            dsk2 = dict(
                (
                    (c, i),
                    (empty_safe_aggregate, aggregate, [(b, j) for j in inds], False),
                )
                for i, inds in enumerate(partition_all(split_every, range(k)))
            )
            dsk.update(dsk2)
            k = len(dsk2)
            b = c
            depth += 1
        dsk[(fmt, 0)] = (
            empty_safe_aggregate,
            aggregate,
            [(b, j) for j in range(k)],
            True,
        )
        graph = HighLevelGraph.from_collections(fmt, dsk, dependencies=[self])
        if out_type is Item:
            dsk[fmt] = dsk.pop((fmt, 0))
            return Item(graph, fmt)
        else:
            return Bag(graph, fmt, 1)
    def sum(self, split_every=None):
        
        return self.reduction(sum, sum, split_every=split_every)
    def max(self, split_every=None):
        
        return self.reduction(max, max, split_every=split_every)
    def min(self, split_every=None):
        
        return self.reduction(min, min, split_every=split_every)
    def any(self, split_every=None):
        
        return self.reduction(any, any, split_every=split_every)
    def all(self, split_every=None):
        
        return self.reduction(all, all, split_every=split_every)
    def count(self, split_every=None):
        
        return self.reduction(count, sum, split_every=split_every)
    def mean(self):
        
        def mean_chunk(seq):
            total, n = 0.0, 0
            for x in seq:
                total += x
                n += 1
            return total, n
        def mean_aggregate(x):
            totals, counts = list(zip(*x))
            return 1.0 * sum(totals) / sum(counts)
        return self.reduction(mean_chunk, mean_aggregate, split_every=False)
    def var(self, ddof=0):
        
        return self.reduction(
            chunk.var_chunk, partial(chunk.var_aggregate, ddof=ddof), split_every=False
        )
    def std(self, ddof=0):
        
        return self.var(ddof=ddof).apply(math.sqrt)
    def join(self, other, on_self, on_other=None):
        
        name = "join-" + tokenize(self, other, on_self, on_other)
        dsk = {}
        if isinstance(other, Bag):
            if other.npartitions == 1:
                dsk.update(other.dask)
                other = other.__dask_keys__()[0]
                dsk["join-%s-other" % name] = (list, other)
            else:
                msg = (
                    "Multi-bag joins are not implemented. "
                    "We recommend Dask dataframe if appropriate"
                )
                raise NotImplementedError(msg)
        elif isinstance(other, Delayed):
            dsk.update(other.dask)
            other = other._key
        elif isinstance(other, Iterable):
            other = other
        else:
            msg = (
                "Joined argument must be single-partition Bag, "
                " delayed object, or Iterable, got %s" % type(other).__name
            )
            raise TypeError(msg)
        if on_other is None:
            on_other = on_self
        dsk.update(
            {
                (name, i): (list, (join, on_other, other, on_self, (self.name, i)))
                for i in range(self.npartitions)
            }
        )
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return type(self)(graph, name, self.npartitions)
    def product(self, other):
        
        assert isinstance(other, Bag)
        name = "product-" + tokenize(self, other)
        n, m = self.npartitions, other.npartitions
        dsk = dict(
            (
                (name, i * m + j),
                (list, (itertools.product, (self.name, i), (other.name, j))),
            )
            for i in range(n)
            for j in range(m)
        )
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self, other])
        return type(self)(graph, name, n * m)
    def foldby(
        self,
        key,
        binop,
        initial=no_default,
        combine=None,
        combine_initial=no_default,
        split_every=None,
    ):
        
        if split_every is None:
            split_every = 8
        if split_every is False:
            split_every = self.npartitions
        token = tokenize(self, key, binop, initial, combine, combine_initial)
        a = "foldby-a-" + token
        if combine is None:
            combine = binop
        if initial is not no_default:
            dsk = {
                (a, i): (reduceby, key, binop, (self.name, i), initial)
                for i in range(self.npartitions)
            }
        else:
            dsk = {
                (a, i): (reduceby, key, binop, (self.name, i))
                for i in range(self.npartitions)
            }
        combine2 = partial(chunk.foldby_combine2, combine)
        depth = 0
        k = self.npartitions
        b = a
        while k > split_every:
            c = b + str(depth)
            if combine_initial is not no_default:
                dsk2 = {
                    (c, i): (
                        reduceby,
                        0,
                        combine2,
                        (toolz.concat, (map, dictitems, [(b, j) for j in inds])),
                        combine_initial,
                    )
                    for i, inds in enumerate(partition_all(split_every, range(k)))
                }
            else:
                dsk2 = {
                    (c, i): (
                        merge_with,
                        (partial, reduce, combine),
                        [(b, j) for j in inds],
                    )
                    for i, inds in enumerate(partition_all(split_every, range(k)))
                }
            dsk.update(dsk2)
            k = len(dsk2)
            b = c
            depth += 1
        e = "foldby-b-" + token
        if combine_initial is not no_default:
            dsk[(e, 0)] = (
                dictitems,
                (
                    reduceby,
                    0,
                    combine2,
                    (toolz.concat, (map, dictitems, [(b, j) for j in range(k)])),
                    combine_initial,
                ),
            )
        else:
            dsk[(e, 0)] = (
                dictitems,
                (merge_with, (partial, reduce, combine), [(b, j) for j in range(k)]),
            )
        graph = HighLevelGraph.from_collections(e, dsk, dependencies=[self])
        return type(self)(graph, e, 1)
    def take(self, k, npartitions=1, compute=True, warn=True):
        
        if npartitions <= -1:
            npartitions = self.npartitions
        if npartitions > self.npartitions:
            raise ValueError(
                "only {} partitions, take "
                "received {}".format(self.npartitions, npartitions)
            )
        token = tokenize(self, k, npartitions)
        name = "take-" + token
        if npartitions > 1:
            name_p = "take-partial-" + token
            dsk = {}
            for i in range(npartitions):
                dsk[(name_p, i)] = (list, (take, k, (self.name, i)))
            concat = (toolz.concat, ([(name_p, i) for i in range(npartitions)]))
            dsk[(name, 0)] = (safe_take, k, concat, warn)
        else:
            dsk = {(name, 0): (safe_take, k, (self.name, 0), warn)}
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        b = Bag(graph, name, 1)
        if compute:
            return tuple(b.compute())
        else:
            return b
    def flatten(self):
        
        name = "flatten-" + tokenize(self)
        dsk = dict(
            ((name, i), (list, (toolz.concat, (self.name, i))))
            for i in range(self.npartitions)
        )
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return type(self)(graph, name, self.npartitions)
    def __iter__(self):
        return iter(self.compute())
    def groupby(
        self,
        grouper,
        method=None,
        npartitions=None,
        blocksize=2 ** 20,
        max_branch=None,
        shuffle=None,
    ):
        
        if method is not None:
            raise Exception("The method= keyword has been moved to shuffle=")
        if shuffle is None:
            shuffle = config.get("shuffle", None)
        if shuffle is None:
            if "distributed" in config.get("scheduler", ""):
                shuffle = "tasks"
            else:
                shuffle = "disk"
        if shuffle == "disk":
            return groupby_disk(
                self, grouper, npartitions=npartitions, blocksize=blocksize
            )
        elif shuffle == "tasks":
            return groupby_tasks(self, grouper, max_branch=max_branch)
        else:
            msg = "Shuffle must be 'disk' or 'tasks'"
            raise NotImplementedError(msg)
    def to_dataframe(self, meta=None, columns=None):
        
        import pandas as pd
        import dask.dataframe as dd
        if meta is None:
            head = self.take(1, warn=False)
            if len(head) == 0:
                raise ValueError(
                    "`dask.bag.Bag.to_dataframe` failed to "
                    "properly infer metadata, please pass in "
                    "metadata via the `meta` keyword"
                )
            meta = pd.DataFrame(list(head), columns=columns)
        elif columns is not None:
            raise ValueError("Can't specify both `meta` and `columns`")
        else:
            meta = dd.utils.make_meta(meta)
        # Serializing the columns and dtypes is much smaller than serializing
        # the empty frame
        cols = list(meta.columns)
        dtypes = meta.dtypes.to_dict()
        name = "to_dataframe-" + tokenize(self, cols, dtypes)
        dsk = self.__dask_optimize__(self.dask, self.__dask_keys__())
        dsk.update(
            {
                (name, i): (to_dataframe, (self.name, i), cols, dtypes)
                for i in range(self.npartitions)
            }
        )
        divisions = [None] * (self.npartitions + 1)
        return dd.DataFrame(dsk, name, meta, divisions)
    def to_delayed(self, optimize_graph=True):
        
        from dask.delayed import Delayed
        keys = self.__dask_keys__()
        dsk = self.__dask_graph__()
        if optimize_graph:
            dsk = self.__dask_optimize__(dsk, keys)
        return [Delayed(k, dsk) for k in keys]
    def repartition(self, npartitions):
        
        new_name = "repartition-%d-%s" % (npartitions, tokenize(self, npartitions))
        if npartitions == self.npartitions:
            return self
        elif npartitions < self.npartitions:
            ratio = self.npartitions / npartitions
            new_partitions_boundaries = [
                int(old_partition_index * ratio)
                for old_partition_index in range(npartitions + 1)
            ]
            dsk = {}
            for new_partition_index in range(npartitions):
                value = (
                    list,
                    (
                        toolz.concat,
                        [
                            (self.name, old_partition_index)
                            for old_partition_index in range(
                                new_partitions_boundaries[new_partition_index],
                                new_partitions_boundaries[new_partition_index + 1],
                            )
                        ],
                    ),
                )
                dsk[new_name, new_partition_index] = value
        else:  # npartitions > self.npartitions
            ratio = npartitions / self.npartitions
            split_name = "split-%s" % tokenize(self, npartitions)
            dsk = {}
            last = 0
            j = 0
            for i in range(self.npartitions):
                new = last + ratio
                if i == self.npartitions - 1:
                    k = npartitions - j
                else:
                    k = int(new - last)
                dsk[(split_name, i)] = (split, (self.name, i), k)
                for jj in range(k):
                    dsk[(new_name, j)] = (operator.getitem, (split_name, i), jj)
                    j += 1
                last = new
        graph = HighLevelGraph.from_collections(new_name, dsk, dependencies=[self])
        return Bag(graph, name=new_name, npartitions=npartitions)
    def accumulate(self, binop, initial=no_default):
        
        if not _implement_accumulate:
            raise NotImplementedError(
                "accumulate requires `toolz` > 0.7.4 or `cytoolz` > 0.7.3."
            )
        token = tokenize(self, binop, initial)
        binop_name = funcname(binop)
        a = "%s-part-%s" % (binop_name, token)
        b = "%s-first-%s" % (binop_name, token)
        c = "%s-second-%s" % (binop_name, token)
        dsk = {
            (a, 0): (accumulate_part, binop, (self.name, 0), initial, True),
            (b, 0): (first, (a, 0)),
            (c, 0): (second, (a, 0)),
        }
        for i in range(1, self.npartitions):
            dsk[(a, i)] = (accumulate_part, binop, (self.name, i), (c, i - 1))
            dsk[(b, i)] = (first, (a, i))
            dsk[(c, i)] = (second, (a, i))
        graph = HighLevelGraph.from_collections(b, dsk, dependencies=[self])
        return Bag(graph, b, self.npartitions)
def accumulate_part(binop, seq, initial, is_first=False):
    if initial == no_default:
        res = list(accumulate(binop, seq))
    else:
        res = list(accumulate(binop, seq, initial=initial))
    if is_first:
        return res, res[-1] if res else [], initial
    return res[1:], res[-1]
def partition(grouper, sequence, npartitions, p, nelements=2 ** 20):
    
    for block in partition_all(nelements, sequence):
        d = groupby(grouper, block)
        d2 = defaultdict(list)
        for k, v in d.items():
            d2[abs(hash(k)) % npartitions].extend(v)
        p.append(d2, fsync=True)
    return p
def collect(grouper, group, p, barrier_token):
    
    d = groupby(grouper, p.get(group, lock=False))
    return list(d.items())
def from_sequence(seq, partition_size=None, npartitions=None):
    
    seq = list(seq)
    if npartitions and not partition_size:
        partition_size = int(math.ceil(len(seq) / npartitions))
    if npartitions is None and partition_size is None:
        if len(seq) < 100:
            partition_size = 1
        else:
            partition_size = int(len(seq) / 100)
    parts = list(partition_all(partition_size, seq))
    name = "from_sequence-" + tokenize(seq, partition_size)
    if len(parts) > 0:
        d = dict(((name, i), list(part)) for i, part in enumerate(parts))
    else:
        d = {(name, 0): []}
    return Bag(d, name, len(d))
def from_url(urls):
    
    if isinstance(urls, str):
        urls = [urls]
    name = "from_url-" + uuid.uuid4().hex
    dsk = {}
    for i, u in enumerate(urls):
        dsk[(name, i)] = (list, (urlopen, u))
    return Bag(dsk, name, len(urls))
def dictitems(d):
    
    return list(d.items())
def concat(bags):
    
    name = "concat-" + tokenize(*bags)
    counter = itertools.count(0)
    dsk = {(name, next(counter)): key for bag in bags for key in bag.__dask_keys__()}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=bags)
    return Bag(graph, name, len(dsk))
def reify(seq):
    if isinstance(seq, Iterator):
        seq = list(seq)
    if seq and isinstance(seq[0], Iterator):
        seq = list(map(list, seq))
    return seq
def from_delayed(values):
    
    from dask.delayed import Delayed, delayed
    if isinstance(values, Delayed):
        values = [values]
    values = [
        delayed(v) if not isinstance(v, Delayed) and hasattr(v, "key") else v
        for v in values
    ]
    name = "bag-from-delayed-" + tokenize(*values)
    names = [(name, i) for i in range(len(values))]
    values2 = [(reify, v.key) for v in values]
    dsk = dict(zip(names, values2))
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=values)
    return Bag(graph, name, len(values))
def chunk_distinct(seq, key=None):
    if key is not None and not callable(key):
        key = partial(chunk.getitem, key=key)
    return list(unique(seq, key=key))
def merge_distinct(seqs, key=None):
    return chunk_distinct(toolz.concat(seqs), key=key)
def merge_frequencies(seqs):
    if isinstance(seqs, Iterable):
        seqs = list(seqs)
    if not seqs:
        return {}
    first, rest = seqs[0], seqs[1:]
    if not rest:
        return first
    out = defaultdict(int)
    out.update(first)
    for d in rest:
        for k, v in d.items():
            out[k] += v
    return out
def bag_range(n, npartitions):
    
    size = n // npartitions
    name = "range-%d-npartitions-%d" % (n, npartitions)
    ijs = list(enumerate(take(npartitions, range(0, n, size))))
    dsk = dict(((name, i), (reify, (range, j, min(j + size, n)))) for i, j in ijs)
    if n % npartitions != 0:
        i, j = ijs[-1]
        dsk[(name, i)] = (reify, (range, j, n))
    return Bag(dsk, name, npartitions)
def bag_zip(*bags):
    
    npartitions = bags[0].npartitions
    assert all(bag.npartitions == npartitions for bag in bags)
    # TODO: do more checks
    name = "zip-" + tokenize(*bags)
    dsk = dict(
        ((name, i), (reify, (zip,) + tuple((bag.name, i) for bag in bags)))
        for i in range(npartitions)
    )
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=bags)
    return Bag(graph, name, npartitions)
def map_chunk(f, iters, iter_kwarg_keys=None, kwargs=None):
    
    if kwargs:
        f = partial(f, **kwargs)
    iters = [iter(a) for a in iters]
    return _MapChunk(f, iters, kwarg_keys=iter_kwarg_keys)
class _MapChunk(Iterator):
    def __init__(self, f, iters, kwarg_keys=None):
        self.f = f
        self.iters = iters
        self.kwarg_keys = kwarg_keys or ()
        self.nkws = len(self.kwarg_keys)
    def __next__(self):
        try:
            vals = [next(i) for i in self.iters]
        except StopIteration:
            self.check_all_iterators_consumed()
            raise
        if self.nkws:
            args = vals[: -self.nkws]
            kwargs = dict(zip(self.kwarg_keys, vals[-self.nkws :]))
            return self.f(*args, **kwargs)
        return self.f(*vals)
    def check_all_iterators_consumed(self):
        if len(self.iters) > 1:
            for i in self.iters:
                if isinstance(i, itertools.repeat):
                    continue
                try:
                    next(i)
                except StopIteration:
                    pass
                else:
                    msg = (
                        "map called with multiple bags that aren't identically "
                        "partitioned. Please ensure that all bag arguments "
                        "have the same partition lengths"
                    )
                    raise ValueError(msg)
def starmap_chunk(f, x, kwargs):
    if kwargs:
        f = partial(f, **kwargs)
    return itertools.starmap(f, x)
def unpack_scalar_dask_kwargs(kwargs):
    
    kwargs2 = {}
    dependencies = []
    for k, v in kwargs.items():
        vv, collections = unpack_collections(v)
        if not collections:
            kwargs2[k] = v
        else:
            kwargs2[k] = vv
            dependencies.extend(collections)
    if dependencies:
        kwargs2 = (dict, (zip, list(kwargs2), list(kwargs2.values())))
    return kwargs2, dependencies
def bag_map(func, *args, **kwargs):
    
    name = "%s-%s" % (funcname(func), tokenize(func, "map", *args, **kwargs))
    dsk = {}
    dependencies = []
    bags = []
    args2 = []
    for a in args:
        if isinstance(a, Bag):
            bags.append(a)
            args2.append(a)
        elif isinstance(a, (Item, Delayed)):
            dependencies.append(a)
            args2.append((itertools.repeat, a.key))
        else:
            args2.append((itertools.repeat, a))
    bag_kwargs = {}
    other_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Bag):
            bag_kwargs[k] = v
            bags.append(v)
        else:
            other_kwargs[k] = v
    other_kwargs, collections = unpack_scalar_dask_kwargs(other_kwargs)
    dependencies.extend(collections)
    if not bags:
        raise ValueError("At least one argument must be a Bag.")
    npartitions = {b.npartitions for b in bags}
    if len(npartitions) > 1:
        raise ValueError("All bags must have the same number of partitions.")
    npartitions = npartitions.pop()
    def build_iters(n):
        args = [(a.name, n) if isinstance(a, Bag) else a for a in args2]
        if bag_kwargs:
            args.extend((b.name, n) for b in bag_kwargs.values())
        return args
    if bag_kwargs:
        iter_kwarg_keys = list(bag_kwargs)
    else:
        iter_kwarg_keys = None
    dsk = {
        (name, n): (
            reify,
            (map_chunk, func, build_iters(n), iter_kwarg_keys, other_kwargs),
        )
        for n in range(npartitions)
    }
    # If all bags are the same type, use that type, otherwise fallback to Bag
    return_type = set(map(type, bags))
    return_type = return_type.pop() if len(return_type) == 1 else Bag
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=bags + dependencies)
    return return_type(graph, name, npartitions)
def map_partitions(func, *args, **kwargs):
    
    name = "%s-%s" % (funcname(func), tokenize(func, "map-partitions", *args, **kwargs))
    dsk = {}
    dependencies = []
    bags = []
    args2 = []
    for a in args:
        if isinstance(a, Bag):
            bags.append(a)
            args2.append(a)
        elif isinstance(a, (Item, Delayed)):
            args2.append(a.key)
            dependencies.append(a)
        else:
            args2.append(a)
    bag_kwargs = {}
    other_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Bag):
            bag_kwargs[k] = v
            bags.append(v)
        else:
            other_kwargs[k] = v
    other_kwargs, collections = unpack_scalar_dask_kwargs(other_kwargs)
    dependencies.extend(collections)
    if not bags:
        raise ValueError("At least one argument must be a Bag.")
    npartitions = {b.npartitions for b in bags}
    if len(npartitions) > 1:
        raise ValueError("All bags must have the same number of partitions.")
    npartitions = npartitions.pop()
    def build_args(n):
        return [(a.name, n) if isinstance(a, Bag) else a for a in args2]
    def build_bag_kwargs(n):
        if not bag_kwargs:
            return {}
        return (
            dict,
            (zip, list(bag_kwargs), [(b.name, n) for b in bag_kwargs.values()]),
        )
    if kwargs:
        dsk = {
            (name, n): (
                apply,
                func,
                build_args(n),
                (merge, build_bag_kwargs(n), other_kwargs),
            )
            for n in range(npartitions)
        }
    else:
        dsk = {(name, n): (func,) + tuple(build_args(n)) for n in range(npartitions)}
    # If all bags are the same type, use that type, otherwise fallback to Bag
    return_type = set(map(type, bags))
    return_type = return_type.pop() if len(return_type) == 1 else Bag
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=bags + dependencies)
    return return_type(graph, name, npartitions)
def _reduce(binop, sequence, initial=no_default):
    if initial is not no_default:
        return reduce(binop, sequence, initial)
    else:
        return reduce(binop, sequence)
def make_group(k, stage):
    def h(x):
        return x[0] // k ** stage % k
    return h
def groupby_tasks(b, grouper, hash=hash, max_branch=32):
    max_branch = max_branch or 32
    n = b.npartitions
    stages = int(math.ceil(math.log(n) / math.log(max_branch))) or 1
    if stages > 1:
        k = int(math.ceil(n ** (1 / stages)))
    else:
        k = n
    groups = []
    splits = []
    joins = []
    inputs = [tuple(digit(i, j, k) for j in range(stages)) for i in range(k ** stages)]
    b2 = b.map(partial(chunk.groupby_tasks_group_hash, hash=hash, grouper=grouper))
    token = tokenize(b, grouper, hash, max_branch)
    start = dict(
        (("shuffle-join-" + token, 0, inp), (b2.name, i) if i < b.npartitions else [])
        for i, inp in enumerate(inputs)
    )
    for stage in range(1, stages + 1):
        group = dict(
            (
                ("shuffle-group-" + token, stage, inp),
                (
                    groupby,
                    (make_group, k, stage - 1),
                    ("shuffle-join-" + token, stage - 1, inp),
                ),
            )
            for inp in inputs
        )
        split = dict(
            (
                ("shuffle-split-" + token, stage, i, inp),
                (dict.get, ("shuffle-group-" + token, stage, inp), i, {}),
            )
            for i in range(k)
            for inp in inputs
        )
        join = dict(
            (
                ("shuffle-join-" + token, stage, inp),
                (
                    list,
                    (
                        toolz.concat,
                        [
                            (
                                "shuffle-split-" + token,
                                stage,
                                inp[stage - 1],
                                insert(inp, stage - 1, j),
                            )
                            for j in range(k)
                        ],
                    ),
                ),
            )
            for inp in inputs
        )
        groups.append(group)
        splits.append(split)
        joins.append(join)
    end = dict(
        (
            ("shuffle-" + token, i),
            (list, (dict.items, (groupby, grouper, (pluck, 1, j)))),
        )
        for i, j in enumerate(join)
    )
    name = "shuffle-" + token
    dsk = merge(start, end, *(groups + splits + joins))
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b2])
    return type(b)(graph, name, len(inputs))
def groupby_disk(b, grouper, npartitions=None, blocksize=2 ** 20):
    if npartitions is None:
        npartitions = b.npartitions
    token = tokenize(b, grouper, npartitions, blocksize)
    import partd
    p = ("partd-" + token,)
    dirname = config.get("temporary_directory", None)
    if dirname:
        file = (apply, partd.File, (), {"dir": dirname})
    else:
        file = (partd.File,)
    try:
        dsk1 = {p: (partd.Python, (partd.Snappy, file))}
    except AttributeError:
        dsk1 = {p: (partd.Python, file)}
    # Partition data on disk
    name = "groupby-part-{0}-{1}".format(funcname(grouper), token)
    dsk2 = dict(
        ((name, i), (partition, grouper, (b.name, i), npartitions, p, blocksize))
        for i in range(b.npartitions)
    )
    # Barrier
    barrier_token = "groupby-barrier-" + token
    dsk3 = {barrier_token: (chunk.barrier,) + tuple(dsk2)}
    # Collect groups
    name = "groupby-collect-" + token
    dsk4 = dict(
        ((name, i), (collect, grouper, i, p, barrier_token)) for i in range(npartitions)
    )
    dsk = merge(dsk1, dsk2, dsk3, dsk4)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b])
    return type(b)(graph, name, npartitions)
def empty_safe_apply(func, part, is_last):
    if isinstance(part, Iterator):
        try:
            _, part = peek(part)
        except StopIteration:
            if not is_last:
                return no_result
        return func(part)
    elif not is_last and len(part) == 0:
        return no_result
    else:
        return func(part)
def empty_safe_aggregate(func, parts, is_last):
    parts2 = (p for p in parts if p is not no_result)
    return empty_safe_apply(func, parts2, is_last)
def safe_take(n, b, warn=True):
    r = list(take(n, b))
    if len(r) != n and warn:
        warnings.warn(
            "Insufficient elements for `take`. {0} elements "
            "requested, only {1} elements available. Try passing "
            "larger `npartitions` to `take`.".format(n, len(r))
        )
    return r
def random_sample(x, state_data, prob):
    
    random_state = Random()
    random_state.setstate(state_data)
    for i in x:
        if random_state.random() < prob:
            yield i
def random_state_data_python(n, random_state=None):
    
    if not isinstance(random_state, Random):
        random_state = Random(random_state)
    maxuint32 = 1 << 32
    return [
        (
            3,
            tuple(random_state.randint(0, maxuint32) for i in range(624)) + (624,),
            None,
        )
        for i in range(n)
    ]
def split(seq, n):
    
    if not isinstance(seq, (list, tuple)):
        seq = list(seq)
    part = len(seq) / n
    L = [seq[int(part * i) : int(part * (i + 1))] for i in range(n - 1)]
    L.append(seq[int(part * (n - 1)) :])
    return L
def to_dataframe(seq, columns, dtypes):
    import pandas as pd
    seq = reify(seq)
    # pd.DataFrame expects lists, only copy if necessary
    if not isinstance(seq, list):
        seq = list(seq)
    res = pd.DataFrame(seq, columns=list(columns))
    return res.astype(dtypes, copy=False)
