from math import ceil
from operator import getitem
import os
from threading import Lock
import pandas as pd
import numpy as np
from toolz import merge
from ...base import tokenize
from ... import array as da
from ...delayed import delayed
from ..core import DataFrame, Series, Index, new_dd_object, has_parallel_type
from ..shuffle import set_partition
from ..utils import insert_meta_param_description, check_meta, make_meta
from ...utils import M, ensure_dict
lock = Lock()
def _meta_from_array(x, columns=None, index=None):
    
    if x.ndim > 2:
        raise ValueError(
            "from_array does not input more than 2D array, got"
            " array with shape %r" % (x.shape,)
        )
    if index is not None:
        if not isinstance(index, Index):
            raise ValueError("'index' must be an instance of dask.dataframe.Index")
        index = index._meta
    if getattr(x.dtype, "names", None) is not None:
        # record array has named columns
        if columns is None:
            columns = list(x.dtype.names)
        elif np.isscalar(columns):
            raise ValueError("For a struct dtype, columns must be a list.")
        elif not all(i in x.dtype.names for i in columns):
            extra = sorted(set(columns).difference(x.dtype.names))
            raise ValueError("dtype {0} doesn't have fields {1}".format(x.dtype, extra))
        fields = x.dtype.fields
        dtypes = [fields[n][0] if n in fields else "f8" for n in columns]
    elif x.ndim == 1:
        if np.isscalar(columns) or columns is None:
            return pd.Series([], name=columns, dtype=x.dtype, index=index)
        elif len(columns) == 1:
            return pd.DataFrame(
                np.array([], dtype=x.dtype), columns=columns, index=index
            )
        raise ValueError(
            "For a 1d array, columns must be a scalar or single element list"
        )
    else:
        if np.isnan(x.shape[1]):
            raise ValueError("Shape along axis 1 must be known")
        if columns is None:
            columns = list(range(x.shape[1])) if x.ndim == 2 else [0]
        elif len(columns) != x.shape[1]:
            raise ValueError(
                "Number of column names must match width of the "
                "array. Got {0} names for {1} "
                "columns".format(len(columns), x.shape[1])
            )
        dtypes = [x.dtype] * len(columns)
    data = {c: np.array([], dtype=dt) for (c, dt) in zip(columns, dtypes)}
    return pd.DataFrame(data, columns=columns, index=index)
def from_array(x, chunksize=50000, columns=None):
    
    if isinstance(x, da.Array):
        return from_dask_array(x, columns=columns)
    meta = _meta_from_array(x, columns)
    divisions = tuple(range(0, len(x), chunksize))
    divisions = divisions + (len(x) - 1,)
    token = tokenize(x, chunksize, columns)
    name = "from_array-" + token
    dsk = {}
    for i in range(0, int(ceil(len(x) / chunksize))):
        data = (getitem, x, slice(i * chunksize, (i + 1) * chunksize))
        if isinstance(meta, pd.Series):
            dsk[name, i] = (pd.Series, data, None, meta.dtype, meta.name)
        else:
            dsk[name, i] = (pd.DataFrame, data, None, meta.columns)
    return new_dd_object(dsk, name, meta, divisions)
def from_pandas(data, npartitions=None, chunksize=None, sort=True, name=None):
    
    if isinstance(getattr(data, "index", None), pd.MultiIndex):
        raise NotImplementedError("Dask does not support MultiIndex Dataframes.")
    if not has_parallel_type(data):
        raise TypeError("Input must be a pandas DataFrame or Series")
    if (npartitions is None) == (chunksize is None):
        raise ValueError("Exactly one of npartitions and chunksize must be specified.")
    nrows = len(data)
    if chunksize is None:
        chunksize = int(ceil(nrows / npartitions))
    else:
        npartitions = int(ceil(nrows / chunksize))
    name = name or ("from_pandas-" + tokenize(data, chunksize))
    if not nrows:
        return new_dd_object({(name, 0): data}, name, data, [None, None])
    if sort and not data.index.is_monotonic_increasing:
        data = data.sort_index(ascending=True)
    if sort:
        divisions, locations = sorted_division_locations(
            data.index, chunksize=chunksize
        )
    else:
        locations = list(range(0, nrows, chunksize)) + [len(data)]
        divisions = [None] * len(locations)
    dsk = {
        (name, i): data.iloc[start:stop]
        for i, (start, stop) in enumerate(zip(locations[:-1], locations[1:]))
    }
    return new_dd_object(dsk, name, data, divisions)
def from_bcolz(x, chunksize=None, categorize=True, index=None, lock=lock, **kwargs):
    
    if lock is True:
        lock = Lock()
    import dask.array as da
    import bcolz
    if isinstance(x, str):
        x = bcolz.ctable(rootdir=x)
    bc_chunklen = max(x[name].chunklen for name in x.names)
    if chunksize is None and bc_chunklen > 10000:
        chunksize = bc_chunklen
    categories = dict()
    if categorize:
        for name in x.names:
            if (
                np.issubdtype(x.dtype[name], np.string_)
                or np.issubdtype(x.dtype[name], np.unicode_)
                or np.issubdtype(x.dtype[name], np.object_)
            ):
                a = da.from_array(x[name], chunks=(chunksize * len(x.names),))
                categories[name] = da.unique(a).compute()
    columns = tuple(x.dtype.names)
    divisions = tuple(range(0, len(x), chunksize))
    divisions = divisions + (len(x) - 1,)
    if x.rootdir:
        token = tokenize(
            (x.rootdir, os.path.getmtime(x.rootdir)),
            chunksize,
            categorize,
            index,
            kwargs,
        )
    else:
        token = tokenize(
            (id(x), x.shape, x.dtype), chunksize, categorize, index, kwargs
        )
    new_name = "from_bcolz-" + token
    dsk = dict(
        (
            (new_name, i),
            (
                dataframe_from_ctable,
                x,
                (slice(i * chunksize, (i + 1) * chunksize),),
                columns,
                categories,
                lock,
            ),
        )
        for i in range(0, int(ceil(len(x) / chunksize)))
    )
    meta = dataframe_from_ctable(x, slice(0, 0), columns, categories, lock)
    result = DataFrame(dsk, new_name, meta, divisions)
    if index:
        assert index in x.names
        a = da.from_array(x[index], chunks=(chunksize * len(x.names),))
        q = np.linspace(0, 100, len(x) // chunksize + 2)
        divisions = tuple(da.percentile(a, q).compute())
        return set_partition(result, index, divisions, **kwargs)
    else:
        return result
def dataframe_from_ctable(x, slc, columns=None, categories=None, lock=lock):
    
    import bcolz
    if columns is None:
        columns = x.dtype.names
    if isinstance(columns, tuple):
        columns = list(columns)
    x = x[columns]
    if type(slc) is slice:
        start = slc.start
        stop = slc.stop if slc.stop < len(x) else len(x)
    else:
        start = slc[0].start
        stop = slc[0].stop if slc[0].stop < len(x) else len(x)
    idx = pd.Index(range(start, stop))
    if lock:
        lock.acquire()
    try:
        if isinstance(x, bcolz.ctable):
            chunks = [x[name][slc] for name in columns]
            if categories is not None:
                chunks = [
                    pd.Categorical.from_codes(
                        np.searchsorted(categories[name], chunk), categories[name], True
                    )
                    if name in categories
                    else chunk
                    for name, chunk in zip(columns, chunks)
                ]
            result = pd.DataFrame(
                dict(zip(columns, chunks)), columns=columns, index=idx
            )
        elif isinstance(x, bcolz.carray):
            chunk = x[slc]
            if categories is not None and columns and columns in categories:
                chunk = pd.Categorical.from_codes(
                    np.searchsorted(categories[columns], chunk),
                    categories[columns],
                    True,
                )
            result = pd.Series(chunk, name=columns, index=idx)
    finally:
        if lock:
            lock.release()
    return result
def from_dask_array(x, columns=None, index=None):
    
    meta = _meta_from_array(x, columns, index)
    if x.ndim == 2 and len(x.chunks[1]) > 1:
        x = x.rechunk({1: x.shape[1]})
    name = "from-dask-array" + tokenize(x, columns)
    to_merge = []
    if index is not None:
        if not isinstance(index, Index):
            raise ValueError("'index' must be an instance of dask.dataframe.Index")
        if index.npartitions != x.numblocks[0]:
            msg = (
                "The index and array have different numbers of blocks. "
                "({} != {})".format(index.npartitions, x.numblocks[0])
            )
            raise ValueError(msg)
        divisions = index.divisions
        to_merge.append(ensure_dict(index.dask))
        index = index.__dask_keys__()
    elif np.isnan(sum(x.shape)):
        divisions = [None] * (len(x.chunks[0]) + 1)
        index = [None] * len(x.chunks[0])
    else:
        divisions = [0]
        for c in x.chunks[0]:
            divisions.append(divisions[-1] + c)
        index = [
            (np.arange, a, b, 1, "i8") for a, b in zip(divisions[:-1], divisions[1:])
        ]
        divisions[-1] -= 1
    dsk = {}
    for i, (chunk, ind) in enumerate(zip(x.__dask_keys__(), index)):
        if x.ndim == 2:
            chunk = chunk[0]
        if isinstance(meta, pd.Series):
            dsk[name, i] = (pd.Series, chunk, ind, x.dtype, meta.name)
        else:
            dsk[name, i] = (pd.DataFrame, chunk, ind, meta.columns)
    to_merge.extend([ensure_dict(x.dask), dsk])
    return new_dd_object(merge(*to_merge), name, meta, divisions)
def _link(token, result):
    
    return None
def _df_to_bag(df, index=False):
    if isinstance(df, pd.DataFrame):
        return list(map(tuple, df.itertuples(index)))
    elif isinstance(df, pd.Series):
        return list(df.iteritems()) if index else list(df)
def to_bag(df, index=False):
    
    from ...bag.core import Bag
    if not isinstance(df, (DataFrame, Series)):
        raise TypeError("df must be either DataFrame or Series")
    name = "to_bag-" + tokenize(df, index)
    dsk = dict(
        ((name, i), (_df_to_bag, block, index))
        for (i, block) in enumerate(df.__dask_keys__())
    )
    dsk.update(df.__dask_optimize__(df.__dask_graph__(), df.__dask_keys__()))
    return Bag(dsk, name, df.npartitions)
def to_records(df):
    
    return df.map_partitions(M.to_records)
@insert_meta_param_description
def from_delayed(
    dfs, meta=None, divisions=None, prefix="from-delayed", verify_meta=True
):
    
    from dask.delayed import Delayed
    if isinstance(dfs, Delayed):
        dfs = [dfs]
    dfs = [
        delayed(df) if not isinstance(df, Delayed) and hasattr(df, "key") else df
        for df in dfs
    ]
    for df in dfs:
        if not isinstance(df, Delayed):
            raise TypeError("Expected Delayed object, got %s" % type(df).__name__)
    if meta is None:
        meta = delayed(make_meta)(dfs[0]).compute()
    else:
        meta = make_meta(meta)
    name = prefix + "-" + tokenize(*dfs)
    dsk = merge(df.dask for df in dfs)
    if verify_meta:
        dsk.update(
            {
                (name, i): (check_meta, df.key, meta, "from_delayed")
                for (i, df) in enumerate(dfs)
            }
        )
    else:
        dsk.update({(name, i): df.key for (i, df) in enumerate(dfs)})
    if divisions is None or divisions == "sorted":
        divs = [None] * (len(dfs) + 1)
    else:
        divs = tuple(divisions)
        if len(divs) != len(dfs) + 1:
            raise ValueError("divisions should be a tuple of len(dfs) + 1")
    df = new_dd_object(dsk, name, meta, divs)
    if divisions == "sorted":
        from ..shuffle import compute_divisions
        divisions = compute_divisions(df)
        df.divisions = divisions
    return df
def sorted_division_locations(seq, npartitions=None, chunksize=None):
    
    if (npartitions is None) == (chunksize is None):
        raise ValueError("Exactly one of npartitions and chunksize must be specified.")
    if npartitions:
        chunksize = ceil(len(seq) / npartitions)
    positions = [0]
    values = [seq[0]]
    for pos in list(range(0, len(seq), chunksize)):
        if pos <= positions[-1]:
            continue
        while pos + 1 < len(seq) and seq[pos - 1] == seq[pos]:
            pos += 1
        values.append(seq[pos])
        if pos == len(seq) - 1:
            pos += 1
        positions.append(pos)
    if positions[-1] != len(seq):
        positions.append(len(seq))
        values.append(seq[-1])
    return values, positions
DataFrame.to_records.__doc__ = to_records.__doc__
DataFrame.to_bag.__doc__ = to_bag.__doc__
