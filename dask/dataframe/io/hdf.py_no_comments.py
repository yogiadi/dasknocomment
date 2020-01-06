from fnmatch import fnmatch
from glob import glob
import os
import uuid
from warnings import warn
import pandas as pd
from toolz import merge
# this import checks for the importability of fsspec
from ...bytes import read_bytes  # noqa
from fsspec.utils import build_name_function, stringify_path
from .io import _link
from ...base import get_scheduler
from ..core import DataFrame, new_dd_object
from ... import config, multiprocessing
from ...base import tokenize, compute_as_if_collection
from ...delayed import Delayed, delayed
from ...utils import get_scheduler_lock
def _pd_to_hdf(pd_to_hdf, lock, args, kwargs=None):
    
    if lock:
        lock.acquire()
    try:
        pd_to_hdf(*args, **kwargs)
    finally:
        if lock:
            lock.release()
    return None
def to_hdf(
    df,
    path,
    key,
    mode="a",
    append=False,
    scheduler=None,
    name_function=None,
    compute=True,
    lock=None,
    dask_kwargs={},
    **kwargs
):
    
    name = "to-hdf-" + uuid.uuid1().hex
    pd_to_hdf = getattr(df._partition_type, "to_hdf")
    single_file = True
    single_node = True
    path = stringify_path(path)
    # if path is string, format using i_name
    if isinstance(path, str):
        if path.count("*") + key.count("*") > 1:
            raise ValueError(
                "A maximum of one asterisk is accepted in file path and dataset key"
            )
        fmt_obj = lambda path, i_name: path.replace("*", i_name)
        if "*" in path:
            single_file = False
    else:
        if key.count("*") > 1:
            raise ValueError("A maximum of one asterisk is accepted in dataset key")
        fmt_obj = lambda path, _: path
    if "*" in key:
        single_node = False
    if "format" in kwargs and kwargs["format"] not in ["t", "table"]:
        raise ValueError("Dask only support 'table' format in hdf files.")
    if mode not in ("a", "w", "r+"):
        raise ValueError("Mode must be one of 'a', 'w' or 'r+'")
    if name_function is None:
        name_function = build_name_function(df.npartitions - 1)
    # we guarantee partition order is preserved when its saved and read
    # so we enforce name_function to maintain the order of its input.
    if not (single_file and single_node):
        formatted_names = [name_function(i) for i in range(df.npartitions)]
        if formatted_names != sorted(formatted_names):
            warn(
                "To preserve order between partitions name_function "
                "must preserve the order of its input"
            )
    # If user did not specify scheduler and write is sequential default to the
    # sequential scheduler. otherwise let the _get method choose the scheduler
    if (
        scheduler is None
        and not config.get("scheduler", None)
        and single_node
        and single_file
    ):
        scheduler = "single-threaded"
    # handle lock default based on whether we're writing to a single entity
    _actual_get = get_scheduler(collections=[df], scheduler=scheduler)
    if lock is None:
        if not single_node:
            lock = True
        elif not single_file and _actual_get is not multiprocessing.get:
            # if we're writing to multiple files with the multiprocessing
            # scheduler we don't need to lock
            lock = True
        else:
            lock = False
    if lock:
        lock = get_scheduler_lock(df, scheduler=scheduler)
    kwargs.update({"format": "table", "mode": mode, "append": append})
    dsk = dict()
    i_name = name_function(0)
    dsk[(name, 0)] = (
        _pd_to_hdf,
        pd_to_hdf,
        lock,
        [(df._name, 0), fmt_obj(path, i_name), key.replace("*", i_name)],
        kwargs,
    )
    kwargs2 = kwargs.copy()
    if single_file:
        kwargs2["mode"] = "a"
    if single_node:
        kwargs2["append"] = True
    filenames = []
    for i in range(0, df.npartitions):
        i_name = name_function(i)
        filenames.append(fmt_obj(path, i_name))
    for i in range(1, df.npartitions):
        i_name = name_function(i)
        task = (
            _pd_to_hdf,
            pd_to_hdf,
            lock,
            [(df._name, i), fmt_obj(path, i_name), key.replace("*", i_name)],
            kwargs2,
        )
        if single_file:
            link_dep = i - 1 if single_node else 0
            task = (_link, (name, link_dep), task)
        dsk[(name, i)] = task
    dsk = merge(df.dask, dsk)
    if single_file and single_node:
        keys = [(name, df.npartitions - 1)]
    else:
        keys = [(name, i) for i in range(df.npartitions)]
    if compute:
        compute_as_if_collection(
            DataFrame, dsk, keys, scheduler=scheduler, **dask_kwargs
        )
        return filenames
    else:
        return delayed([Delayed(k, dsk) for k in keys])
dont_use_fixed_error_message = 
read_hdf_error_msg = 
def _read_single_hdf(
    path,
    key,
    start=0,
    stop=None,
    columns=None,
    chunksize=int(1e6),
    sorted_index=False,
    lock=None,
    mode="a",
):
    
    def get_keys_stops_divisions(path, key, stop, sorted_index, chunksize):
        
        with pd.HDFStore(path, mode=mode) as hdf:
            keys = [k for k in hdf.keys() if fnmatch(k, key)]
            stops = []
            divisions = []
            for k in keys:
                storer = hdf.get_storer(k)
                if storer.format_type != "table":
                    raise TypeError(dont_use_fixed_error_message)
                if stop is None:
                    stops.append(storer.nrows)
                elif stop > storer.nrows:
                    raise ValueError(
                        "Stop keyword exceeds dataset number "
                        "of rows ({})".format(storer.nrows)
                    )
                else:
                    stops.append(stop)
                if sorted_index:
                    division = [
                        storer.read_column("index", start=start, stop=start + 1)[0]
                        for start in range(0, storer.nrows, chunksize)
                    ]
                    division_end = storer.read_column(
                        "index", start=storer.nrows - 1, stop=storer.nrows
                    )[0]
                    division.append(division_end)
                    divisions.append(division)
                else:
                    divisions.append(None)
        return keys, stops, divisions
    def one_path_one_key(path, key, start, stop, columns, chunksize, division, lock):
        
        empty = pd.read_hdf(path, key, mode=mode, stop=0)
        if columns is not None:
            empty = empty[columns]
        token = tokenize(
            (path, os.path.getmtime(path), key, start, stop, empty, chunksize, division)
        )
        name = "read-hdf-" + token
        if empty.ndim == 1:
            base = {"name": empty.name, "mode": mode}
        else:
            base = {"columns": empty.columns, "mode": mode}
        if start >= stop:
            raise ValueError(
                "Start row number ({}) is above or equal to stop "
                "row number ({})".format(start, stop)
            )
        def update(s):
            new = base.copy()
            new.update({"start": s, "stop": s + chunksize})
            return new
        dsk = dict(
            ((name, i), (_pd_read_hdf, path, key, lock, update(s)))
            for i, s in enumerate(range(start, stop, chunksize))
        )
        if division:
            divisions = division
        else:
            divisions = [None] * (len(dsk) + 1)
        return new_dd_object(dsk, name, empty, divisions)
    keys, stops, divisions = get_keys_stops_divisions(
        path, key, stop, sorted_index, chunksize
    )
    if (start != 0 or stop is not None) and len(keys) > 1:
        raise NotImplementedError(read_hdf_error_msg)
    from ..multi import concat
    return concat(
        [
            one_path_one_key(path, k, start, s, columns, chunksize, d, lock)
            for k, s, d in zip(keys, stops, divisions)
        ]
    )
def _pd_read_hdf(path, key, lock, kwargs):
    
    if lock:
        lock.acquire()
    try:
        result = pd.read_hdf(path, key, **kwargs)
    finally:
        if lock:
            lock.release()
    return result
def read_hdf(
    pattern,
    key,
    start=0,
    stop=None,
    columns=None,
    chunksize=1000000,
    sorted_index=False,
    lock=True,
    mode="a",
):
    
    if lock is True:
        lock = get_scheduler_lock()
    key = key if key.startswith("/") else "/" + key
    # Convert path-like objects to a string
    pattern = stringify_path(pattern)
    if isinstance(pattern, str):
        paths = sorted(glob(pattern))
    else:
        paths = pattern
    if (start != 0 or stop is not None) and len(paths) > 1:
        raise NotImplementedError(read_hdf_error_msg)
    if chunksize <= 0:
        raise ValueError("Chunksize must be a positive integer")
    if (start != 0 or stop is not None) and sorted_index:
        raise ValueError(
            "When assuming pre-partitioned data, data must be "
            "read in its entirety using the same chunksizes"
        )
    from ..multi import concat
    return concat(
        [
            _read_single_hdf(
                path,
                key,
                start=start,
                stop=stop,
                columns=columns,
                chunksize=chunksize,
                sorted_index=sorted_index,
                lock=lock,
                mode=mode,
            )
            for path in paths
        ]
    )
from ..core import _Frame
_Frame.to_hdf.__doc__ = to_hdf.__doc__
