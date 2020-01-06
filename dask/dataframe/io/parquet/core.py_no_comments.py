from distutils.version import LooseVersion
import toolz
import warnings
from ....bytes import core  # noqa
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path
from ...core import DataFrame, new_dd_object
from ....base import tokenize
from ....utils import import_required, natural_sort_key, parse_bytes
from collections.abc import Mapping
from ...methods import concat
try:
    import snappy
    snappy.compress
except (ImportError, AttributeError):
    snappy = None
__all__ = ("read_parquet", "to_parquet")
# ----------------------------------------------------------------------
# User API
class ParquetSubgraph(Mapping):
    
    def __init__(self, name, engine, fs, meta, columns, index, parts, kwargs):
        self.name = name
        self.engine = engine
        self.fs = fs
        self.meta = meta
        self.columns = columns
        self.index = index
        self.parts = parts
        self.kwargs = kwargs
    def __repr__(self):
        return "ParquetSubgraph<name='{}', n_parts={}, columns={}>".format(
            self.name, len(self.parts), list(self.columns)
        )
    def __getitem__(self, key):
        try:
            name, i = key
        except ValueError:
            # too many / few values to unpack
            raise KeyError(key) from None
        if name != self.name:
            raise KeyError(key)
        if i < 0 or i >= len(self.parts):
            raise KeyError(key)
        part = self.parts[i]
        if not isinstance(part, list):
            part = [part]
        return (
            read_parquet_part,
            self.engine.read_partition,
            self.fs,
            self.meta,
            [p["piece"] for p in part],
            self.columns,
            self.index,
            toolz.merge(part[0]["kwargs"], self.kwargs or {}),
        )
    def __len__(self):
        return len(self.parts)
    def __iter__(self):
        for i in range(len(self)):
            yield (self.name, i)
def read_parquet(
    path,
    columns=None,
    filters=None,
    categories=None,
    index=None,
    storage_options=None,
    engine="auto",
    gather_statistics=None,
    split_row_groups=True,
    chunksize=None,
    **kwargs
):
    
    if isinstance(columns, str):
        df = read_parquet(
            path,
            [columns],
            filters,
            categories,
            index,
            storage_options,
            engine,
            gather_statistics,
        )
        return df[columns]
    if columns is not None:
        columns = list(columns)
    name = "read-parquet-" + tokenize(
        path,
        columns,
        filters,
        categories,
        index,
        storage_options,
        engine,
        gather_statistics,
    )
    if isinstance(engine, str):
        engine = get_engine(engine)
    if hasattr(path, "name"):
        path = stringify_path(path)
    fs, _, paths = get_fs_token_paths(path, mode="rb", storage_options=storage_options)
    paths = sorted(paths, key=natural_sort_key)  # numeric rather than glob ordering
    auto_index_allowed = False
    if index is None:
        # User is allowing auto-detected index
        auto_index_allowed = True
    if index and isinstance(index, str):
        index = [index]
    meta, statistics, parts = engine.read_metadata(
        fs,
        paths,
        categories=categories,
        index=index,
        gather_statistics=gather_statistics,
        filters=filters,
        split_row_groups=split_row_groups,
        **kwargs
    )
    if meta.index.name is not None:
        index = meta.index.name
    # Parse dataset statistics from metadata (if available)
    parts, divisions, index, index_in_columns = process_statistics(
        parts, statistics, filters, index, chunksize
    )
    # Account for index and columns arguments.
    # Modify `meta` dataframe accordingly
    meta, index, columns = set_index_columns(
        meta, index, columns, index_in_columns, auto_index_allowed
    )
    subgraph = ParquetSubgraph(name, engine, fs, meta, columns, index, parts, kwargs)
    # Set the index that was previously treated as a column
    if index_in_columns:
        meta = meta.set_index(index)
    if len(divisions) < 2:
        # empty dataframe - just use meta
        subgraph = {(name, 0): meta}
        divisions = (None, None)
    return new_dd_object(subgraph, name, meta, divisions)
def read_parquet_part(func, fs, meta, part, columns, index, kwargs):
    
    if isinstance(part, list):
        dfs = [func(fs, rg, columns.copy(), index, **kwargs) for rg in part]
        df = concat(dfs, axis=0)
    else:
        df = func(fs, part, columns, index, **kwargs)
    if meta.columns.name:
        df.columns.name = meta.columns.name
    columns = columns or []
    index = index or []
    return df[[c for c in columns if c not in index]]
def to_parquet(
    df,
    path,
    engine="auto",
    compression="default",
    write_index=True,
    append=False,
    ignore_divisions=False,
    partition_on=None,
    storage_options=None,
    write_metadata_file=True,
    compute=True,
    **kwargs
):
    
    from dask import delayed
    if compression == "default":
        if snappy is not None:
            compression = "snappy"
        else:
            compression = None
    partition_on = partition_on or []
    if isinstance(partition_on, str):
        partition_on = [partition_on]
    if set(partition_on) - set(df.columns):
        raise ValueError(
            "Partitioning on non-existent column. "
            "partition_on=%s ."
            "columns=%s" % (str(partition_on), str(list(df.columns)))
        )
    if isinstance(engine, str):
        engine = get_engine(engine)
    if hasattr(path, "name"):
        path = stringify_path(path)
    fs, _, _ = get_fs_token_paths(path, mode="wb", storage_options=storage_options)
    # Trim any protocol information from the path before forwarding
    path = fs._strip_protocol(path)
    # Save divisions and corresponding index name. This is necessary,
    # because we may be resetting the index to write the file
    division_info = {"divisions": df.divisions, "name": df.index.name}
    if division_info["name"] is None:
        # As of 0.24.2, pandas will rename an index with name=None
        # when df.reset_index() is called.  The default name is "index",
        # (or "level_0" if "index" is already a column name)
        division_info["name"] = "index" if "index" not in df.columns else "level_0"
    # If write_index==True (default), reset the index and record the
    # name of the original index in `index_cols` (will be `index` if None,
    # or `level_0` if `index` is already a column name).
    # `fastparquet` will use `index_cols` to specify the index column(s)
    # in the metadata.  `pyarrow` will revert the `reset_index` call
    # below if `index_cols` is populated (because pyarrow will want to handle
    # index preservation itself).  For both engines, the column index
    # will be written to "pandas metadata" if write_index=True
    index_cols = []
    if write_index:
        real_cols = set(df.columns)
        df = df.reset_index()
        index_cols = [c for c in set(df.columns).difference(real_cols)]
    else:
        # Not writing index - might as well drop it
        df = df.reset_index(drop=True)
    _to_parquet_kwargs = {
        "engine",
        "compression",
        "write_index",
        "append",
        "ignore_divisions",
        "partition_on",
        "storage_options",
        "write_metadata_file",
        "compute",
    }
    kwargs_pass = {k: v for k, v in kwargs.items() if k not in _to_parquet_kwargs}
    # Engine-specific initialization steps to write the dataset.
    # Possibly create parquet metadata, and load existing stuff if appending
    meta, i_offset = engine.initialize_write(
        df,
        fs,
        path,
        append=append,
        ignore_divisions=ignore_divisions,
        partition_on=partition_on,
        division_info=division_info,
        index_cols=index_cols,
        **kwargs_pass
    )
    # Use i_offset and df.npartitions to define file-name list
    filenames = ["part.%i.parquet" % (i + i_offset) for i in range(df.npartitions)]
    # write parts
    dwrite = delayed(engine.write_partition)
    parts = [
        dwrite(
            d,
            path,
            fs,
            filename,
            partition_on,
            write_metadata_file,
            fmd=meta,
            compression=compression,
            index_cols=index_cols,
            **kwargs_pass
        )
        for d, filename in zip(df.to_delayed(), filenames)
    ]
    # single task to complete
    out = delayed(lambda x: None)(parts)
    if write_metadata_file:
        out = delayed(engine.write_metadata)(
            parts, meta, fs, path, append=append, compression=compression
        )
    if compute:
        out = out.compute()
    return out
_ENGINES = {}
def get_engine(engine):
    
    if engine in _ENGINES:
        return _ENGINES[engine]
    if engine == "auto":
        for eng in ["fastparquet", "pyarrow"]:
            try:
                return get_engine(eng)
            except RuntimeError:
                pass
        else:
            raise RuntimeError("Please install either fastparquet or pyarrow")
    elif engine == "fastparquet":
        import_required("fastparquet", "`fastparquet` not installed")
        from .fastparquet import FastParquetEngine
        _ENGINES["fastparquet"] = eng = FastParquetEngine
        return eng
    elif engine == "pyarrow" or engine == "arrow":
        pa = import_required("pyarrow", "`pyarrow` not installed")
        from .arrow import ArrowEngine
        if LooseVersion(pa.__version__) < "0.13.1":
            raise RuntimeError("PyArrow version >= 0.13.1 required")
        _ENGINES["pyarrow"] = eng = ArrowEngine
        return eng
    else:
        raise ValueError(
            'Unsupported engine: "{0}".'.format(engine)
            + '  Valid choices include "pyarrow" and "fastparquet".'
        )
#####################
# Utility Functions #
#####################
def sorted_columns(statistics):
    
    if not statistics:
        return []
    out = []
    for i, c in enumerate(statistics[0]["columns"]):
        if not all(
            "min" in s["columns"][i] and "max" in s["columns"][i] for s in statistics
        ):
            continue
        divisions = [c["min"]]
        max = c["max"]
        success = True
        for stats in statistics[1:]:
            c = stats["columns"][i]
            if c["min"] is None:
                success = False
                break
            if c["min"] >= max:
                divisions.append(c["min"])
                max = c["max"]
            else:
                success = False
                break
        if success:
            divisions.append(max)
            assert divisions == sorted(divisions)
            out.append({"name": c["name"], "divisions": divisions})
    return out
def apply_filters(parts, statistics, filters):
    
    def apply_conjunction(parts, statistics, conjunction):
        for column, operator, value in conjunction:
            out_parts = []
            out_statistics = []
            for part, stats in zip(parts, statistics):
                if "filter" in stats and stats["filter"]:
                    continue  # Filtered by engine
                try:
                    c = toolz.groupby("name", stats["columns"])[column][0]
                    min = c["min"]
                    max = c["max"]
                except KeyError:
                    out_parts.append(part)
                    out_statistics.append(stats)
                else:
                    if (
                        operator == "=="
                        and min <= value <= max
                        or operator == "<"
                        and min < value
                        or operator == "<="
                        and min <= value
                        or operator == ">"
                        and max > value
                        or operator == ">="
                        and max >= value
                        or operator == "in"
                        and any(min <= item <= max for item in value)
                    ):
                        out_parts.append(part)
                        out_statistics.append(stats)
            parts, statistics = out_parts, out_statistics
        return parts, statistics
    conjunction, *disjunction = filters if isinstance(filters[0], list) else [filters]
    out_parts, out_statistics = apply_conjunction(parts, statistics, conjunction)
    for conjunction in disjunction:
        for part, stats in zip(*apply_conjunction(parts, statistics, conjunction)):
            if part not in out_parts:
                out_parts.append(part)
                out_statistics.append(stats)
    return out_parts, out_statistics
def process_statistics(parts, statistics, filters, index, chunksize):
    
    index_in_columns = False
    if statistics:
        result = list(
            zip(
                *[
                    (part, stats)
                    for part, stats in zip(parts, statistics)
                    if stats["num-rows"] > 0
                ]
            )
        )
        parts, statistics = result or [[], []]
        if filters:
            parts, statistics = apply_filters(parts, statistics, filters)
        # Aggregate parts/statistics if we are splitting by row-group
        if chunksize:
            parts, statistics = aggregate_row_groups(parts, statistics, chunksize)
        out = sorted_columns(statistics)
        if index and isinstance(index, str):
            index = [index]
        if index and out:
            # Only one valid column
            out = [o for o in out if o["name"] in index]
        if index is not False and len(out) == 1:
            # Use only sorted column with statistics as the index
            divisions = out[0]["divisions"]
            if index is None:
                index_in_columns = True
                index = [out[0]["name"]]
            elif index != [out[0]["name"]]:
                raise ValueError("Specified index is invalid.\nindex: {}".format(index))
        elif index is not False and len(out) > 1:
            if any(o["name"] == "index" for o in out):
                # Use sorted column named "index" as the index
                [o] = [o for o in out if o["name"] == "index"]
                divisions = o["divisions"]
                if index is None:
                    index = [o["name"]]
                    index_in_columns = True
                elif index != [o["name"]]:
                    raise ValueError(
                        "Specified index is invalid.\nindex: {}".format(index)
                    )
            else:
                # Multiple sorted columns found, cannot autodetect the index
                warnings.warn(
                    "Multiple sorted columns found %s, cannot\n "
                    "autodetect index. Will continue without an index.\n"
                    "To pick an index column, use the index= keyword; to \n"
                    "silence this warning use index=False."
                    "" % [o["name"] for o in out],
                    RuntimeWarning,
                )
                index = False
                divisions = [None] * (len(parts) + 1)
        else:
            divisions = [None] * (len(parts) + 1)
    else:
        divisions = [None] * (len(parts) + 1)
    return parts, divisions, index, index_in_columns
def set_index_columns(meta, index, columns, index_in_columns, auto_index_allowed):
    
    ignore_index_column_intersection = False
    if columns is None:
        # User didn't specify columns, so ignore any intersection
        # of auto-detected values with the index (if necessary)
        ignore_index_column_intersection = True
        columns = [c for c in meta.columns]
    if not set(columns).issubset(set(meta.columns)):
        raise ValueError(
            "The following columns were not found in the dataset %s\n"
            "The following columns were found %s"
            % (set(columns) - set(meta.columns), meta.columns)
        )
    if index:
        if isinstance(index, str):
            index = [index]
        if isinstance(columns, str):
            columns = [columns]
        if ignore_index_column_intersection:
            columns = [col for col in columns if col not in index]
        if set(index).intersection(columns):
            if auto_index_allowed:
                raise ValueError(
                    "Specified index and column arguments must not intersect"
                    " (set index=False or remove the detected index from columns).\n"
                    "index: {} | column: {}".format(index, columns)
                )
            else:
                raise ValueError(
                    "Specified index and column arguments must not intersect.\n"
                    "index: {} | column: {}".format(index, columns)
                )
        # Leaving index as a column in `meta`, because the index
        # will be reset below (in case the index was detected after
        # meta was created)
        if index_in_columns:
            meta = meta[columns + index]
        else:
            meta = meta[columns]
    else:
        meta = meta[list(columns)]
    return meta, index, columns
def aggregate_row_groups(parts, stats, chunksize):
    if not stats[0].get("file_path_0", None):
        return parts, stats
    parts_agg = []
    stats_agg = []
    chunksize = parse_bytes(chunksize)
    next_part, next_stat = [parts[0].copy()], stats[0].copy()
    for i in range(1, len(parts)):
        stat, part = stats[i], parts[i]
        if (stat["file_path_0"] == next_stat["file_path_0"]) and (
            (next_stat["total_byte_size"] + stat["total_byte_size"]) <= chunksize
        ):
            # Update part list
            next_part.append(part)
            # Update Statistics
            next_stat["total_byte_size"] += stat["total_byte_size"]
            next_stat["num-rows"] += stat["num-rows"]
            for col, col_add in zip(next_stat["columns"], stat["columns"]):
                if col["name"] != col_add["name"]:
                    raise ValueError("Columns are different!!")
                if "null_count" in col:
                    col["null_count"] += col_add["null_count"]
                if "min" in col:
                    col["min"] = min(col["min"], col_add["min"])
                if "max" in col:
                    col["max"] = max(col["max"], col_add["max"])
        else:
            parts_agg.append(next_part)
            stats_agg.append(next_stat)
            next_part, next_stat = [part.copy()], stat.copy()
    parts_agg.append(next_part)
    stats_agg.append(next_stat)
    return parts_agg, stats_agg
DataFrame.to_parquet.__doc__ = to_parquet.__doc__
