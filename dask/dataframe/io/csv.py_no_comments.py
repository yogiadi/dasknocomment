from collections.abc import Mapping
from io import BytesIO
from warnings import warn, catch_warnings, simplefilter
try:
    import psutil
except ImportError:
    psutil = None
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_object_dtype,
    is_datetime64_any_dtype,
    CategoricalDtype,
)
# this import checks for the importability of fsspec
from ...bytes import read_bytes, open_file, open_files
from ...delayed import delayed
from ...utils import asciitable, parse_bytes
from ..utils import clear_known_categories
from .io import from_delayed
import fsspec.implementations.local
from fsspec.compression import compr
def pandas_read_text(
    reader,
    b,
    header,
    kwargs,
    dtypes=None,
    columns=None,
    write_header=True,
    enforce=False,
    path=None,
):
    
    bio = BytesIO()
    if write_header and not b.startswith(header.rstrip()):
        bio.write(header)
    bio.write(b)
    bio.seek(0)
    df = reader(bio, **kwargs)
    if dtypes:
        coerce_dtypes(df, dtypes)
    if enforce and columns and (list(df.columns) != list(columns)):
        raise ValueError("Columns do not match", df.columns, columns)
    elif columns:
        df.columns = columns
    if path:
        colname, path, paths = path
        code = paths.index(path)
        df = df.assign(
            **{colname: pd.Categorical.from_codes(np.full(len(df), code), paths)}
        )
    return df
def coerce_dtypes(df, dtypes):
    
    bad_dtypes = []
    bad_dates = []
    errors = []
    for c in df.columns:
        if c in dtypes and df.dtypes[c] != dtypes[c]:
            actual = df.dtypes[c]
            desired = dtypes[c]
            if is_float_dtype(actual) and is_integer_dtype(desired):
                bad_dtypes.append((c, actual, desired))
            elif is_object_dtype(actual) and is_datetime64_any_dtype(desired):
                # This can only occur when parse_dates is specified, but an
                # invalid date is encountered. Pandas then silently falls back
                # to object dtype. Since `object_array.astype(datetime)` will
                # silently overflow, error here and report.
                bad_dates.append(c)
            else:
                try:
                    df[c] = df[c].astype(dtypes[c])
                except Exception as e:
                    bad_dtypes.append((c, actual, desired))
                    errors.append((c, e))
    if bad_dtypes:
        if errors:
            ex = "\n".join(
                "- %s\n  %r" % (c, e)
                for c, e in sorted(errors, key=lambda x: str(x[0]))
            )
            exceptions = (
                "The following columns also raised exceptions on "
                "conversion:\n\n%s\n\n"
            ) % ex
            extra = ""
        else:
            exceptions = ""
            # All mismatches are int->float, also suggest `assume_missing=True`
            extra = (
                "\n\nAlternatively, provide `assume_missing=True` "
                "to interpret\n"
                "all unspecified integer columns as floats."
            )
        bad_dtypes = sorted(bad_dtypes, key=lambda x: str(x[0]))
        table = asciitable(["Column", "Found", "Expected"], bad_dtypes)
        dtype_kw = "dtype={%s}" % ",\n       ".join(
            "%r: '%s'" % (k, v) for (k, v, _) in bad_dtypes
        )
        dtype_msg = (
            "{table}\n\n"
            "{exceptions}"
            "Usually this is due to dask's dtype inference failing, and\n"
            "*may* be fixed by specifying dtypes manually by adding:\n\n"
            "{dtype_kw}\n\n"
            "to the call to `read_csv`/`read_table`."
            "{extra}"
        ).format(table=table, exceptions=exceptions, dtype_kw=dtype_kw, extra=extra)
    else:
        dtype_msg = None
    if bad_dates:
        also = " also " if bad_dtypes else " "
        cols = "\n".join("- %s" % c for c in bad_dates)
        date_msg = (
            "The following columns{also}failed to properly parse as dates:\n\n"
            "{cols}\n\n"
            "This is usually due to an invalid value in that column. To\n"
            "diagnose and fix it's recommended to drop these columns from the\n"
            "`parse_dates` keyword, and manually convert them to dates later\n"
            "using `dd.to_datetime`."
        ).format(also=also, cols=cols)
    else:
        date_msg = None
    if bad_dtypes or bad_dates:
        rule = "\n\n%s\n\n" % ("-" * 61)
        msg = "Mismatched dtypes found in `pd.read_csv`/`pd.read_table`.\n\n%s" % (
            rule.join(filter(None, [dtype_msg, date_msg]))
        )
        raise ValueError(msg)
def text_blocks_to_pandas(
    reader,
    block_lists,
    header,
    head,
    kwargs,
    collection=True,
    enforce=False,
    specified_dtypes=None,
    path=None,
):
    
    dtypes = head.dtypes.to_dict()
    # dtypes contains only instances of CategoricalDtype, which causes issues
    # in coerce_dtypes for non-uniform categories across partitions.
    # We will modify `dtype` (which is inferred) to
    # 1. contain instances of CategoricalDtypes for user-provided types
    # 2. contain 'category' for data inferred types
    categoricals = head.select_dtypes(include=["category"]).columns
    known_categoricals = []
    unknown_categoricals = categoricals
    if isinstance(specified_dtypes, Mapping):
        known_categoricals = [
            k
            for k in categoricals
            if isinstance(specified_dtypes.get(k), CategoricalDtype)
            and specified_dtypes.get(k).categories is not None
        ]
        unknown_categoricals = categoricals.difference(known_categoricals)
    elif (
        isinstance(specified_dtypes, CategoricalDtype)
        and specified_dtypes.categories is None
    ):
        known_categoricals = []
        unknown_categoricals = categoricals
    # Fixup the dtypes
    for k in unknown_categoricals:
        dtypes[k] = "category"
    columns = list(head.columns)
    delayed_pandas_read_text = delayed(pandas_read_text, pure=True)
    dfs = []
    colname, paths = path or (None, None)
    for i, blocks in enumerate(block_lists):
        if not blocks:
            continue
        if path:
            path_info = (colname, paths[i], paths)
        else:
            path_info = None
        df = delayed_pandas_read_text(
            reader,
            blocks[0],
            header,
            kwargs,
            dtypes,
            columns,
            write_header=False,
            enforce=enforce,
            path=path_info,
        )
        dfs.append(df)
        rest_kwargs = kwargs.copy()
        rest_kwargs.pop("skiprows", None)
        for b in blocks[1:]:
            dfs.append(
                delayed_pandas_read_text(
                    reader,
                    b,
                    header,
                    rest_kwargs,
                    dtypes,
                    columns,
                    enforce=enforce,
                    path=path_info,
                )
            )
    if collection:
        if path:
            head = head.assign(
                **{
                    colname: pd.Categorical.from_codes(
                        np.zeros(len(head), dtype=int), paths
                    )
                }
            )
        if len(unknown_categoricals):
            head = clear_known_categories(head, cols=unknown_categoricals)
        return from_delayed(dfs, head)
    else:
        return dfs
def auto_blocksize(total_memory, cpu_count):
    memory_factor = 10
    blocksize = int(total_memory // cpu_count / memory_factor)
    return min(blocksize, int(64e6))
# guess blocksize if psutil is installed or use acceptable default one if not
if psutil is not None:
    with catch_warnings():
        simplefilter("ignore", RuntimeWarning)
        TOTAL_MEM = psutil.virtual_memory().total
        CPU_COUNT = psutil.cpu_count()
        AUTO_BLOCKSIZE = auto_blocksize(TOTAL_MEM, CPU_COUNT)
else:
    AUTO_BLOCKSIZE = 2 ** 25
def read_pandas(
    reader,
    urlpath,
    blocksize=AUTO_BLOCKSIZE,
    collection=True,
    lineterminator=None,
    compression=None,
    sample=256000,
    enforce=False,
    assume_missing=False,
    storage_options=None,
    include_path_column=False,
    **kwargs
):
    reader_name = reader.__name__
    if lineterminator is not None and len(lineterminator) == 1:
        kwargs["lineterminator"] = lineterminator
    else:
        lineterminator = "\n"
    if include_path_column and isinstance(include_path_column, bool):
        include_path_column = "path"
    if "index" in kwargs or "index_col" in kwargs:
        raise ValueError(
            "Keywords 'index' and 'index_col' not supported. "
            "Use dd.{0}(...).set_index('my-index') "
            "instead".format(reader_name)
        )
    for kw in ["iterator", "chunksize"]:
        if kw in kwargs:
            raise ValueError("{0} not supported for dd.{1}".format(kw, reader_name))
    if kwargs.get("nrows", None):
        raise ValueError(
            "The 'nrows' keyword is not supported by "
            "`dd.{0}`. To achieve the same behavior, it's "
            "recommended to use `dd.{0}(...)."
            "head(n=nrows)`".format(reader_name)
        )
    if isinstance(kwargs.get("skiprows"), int):
        skiprows = lastskiprow = firstrow = kwargs.get("skiprows")
    elif kwargs.get("skiprows") is None:
        skiprows = lastskiprow = firstrow = 0
    else:
        # When skiprows is a list, we expect more than max(skiprows) to
        # be included in the sample. This means that [0,2] will work well,
        # but [0, 440] might not work.
        skiprows = set(kwargs.get("skiprows"))
        lastskiprow = max(skiprows)
        # find the firstrow that is not skipped, for use as header
        firstrow = min(set(range(len(skiprows) + 1)) - set(skiprows))
    if isinstance(kwargs.get("header"), list):
        raise TypeError(
            "List of header rows not supported for dd.{0}".format(reader_name)
        )
    if isinstance(kwargs.get("converters"), dict) and include_path_column:
        path_converter = kwargs.get("converters").get(include_path_column, None)
    else:
        path_converter = None
    if isinstance(blocksize, str):
        blocksize = parse_bytes(blocksize)
    if blocksize and compression:
        # NONE of the compressions should use chunking
        warn(
            "Warning %s compression does not support breaking apart files\n"
            "Please ensure that each individual file can fit in memory and\n"
            "use the keyword ``blocksize=None to remove this message``\n"
            "Setting ``blocksize=None``" % compression
        )
        blocksize = None
    if compression not in compr:
        raise NotImplementedError("Compression format %s not installed" % compression)
    if blocksize and sample and blocksize < sample and lastskiprow != 0:
        warn(
            "Unexpected behavior can result from passing skiprows when\n"
            "blocksize is smaller than sample size.\n"
            "Setting ``sample=blocksize``"
        )
        sample = blocksize
    b_lineterminator = lineterminator.encode()
    b_out = read_bytes(
        urlpath,
        delimiter=b_lineterminator,
        blocksize=blocksize,
        sample=sample,
        compression=compression,
        include_path=include_path_column,
        **(storage_options or {})
    )
    if include_path_column:
        b_sample, values, paths = b_out
        if path_converter:
            paths = [path_converter(path) for path in paths]
        path = (include_path_column, paths)
    else:
        b_sample, values = b_out
        path = None
    if not isinstance(values[0], (tuple, list)):
        values = [values]
    # If we have not sampled, then use the first row of the first values
    # as a representative sample.
    if b_sample is False and len(values[0]):
        b_sample = values[0][0].compute()
    # Get header row, and check that sample is long enough. If the file
    # contains a header row, we need at least 2 nonempty rows + the number of
    # rows to skip.
    names = kwargs.get("names", None)
    header = kwargs.get("header", "infer" if names is None else None)
    need = 1 if header is None else 2
    parts = b_sample.split(b_lineterminator, lastskiprow + need)
    # If the last partition is empty, don't count it
    nparts = 0 if not parts else len(parts) - int(not parts[-1])
    if sample is not False and nparts < lastskiprow + need and len(b_sample) >= sample:
        raise ValueError(
            "Sample is not large enough to include at least one "
            "row of data. Please increase the number of bytes "
            "in `sample` in the call to `read_csv`/`read_table`"
        )
    header = b"" if header is None else parts[firstrow] + b_lineterminator
    # Use sample to infer dtypes and check for presence of include_path_column
    head = reader(BytesIO(b_sample), **kwargs)
    if include_path_column and (include_path_column in head.columns):
        raise ValueError(
            "Files already contain the column name: %s, so the "
            "path column cannot use this name. Please set "
            "`include_path_column` to a unique name." % include_path_column
        )
    specified_dtypes = kwargs.get("dtype", {})
    if specified_dtypes is None:
        specified_dtypes = {}
    # If specified_dtypes is a single type, then all columns were specified
    if assume_missing and isinstance(specified_dtypes, dict):
        # Convert all non-specified integer columns to floats
        for c in head.columns:
            if is_integer_dtype(head[c].dtype) and c not in specified_dtypes:
                head[c] = head[c].astype(float)
    return text_blocks_to_pandas(
        reader,
        values,
        header,
        head,
        kwargs,
        collection=collection,
        enforce=enforce,
        specified_dtypes=specified_dtypes,
        path=path,
    )
READ_DOC_TEMPLATE = 
def make_reader(reader, reader_name, file_type):
    def read(
        urlpath,
        blocksize=AUTO_BLOCKSIZE,
        collection=True,
        lineterminator=None,
        compression=None,
        sample=256000,
        enforce=False,
        assume_missing=False,
        storage_options=None,
        include_path_column=False,
        **kwargs
    ):
        return read_pandas(
            reader,
            urlpath,
            blocksize=blocksize,
            collection=collection,
            lineterminator=lineterminator,
            compression=compression,
            sample=sample,
            enforce=enforce,
            assume_missing=assume_missing,
            storage_options=storage_options,
            include_path_column=include_path_column,
            **kwargs
        )
    read.__doc__ = READ_DOC_TEMPLATE.format(reader=reader_name, file_type=file_type)
    read.__name__ = reader_name
    import traceback
    traceback.print_stack()
    return read
read_csv = make_reader(pd.read_csv, "read_csv", "CSV")
read_table = make_reader(pd.read_table, "read_table", "delimited")
read_fwf = make_reader(pd.read_fwf, "read_fwf", "fixed-width")
def _write_csv(df, fil, *, depend_on=None, **kwargs):
    with fil as f:
        df.to_csv(f, **kwargs)
    return None
def to_csv(
    df,
    filename,
    single_file=False,
    encoding="utf-8",
    mode="wt",
    name_function=None,
    compression=None,
    compute=True,
    scheduler=None,
    storage_options=None,
    header_first_partition_only=None,
    **kwargs
):
    
    if single_file and name_function is not None:
        raise ValueError("name_function is not supported under the single file mode")
    if header_first_partition_only is None:
        header_first_partition_only = single_file
    elif not header_first_partition_only and single_file:
        raise ValueError(
            "header_first_partition_only cannot be False in the single file mode."
        )
    file_options = dict(
        compression=compression,
        encoding=encoding,
        newline="",
        **(storage_options or {})
    )
    to_csv_chunk = delayed(_write_csv, pure=False)
    dfs = df.to_delayed()
    if single_file:
        first_file = open_file(filename, mode=mode, **file_options)
        if not isinstance(first_file.fs, fsspec.implementations.local.LocalFileSystem):
            warn("Appending data to a network storage system may not work.")
        value = to_csv_chunk(dfs[0], first_file, **kwargs)
        append_mode = mode.replace("w", "") + "a"
        append_file = open_file(filename, mode=append_mode, **file_options)
        kwargs["header"] = False
        for d in dfs[1:]:
            value = to_csv_chunk(d, append_file, depend_on=value, **kwargs)
        values = [value]
        files = [first_file]
    else:
        files = open_files(
            filename,
            mode=mode,
            name_function=name_function,
            num=df.npartitions,
            **file_options
        )
        values = [to_csv_chunk(dfs[0], files[0], **kwargs)]
        if header_first_partition_only:
            kwargs["header"] = False
        values.extend(
            [to_csv_chunk(d, f, **kwargs) for d, f in zip(dfs[1:], files[1:])]
        )
    if compute:
        delayed(values).compute(scheduler=scheduler)
        return [f.path for f in files]
    else:
        return values
from ..core import _Frame
_Frame.to_csv.__doc__ = to_csv.__doc__
