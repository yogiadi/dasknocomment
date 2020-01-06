import io
import pandas as pd
from dask.bytes import open_files, read_bytes
import dask
from ..utils import insert_meta_param_description, make_meta
def to_json(
    df,
    url_path,
    orient="records",
    lines=None,
    storage_options=None,
    compute=True,
    encoding="utf-8",
    errors="strict",
    compression=None,
    **kwargs
):
    
    if lines is None:
        lines = orient == "records"
    if orient != "records" and lines:
        raise ValueError(
            "Line-delimited JSON is only available with" 'orient="records".'
        )
    kwargs["orient"] = orient
    kwargs["lines"] = lines and orient == "records"
    outfiles = open_files(
        url_path,
        "wt",
        encoding=encoding,
        errors=errors,
        name_function=kwargs.pop("name_function", None),
        num=df.npartitions,
        compression=compression,
        **(storage_options or {})
    )
    parts = [
        dask.delayed(write_json_partition)(d, outfile, kwargs)
        for outfile, d in zip(outfiles, df.to_delayed())
    ]
    if compute:
        dask.compute(parts)
        return [f.path for f in outfiles]
    else:
        return parts
def write_json_partition(df, openfile, kwargs):
    with openfile as f:
        df.to_json(f, **kwargs)
@insert_meta_param_description
def read_json(
    url_path,
    orient="records",
    lines=None,
    storage_options=None,
    blocksize=None,
    sample=2 ** 20,
    encoding="utf-8",
    errors="strict",
    compression="infer",
    meta=None,
    engine=pd.read_json,
    **kwargs
):
    
    import dask.dataframe as dd
    if lines is None:
        lines = orient == "records"
    if orient != "records" and lines:
        raise ValueError(
            "Line-delimited JSON is only available with" 'orient="records".'
        )
    if blocksize and (orient != "records" or not lines):
        raise ValueError(
            "JSON file chunking only allowed for JSON-lines"
            "input (orient='records', lines=True)."
        )
    storage_options = storage_options or {}
    if blocksize:
        first, chunks = read_bytes(
            url_path,
            b"\n",
            blocksize=blocksize,
            sample=sample,
            compression=compression,
            **storage_options
        )
        chunks = list(dask.core.flatten(chunks))
        if meta is None:
            meta = read_json_chunk(first, encoding, errors, engine, kwargs)
        meta = make_meta(meta)
        parts = [
            dask.delayed(read_json_chunk)(
                chunk, encoding, errors, engine, kwargs, meta=meta
            )
            for chunk in chunks
        ]
        return dd.from_delayed(parts, meta=meta)
    else:
        files = open_files(
            url_path,
            "rt",
            encoding=encoding,
            errors=errors,
            compression=compression,
            **storage_options
        )
        parts = [
            dask.delayed(read_json_file)(f, orient, lines, engine, kwargs)
            for f in files
        ]
        return dd.from_delayed(parts, meta=meta)
def read_json_chunk(chunk, encoding, errors, engine, kwargs, meta=None):
    s = io.StringIO(chunk.decode(encoding, errors))
    s.seek(0)
    df = engine(s, orient="records", lines=True, **kwargs)
    if meta is not None and df.empty:
        return meta
    else:
        return df
def read_json_file(f, orient, lines, engine, kwargs):
    with f as f:
        return engine(f, orient=orient, lines=lines, **kwargs)
