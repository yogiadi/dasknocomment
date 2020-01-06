import numpy as np
import pandas as pd
from ... import delayed
from .io import from_delayed, from_pandas
def read_sql_table(
    table,
    uri,
    index_col,
    divisions=None,
    npartitions=None,
    limits=None,
    columns=None,
    bytes_per_chunk=256 * 2 ** 20,
    head_rows=5,
    schema=None,
    meta=None,
    engine_kwargs=None,
    **kwargs
):
    
    import sqlalchemy as sa
    from sqlalchemy import sql
    from sqlalchemy.sql import elements
    if index_col is None:
        raise ValueError("Must specify index column to partition on")
    engine_kwargs = {} if engine_kwargs is None else engine_kwargs
    engine = sa.create_engine(uri, **engine_kwargs)
    m = sa.MetaData()
    if isinstance(table, str):
        table = sa.Table(table, m, autoload=True, autoload_with=engine, schema=schema)
    index = table.columns[index_col] if isinstance(index_col, str) else index_col
    if not isinstance(index_col, (str, elements.Label)):
        raise ValueError(
            "Use label when passing an SQLAlchemy instance as the index (%s)" % index
        )
    if divisions and npartitions:
        raise TypeError("Must supply either divisions or npartitions, not both")
    columns = (
        [(table.columns[c] if isinstance(c, str) else c) for c in columns]
        if columns
        else list(table.columns)
    )
    if index_col not in columns:
        columns.append(
            table.columns[index_col] if isinstance(index_col, str) else index_col
        )
    if isinstance(index_col, str):
        kwargs["index_col"] = index_col
    else:
        # function names get pandas auto-named
        kwargs["index_col"] = index_col.name
    if meta is None:
        # derive metadata from first few rows
        q = sql.select(columns).limit(head_rows).select_from(table)
        head = pd.read_sql(q, engine, **kwargs)
        if head.empty:
            # no results at all
            name = table.name
            schema = table.schema
            head = pd.read_sql_table(name, uri, schema=schema, index_col=index_col)
            return from_pandas(head, npartitions=1)
        bytes_per_row = (head.memory_usage(deep=True, index=True)).sum() / head_rows
        meta = head.iloc[:0]
    else:
        if divisions is None and npartitions is None:
            raise ValueError(
                "Must provide divisions or npartitions when using explicit meta."
            )
    if divisions is None:
        if limits is None:
            # calculate max and min for given index
            q = sql.select([sql.func.max(index), sql.func.min(index)]).select_from(
                table
            )
            minmax = pd.read_sql(q, engine)
            maxi, mini = minmax.iloc[0]
            dtype = minmax.dtypes["max_1"]
        else:
            mini, maxi = limits
            dtype = pd.Series(limits).dtype
        if npartitions is None:
            q = sql.select([sql.func.count(index)]).select_from(table)
            count = pd.read_sql(q, engine)["count_1"][0]
            npartitions = int(round(count * bytes_per_row / bytes_per_chunk)) or 1
        if dtype.kind == "M":
            divisions = pd.date_range(
                start=mini,
                end=maxi,
                freq="%iS" % ((maxi - mini).total_seconds() / npartitions),
            ).tolist()
            divisions[0] = mini
            divisions[-1] = maxi
        elif dtype.kind in ["i", "u", "f"]:
            divisions = np.linspace(mini, maxi, npartitions + 1).tolist()
        else:
            raise TypeError(
                'Provided index column is of type "{}".  If divisions is not provided the '
                "index column type must be numeric or datetime.".format(dtype)
            )
    parts = []
    lowers, uppers = divisions[:-1], divisions[1:]
    for i, (lower, upper) in enumerate(zip(lowers, uppers)):
        cond = index <= upper if i == len(lowers) - 1 else index < upper
        q = sql.select(columns).where(sql.and_(index >= lower, cond)).select_from(table)
        parts.append(
            delayed(_read_sql_chunk)(
                q, uri, meta, engine_kwargs=engine_kwargs, **kwargs
            )
        )
    engine.dispose()
    return from_delayed(parts, meta, divisions=divisions)
def _read_sql_chunk(q, uri, meta, engine_kwargs=None, **kwargs):
    import sqlalchemy as sa
    engine_kwargs = engine_kwargs or {}
    engine = sa.create_engine(uri, **engine_kwargs)
    df = pd.read_sql(q, engine, **kwargs)
    engine.dispose()
    if df.empty:
        return meta
    else:
        return df.astype(meta.dtypes.to_dict(), copy=False)
