import numpy as np
import pandas as pd
from .core import Series, DataFrame, map_partitions, apply_concat_apply
from . import methods
from .utils import is_categorical_dtype, is_scalar, has_known_categories, PANDAS_VERSION
from ..utils import M
import sys
###############################################################
# Dummies
###############################################################
def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype=np.uint8,
    **kwargs
):
    
    if PANDAS_VERSION >= "0.23.0":
        # dtype added to pandas
        kwargs["dtype"] = dtype
    elif dtype != np.uint8:
        # User specified something other than the default.
        raise ValueError(
            "Your version of pandas is '{}'. "
            "The 'dtype' keyword was added in pandas "
            "0.23.0.".format(PANDAS_VERSION)
        )
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return pd.get_dummies(
            data,
            prefix=prefix,
            prefix_sep=prefix_sep,
            dummy_na=dummy_na,
            columns=columns,
            sparse=sparse,
            drop_first=drop_first,
            **kwargs
        )
    not_cat_msg = (
        "`get_dummies` with non-categorical dtypes is not "
        "supported. Please use `df.categorize()` beforehand to "
        "convert to categorical dtype."
    )
    unknown_cat_msg = (
        "`get_dummies` with unknown categories is not "
        "supported. Please use `column.cat.as_known()` or "
        "`df.categorize()` beforehand to ensure known "
        "categories"
    )
    if isinstance(data, Series):
        if not is_categorical_dtype(data):
            raise NotImplementedError(not_cat_msg)
        if not has_known_categories(data):
            raise NotImplementedError(unknown_cat_msg)
    elif isinstance(data, DataFrame):
        if columns is None:
            if (data.dtypes == "object").any():
                raise NotImplementedError(not_cat_msg)
            columns = data._meta.select_dtypes(include=["category"]).columns
        else:
            if not all(is_categorical_dtype(data[c]) for c in columns):
                raise NotImplementedError(not_cat_msg)
        if not all(has_known_categories(data[c]) for c in columns):
            raise NotImplementedError(unknown_cat_msg)
    # We explicitly create `meta` on `data._meta` (the empty version) to
    # work around https://github.com/pandas-dev/pandas/issues/21993
    package_name = data._meta.__class__.__module__.split(".")[0]
    dummies = sys.modules[package_name].get_dummies
    meta = dummies(
        data._meta,
        prefix=prefix,
        prefix_sep=prefix_sep,
        dummy_na=dummy_na,
        columns=columns,
        sparse=sparse,
        drop_first=drop_first,
        **kwargs
    )
    return map_partitions(
        dummies,
        data,
        prefix=prefix,
        prefix_sep=prefix_sep,
        dummy_na=dummy_na,
        columns=columns,
        sparse=sparse,
        drop_first=drop_first,
        meta=meta,
        **kwargs
    )
###############################################################
# Pivot table
###############################################################
def pivot_table(df, index=None, columns=None, values=None, aggfunc="mean"):
    
    if not is_scalar(index) or index is None:
        raise ValueError("'index' must be the name of an existing column")
    if not is_scalar(columns) or columns is None:
        raise ValueError("'columns' must be the name of an existing column")
    if not is_categorical_dtype(df[columns]):
        raise ValueError("'columns' must be category dtype")
    if not has_known_categories(df[columns]):
        raise ValueError(
            "'columns' must have known categories. Please use "
            "`df[columns].cat.as_known()` beforehand to ensure "
            "known categories"
        )
    if not is_scalar(values) or values is None:
        raise ValueError("'values' must be the name of an existing column")
    if not is_scalar(aggfunc) or aggfunc not in ("mean", "sum", "count"):
        raise ValueError("aggfunc must be either 'mean', 'sum' or 'count'")
    # _emulate can't work for empty data
    # the result must have CategoricalIndex columns
    new_columns = pd.CategoricalIndex(df[columns].cat.categories, name=columns)
    meta = pd.DataFrame(
        columns=new_columns, dtype=np.float64, index=pd.Index(df._meta[index])
    )
    kwargs = {"index": index, "columns": columns, "values": values}
    if aggfunc in ["sum", "mean"]:
        pv_sum = apply_concat_apply(
            [df],
            chunk=methods.pivot_sum,
            aggregate=methods.pivot_agg,
            meta=meta,
            token="pivot_table_sum",
            chunk_kwargs=kwargs,
        )
    if aggfunc in ["count", "mean"]:
        pv_count = apply_concat_apply(
            [df],
            chunk=methods.pivot_count,
            aggregate=methods.pivot_agg,
            meta=meta,
            token="pivot_table_count",
            chunk_kwargs=kwargs,
        )
    if aggfunc == "sum":
        return pv_sum
    elif aggfunc == "count":
        return pv_count
    elif aggfunc == "mean":
        return pv_sum / pv_count
    else:
        raise ValueError
###############################################################
# Melt
###############################################################
def melt(
    frame,
    id_vars=None,
    value_vars=None,
    var_name=None,
    value_name="value",
    col_level=None,
):
    
    from dask.dataframe.core import no_default
    return frame.map_partitions(
        M.melt,
        meta=no_default,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
        col_level=col_level,
        token="melt",
    )
