from collections import defaultdict
import pandas as pd
from toolz import partition_all
from numbers import Integral
from ..base import tokenize, compute_as_if_collection
from .accessor import Accessor
from .utils import (
    has_known_categories,
    clear_known_categories,
    is_scalar,
    is_categorical_dtype,
)
def _categorize_block(df, categories, index):
    
    df = df.copy()
    for col, vals in categories.items():
        if is_categorical_dtype(df[col]):
            df[col] = df[col].cat.set_categories(vals)
        else:
            df[col] = pd.Categorical(df[col], categories=vals, ordered=False)
    if index is not None:
        if is_categorical_dtype(df.index):
            ind = df.index.set_categories(index)
        else:
            ind = pd.Categorical(df.index, categories=index, ordered=False)
        ind.name = df.index.name
        df.index = ind
    return df
def _get_categories(df, columns, index):
    res = {}
    for col in columns:
        x = df[col]
        if is_categorical_dtype(x):
            res[col] = pd.Series(x.cat.categories)
        else:
            res[col] = x.dropna().drop_duplicates()
    if index:
        if is_categorical_dtype(df.index):
            return res, df.index.categories
        return res, df.index.dropna().drop_duplicates()
    return res, None
def _get_categories_agg(parts):
    res = defaultdict(list)
    res_ind = []
    for p in parts:
        for k, v in p[0].items():
            res[k].append(v)
        res_ind.append(p[1])
    res = {k: pd.concat(v, ignore_index=True).drop_duplicates() for k, v in res.items()}
    if res_ind[0] is None:
        return res, None
    return res, res_ind[0].append(res_ind[1:]).drop_duplicates()
def categorize(df, columns=None, index=None, split_every=None, **kwargs):
    
    meta = df._meta
    if columns is None:
        columns = list(meta.select_dtypes(["object", "category"]).columns)
    elif is_scalar(columns):
        columns = [columns]
    # Filter out known categorical columns
    columns = [
        c
        for c in columns
        if not (is_categorical_dtype(meta[c]) and has_known_categories(meta[c]))
    ]
    if index is not False:
        if is_categorical_dtype(meta.index):
            index = not has_known_categories(meta.index)
        elif index is None:
            index = meta.index.dtype == object
    # Nothing to do
    if not len(columns) and index is False:
        return df
    if split_every is None:
        split_every = 16
    elif split_every is False:
        split_every = df.npartitions
    elif not isinstance(split_every, Integral) or split_every < 2:
        raise ValueError("split_every must be an integer >= 2")
    token = tokenize(df, columns, index, split_every)
    a = "get-categories-chunk-" + token
    dsk = {
        (a, i): (_get_categories, key, columns, index)
        for (i, key) in enumerate(df.__dask_keys__())
    }
    prefix = "get-categories-agg-" + token
    k = df.npartitions
    depth = 0
    while k > split_every:
        b = prefix + str(depth)
        for part_i, inds in enumerate(partition_all(split_every, range(k))):
            dsk[(b, part_i)] = (_get_categories_agg, [(a, i) for i in inds])
        k = part_i + 1
        a = b
        depth += 1
    dsk[(prefix, 0)] = (_get_categories_agg, [(a, i) for i in range(k)])
    dsk.update(df.dask)
    # Compute the categories
    categories, index = compute_as_if_collection(type(df), dsk, (prefix, 0), **kwargs)
    # Categorize each partition
    return df.map_partitions(_categorize_block, categories, index)
class CategoricalAccessor(Accessor):
    
    _accessor_name = "cat"
    @property
    def known(self):
        
        return has_known_categories(self._series)
    def as_known(self, **kwargs):
        
        if self.known:
            return self._series
        categories = self._property_map("categories").unique().compute(**kwargs)
        return self.set_categories(categories.values)
    def as_unknown(self):
        
        if not self.known:
            return self._series
        out = self._series.copy()
        out._meta = clear_known_categories(out._meta)
        return out
    @property
    def ordered(self):
        return self._delegate_property(self._series._meta, "cat", "ordered")
    @property
    def categories(self):
        
        if not self.known:
            msg = (
                "`df.column.cat.categories` with unknown categories is not "
                "supported.  Please use `column.cat.as_known()` or "
                "`df.categorize()` beforehand to ensure known categories"
            )
            raise NotImplementedError(msg)
        return self._delegate_property(self._series._meta, "cat", "categories")
    @property
    def codes(self):
        
        if not self.known:
            msg = (
                "`df.column.cat.codes` with unknown categories is not "
                "supported.  Please use `column.cat.as_known()` or "
                "`df.categorize()` beforehand to ensure known categories"
            )
            raise NotImplementedError(msg)
        return self._property_map("codes")
    def remove_unused_categories(self):
        
        # get the set of used categories
        present = self._series.dropna().unique()
        present = pd.Index(present.compute())
        if isinstance(self._series._meta, pd.CategoricalIndex):
            meta_cat = self._series._meta
        else:
            meta_cat = self._series._meta.cat
        # Reorder to keep cat:code relationship, filtering unused (-1)
        ordered, mask = present.reindex(meta_cat.categories)
        if mask is None:
            # PANDAS-23963: old and new categories match.
            return self._series
        new_categories = ordered[mask != -1]
        meta = meta_cat.set_categories(new_categories, ordered=meta_cat.ordered)
        return self._series.map_partitions(
            self._delegate_method,
            "cat",
            "set_categories",
            (),
            {"new_categories": new_categories},
            meta=meta,
            token="cat-set_categories",
        )
