import pandas as pd
from textwrap import dedent
import dask.dataframe as dd
import dask.array as da
import numpy as np
style = 
def test_repr():
    df = pd.DataFrame({"x": list(range(100))})
    ddf = dd.from_pandas(df, 3)
    for x in [ddf, ddf.index, ddf.x]:
        assert type(x).__name__ in repr(x)
        assert str(x.npartitions) in repr(x)
def test_repr_meta_mutation():
    # Check that the repr changes when meta changes
    df = pd.DataFrame({"a": range(5), "b": ["a", "b", "c", "d", "e"]})
    ddf = dd.from_pandas(df, npartitions=2)
    s1 = repr(ddf)
    assert repr(ddf) == s1
    ddf.b = ddf.b.astype("category")
    assert repr(ddf) != s1
def test_dataframe_format():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8],
            "B": list("ABCDEFGH"),
            "C": pd.Categorical(list("AAABBBCC")),
        }
    )
    ddf = dd.from_pandas(df, 3)
    exp = (
        "Dask DataFrame Structure:\n"
        "                   A       B                C\n"
        "npartitions=3                                \n"
        "0              int64  object  category[known]\n"
        "3                ...     ...              ...\n"
        "6                ...     ...              ...\n"
        "7                ...     ...              ...\n"
        "Dask Name: from_pandas, 3 tasks"
    )
    assert repr(ddf) == exp
    assert str(ddf) == exp
    exp = (
        "                   A       B                C\n"
        "npartitions=3                                \n"
        "0              int64  object  category[known]\n"
        "3                ...     ...              ...\n"
        "6                ...     ...              ...\n"
        "7                ...     ...              ..."
    )
    assert ddf.to_string() == exp
    exp_table = 
    exp = 
.format(
        exp_table=exp_table
    )
    assert ddf.to_html() == exp
    # table is boxed with div and has style
    exp = 
.format(
        style=style, exp_table=exp_table
    )
    assert ddf._repr_html_() == exp
def test_dataframe_format_with_index():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8],
            "B": list("ABCDEFGH"),
            "C": pd.Categorical(list("AAABBBCC")),
        },
        index=list("ABCDEFGH"),
    )
    ddf = dd.from_pandas(df, 3)
    exp = (
        "Dask DataFrame Structure:\n"
        "                   A       B                C\n"
        "npartitions=3                                \n"
        "A              int64  object  category[known]\n"
        "D                ...     ...              ...\n"
        "G                ...     ...              ...\n"
        "H                ...     ...              ...\n"
        "Dask Name: from_pandas, 3 tasks"
    )
    assert repr(ddf) == exp
    assert str(ddf) == exp
    exp_table = 
    exp = 
.format(
        exp_table=exp_table
    )
    assert ddf.to_html() == exp
    # table is boxed with div and has style
    exp = 
.format(
        style=style, exp_table=exp_table
    )
    assert ddf._repr_html_() == exp
def test_dataframe_format_unknown_divisions():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8],
            "B": list("ABCDEFGH"),
            "C": pd.Categorical(list("AAABBBCC")),
        }
    )
    ddf = dd.from_pandas(df, 3)
    ddf = ddf.clear_divisions()
    assert not ddf.known_divisions
    exp = (
        "Dask DataFrame Structure:\n"
        "                   A       B                C\n"
        "npartitions=3                                \n"
        "               int64  object  category[known]\n"
        "                 ...     ...              ...\n"
        "                 ...     ...              ...\n"
        "                 ...     ...              ...\n"
        "Dask Name: from_pandas, 3 tasks"
    )
    assert repr(ddf) == exp
    assert str(ddf) == exp
    exp = (
        "                   A       B                C\n"
        "npartitions=3                                \n"
        "               int64  object  category[known]\n"
        "                 ...     ...              ...\n"
        "                 ...     ...              ...\n"
        "                 ...     ...              ..."
    )
    assert ddf.to_string() == exp
    exp_table = 
    exp = 
.format(
        exp_table=exp_table
    )
    assert ddf.to_html() == exp
    # table is boxed with div and has style
    exp = 
.format(
        style=style, exp_table=exp_table
    )
    assert ddf._repr_html_() == exp
def test_dataframe_format_long():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8] * 10,
            "B": list("ABCDEFGH") * 10,
            "C": pd.Categorical(list("AAABBBCC") * 10),
        }
    )
    ddf = dd.from_pandas(df, 10)
    exp = (
        "Dask DataFrame Structure:\n"
        "                    A       B                C\n"
        "npartitions=10                                \n"
        "0               int64  object  category[known]\n"
        "8                 ...     ...              ...\n"
        "...               ...     ...              ...\n"
        "72                ...     ...              ...\n"
        "79                ...     ...              ...\n"
        "Dask Name: from_pandas, 10 tasks"
    )
    assert repr(ddf) == exp
    assert str(ddf) == exp
    exp = (
        "                    A       B                C\n"
        "npartitions=10                                \n"
        "0               int64  object  category[known]\n"
        "8                 ...     ...              ...\n"
        "...               ...     ...              ...\n"
        "72                ...     ...              ...\n"
        "79                ...     ...              ..."
    )
    assert ddf.to_string() == exp
    exp_table = 
    exp = 
.format(
        exp_table=exp_table
    )
    assert ddf.to_html() == exp
    # table is boxed with div
    exp = u
.format(
        style=style, exp_table=exp_table
    )
    assert ddf._repr_html_() == exp
def test_series_format():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=list("ABCDEFGH"))
    ds = dd.from_pandas(s, 3)
    exp = 
    assert repr(ds) == exp
    assert str(ds) == exp
    exp = 
    assert ds.to_string() == exp
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=list("ABCDEFGH"), name="XXX")
    ds = dd.from_pandas(s, 3)
    exp = 
    assert repr(ds) == exp
    assert str(ds) == exp
def test_series_format_long():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10, index=list("ABCDEFGHIJ") * 10)
    ds = dd.from_pandas(s, 10)
    exp = (
        "Dask Series Structure:\nnpartitions=10\nA    int64\nB      ...\n"
        "     ...  \nJ      ...\nJ      ...\ndtype: int64\n"
        "Dask Name: from_pandas, 10 tasks"
    )
    assert repr(ds) == exp
    assert str(ds) == exp
    exp = "npartitions=10\nA    int64\nB      ...\n     ...  \nJ      ...\nJ      ..."
    assert ds.to_string() == exp
def test_index_format():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=list("ABCDEFGH"))
    ds = dd.from_pandas(s, 3)
    exp = 
    assert repr(ds.index) == exp
    assert str(ds.index) == exp
    s = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8],
        index=pd.CategoricalIndex([1, 2, 3, 4, 5, 6, 7, 8], name="YYY"),
    )
    ds = dd.from_pandas(s, 3)
    exp = dedent(
        
    )
    assert repr(ds.index) == exp
    assert str(ds.index) == exp
def test_categorical_format():
    s = pd.Series(["a", "b", "c"]).astype("category")
    known = dd.from_pandas(s, npartitions=1)
    unknown = known.cat.as_unknown()
    exp = (
        "Dask Series Structure:\n"
        "npartitions=1\n"
        "0    category[known]\n"
        "2                ...\n"
        "dtype: category\n"
        "Dask Name: from_pandas, 1 tasks"
    )
    assert repr(known) == exp
    exp = (
        "Dask Series Structure:\n"
        "npartitions=1\n"
        "0    category[unknown]\n"
        "2                  ...\n"
        "dtype: category\n"
        "Dask Name: from_pandas, 1 tasks"
    )
    assert repr(unknown) == exp
def test_duplicate_columns_repr():
    arr = da.from_array(np.arange(10).reshape(5, 2), chunks=(5, 2))
    frame = dd.from_dask_array(arr, columns=["a", "a"])
    repr(frame)
