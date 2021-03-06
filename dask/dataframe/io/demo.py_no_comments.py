import pandas as pd
import numpy as np
from ..core import tokenize, DataFrame
from .io import from_delayed
from ...delayed import delayed
from ...utils import random_state_data
__all__ = ["make_timeseries"]
def make_float(n, rstate):
    return rstate.rand(n) * 2 - 1
def make_int(n, rstate, lam=1000):
    return rstate.poisson(lam, size=n)
names = [
    "Alice",
    "Bob",
    "Charlie",
    "Dan",
    "Edith",
    "Frank",
    "George",
    "Hannah",
    "Ingrid",
    "Jerry",
    "Kevin",
    "Laura",
    "Michael",
    "Norbert",
    "Oliver",
    "Patricia",
    "Quinn",
    "Ray",
    "Sarah",
    "Tim",
    "Ursula",
    "Victor",
    "Wendy",
    "Xavier",
    "Yvonne",
    "Zelda",
]
def make_string(n, rstate):
    return rstate.choice(names, size=n)
def make_categorical(n, rstate):
    return pd.Categorical.from_codes(rstate.randint(0, len(names), size=n), names)
make = {
    float: make_float,
    int: make_int,
    str: make_string,
    object: make_string,
    "category": make_categorical,
}
def make_timeseries_part(start, end, dtypes, freq, state_data, kwargs):
    index = pd.date_range(start=start, end=end, freq=freq, name="timestamp")
    state = np.random.RandomState(state_data)
    columns = {}
    for k, dt in dtypes.items():
        kws = {
            kk.rsplit("_", 1)[1]: v
            for kk, v in kwargs.items()
            if kk.rsplit("_", 1)[0] == k
        }
        columns[k] = make[dt](len(index), state, **kws)
    df = pd.DataFrame(columns, index=index, columns=sorted(columns))
    if df.index[-1] == end:
        df = df.iloc[:-1]
    return df
def make_timeseries(
    start="2000-01-01",
    end="2000-12-31",
    dtypes={"name": str, "id": int, "x": float, "y": float},
    freq="10s",
    partition_freq="1M",
    seed=None,
    **kwargs
):
    
    divisions = list(pd.date_range(start=start, end=end, freq=partition_freq))
    state_data = random_state_data(len(divisions) - 1, seed)
    name = "make-timeseries-" + tokenize(
        start, end, dtypes, freq, partition_freq, state_data
    )
    dsk = {
        (name, i): (
            make_timeseries_part,
            divisions[i],
            divisions[i + 1],
            dtypes,
            freq,
            state_data[i],
            kwargs,
        )
        for i in range(len(divisions) - 1)
    }
    head = make_timeseries_part("2000", "2000", dtypes, "1H", state_data[0], kwargs)
    return DataFrame(dsk, name, head, divisions)
def generate_day(
    date,
    open,
    high,
    low,
    close,
    volume,
    freq=pd.Timedelta(seconds=60),
    random_state=None,
):
    
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    if not isinstance(date, pd.Timestamp):
        date = pd.Timestamp(date)
    if not isinstance(freq, pd.Timedelta):
        freq = pd.Timedelta(freq)
    time = pd.date_range(
        date + pd.Timedelta(hours=9),
        date + pd.Timedelta(hours=12 + 4),
        freq=freq / 5,
        name="timestamp",
    )
    n = len(time)
    while True:
        values = (random_state.random_sample(n) - 0.5).cumsum()
        values *= (high - low) / (values.max() - values.min())  # scale
        values += np.linspace(
            open - values[0], close - values[-1], len(values)
        )  # endpoints
        assert np.allclose(open, values[0])
        assert np.allclose(close, values[-1])
        mx = max(close, open)
        mn = min(close, open)
        ind = values > mx
        values[ind] = (values[ind] - mx) * (high - mx) / (values.max() - mx) + mx
        ind = values < mn
        values[ind] = (values[ind] - mn) * (low - mn) / (values.min() - mn) + mn
        # The process fails if min/max are the same as open close.  This is rare
        if np.allclose(values.max(), high) and np.allclose(values.min(), low):
            break
    s = pd.Series(values.round(3), index=time)
    rs = s.resample(freq)
    # TODO: add in volume
    return pd.DataFrame(
        {"open": rs.first(), "close": rs.last(), "high": rs.max(), "low": rs.min()}
    )
def daily_stock(
    symbol,
    start,
    stop,
    freq=pd.Timedelta(seconds=1),
    data_source="yahoo",
    random_state=None,
):
    
    from pandas_datareader import data
    df = data.DataReader(symbol, data_source, start, stop)
    seeds = random_state_data(len(df), random_state=random_state)
    parts = []
    divisions = []
    for i, seed in zip(range(len(df)), seeds):
        s = df.iloc[i]
        if s.isnull().any():
            continue
        part = delayed(generate_day)(
            s.name,
            s.loc["Open"],
            s.loc["High"],
            s.loc["Low"],
            s.loc["Close"],
            s.loc["Volume"],
            freq=freq,
            random_state=seed,
        )
        parts.append(part)
        divisions.append(s.name + pd.Timedelta(hours=9))
    divisions.append(s.name + pd.Timedelta(hours=12 + 4))
    meta = generate_day("2000-01-01", 1, 2, 0, 1, 100)
    return from_delayed(parts, meta=meta, divisions=divisions)
