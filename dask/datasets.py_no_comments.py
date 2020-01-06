import random
from .utils import import_required
def timeseries(
    start="2000-01-01",
    end="2000-01-31",
    freq="1s",
    partition_freq="1d",
    dtypes={"name": str, "id": int, "x": float, "y": float},
    seed=None,
    **kwargs
):
    
    from dask.dataframe.io.demo import make_timeseries
    return make_timeseries(
        start=start,
        end=end,
        freq=freq,
        partition_freq=partition_freq,
        seed=seed,
        dtypes=dtypes,
        **kwargs
    )
def _generate_mimesis(field, schema_description, records_per_partition, seed):
    
    from mimesis.schema import Schema, Field
    field = Field(seed=seed, **field)
    schema = Schema(schema=lambda: schema_description(field))
    for i in range(records_per_partition):
        yield schema.create(iterations=1)[0]
def _make_mimesis(field, schema, npartitions, records_per_partition, seed=None):
    
    import dask.bag as db
    from dask.base import tokenize
    field = field or {}
    random_state = random.Random(seed)
    seeds = [random_state.randint(0, 1 << 32) for _ in range(npartitions)]
    name = "mimesis-" + tokenize(
        field, schema, npartitions, records_per_partition, seed
    )
    dsk = {
        (name, i): (_generate_mimesis, field, schema, records_per_partition, seed)
        for i, seed in enumerate(seeds)
    }
    return db.Bag(dsk, name, npartitions)
def make_people(npartitions=10, records_per_partition=1000, seed=None, locale="en"):
    
    import_required(
        "mimesis",
        "The mimesis module is required for this function.  Try:\n"
        "  pip install mimesis",
    )
    schema = lambda field: {
        "age": field("person.age"),
        "name": (field("person.name"), field("person.surname")),
        "occupation": field("person.occupation"),
        "telephone": field("person.telephone"),
        "address": {"address": field("address.address"), "city": field("address.city")},
        "credit-card": {
            "number": field("payment.credit_card_number"),
            "expiration-date": field("payment.credit_card_expiration_date"),
        },
    }
    return _make_mimesis(
        {"locale": locale}, schema, npartitions, records_per_partition, seed
    )
