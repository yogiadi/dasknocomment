from . import core
def _tiledb_to_chunks(tiledb_array):
    schema = tiledb_array.schema
    return list(schema.domain.dim(i).tile for i in range(schema.ndim))
def from_tiledb(uri, attribute=None, chunks=None, storage_options=None, **kwargs):
    
    import tiledb
    tiledb_config = storage_options or dict()
    key = tiledb_config.pop("key", None)
    if isinstance(uri, tiledb.Array):
        tdb = uri
    else:
        tdb = tiledb.open(uri, attr=attribute, config=tiledb_config, key=key)
    if tdb.schema.sparse:
        raise ValueError("Sparse TileDB arrays are not supported")
    if not attribute:
        if tdb.schema.nattr > 1:
            raise TypeError(
                "keyword 'attribute' must be provided"
                "when loading a multi-attribute TileDB array"
            )
        else:
            attribute = tdb.schema.attr(0).name
    if tdb.iswritable:
        raise ValueError("TileDB array must be open for reading")
    chunks = chunks or _tiledb_to_chunks(tdb)
    assert len(chunks) == tdb.schema.ndim
    return core.from_array(tdb, chunks, name="tiledb-%s" % uri)
def to_tiledb(
    darray, uri, compute=True, return_stored=False, storage_options=None, **kwargs
):
    
    import tiledb
    tiledb_config = storage_options or dict()
    # encryption key, if any
    key = tiledb_config.pop("key", None)
    if not core._check_regular_chunks(darray.chunks):
        raise ValueError(
            "Attempt to save array to TileDB with irregular "
            "chunking, please call `arr.rechunk(...)` first."
        )
    if isinstance(uri, str):
        chunks = [c[0] for c in darray.chunks]
        key = kwargs.pop("key", None)
        # create a suitable, empty, writable TileDB array
        tdb = tiledb.empty_like(
            uri, darray, tile=chunks, config=tiledb_config, key=key, **kwargs
        )
    elif isinstance(uri, tiledb.Array):
        tdb = uri
        # sanity checks
        if not ((darray.dtype == tdb.dtype) and (darray.ndim == tdb.ndim)):
            raise ValueError(
                "Target TileDB array layout is not compatible with source array"
            )
    else:
        raise ValueError(
            "'uri' must be string pointing to supported TileDB store location "
            "or an open, writable TileDB array."
        )
    if not (tdb.isopen and tdb.iswritable):
        raise ValueError("Target TileDB array is not open and writable.")
    return darray.store(tdb, lock=False, compute=compute, return_stored=return_stored)
