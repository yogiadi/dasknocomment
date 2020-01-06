import io
import uuid
from ..highlevelgraph import HighLevelGraph
MAGIC = b"Obj\x01"
SYNC_SIZE = 16
def read_long(fo):
    
    c = fo.read(1)
    b = ord(c)
    n = b & 0x7F
    shift = 7
    while (b & 0x80) != 0:
        b = ord(fo.read(1))
        n |= (b & 0x7F) << shift
        shift += 7
    return (n >> 1) ^ -(n & 1)
def read_bytes(fo):
    
    size = read_long(fo)
    return fo.read(size)
def read_header(fo):
    
    assert fo.read(len(MAGIC)) == MAGIC, "Magic avro bytes missing"
    meta = {}
    out = {"meta": meta}
    while True:
        n_keys = read_long(fo)
        if n_keys == 0:
            break
        for i in range(n_keys):
            # ignore dtype mapping for bag version
            read_bytes(fo)  # schema keys
            read_bytes(fo)  # schema values
    out["sync"] = fo.read(SYNC_SIZE)
    out["header_size"] = fo.tell()
    fo.seek(0)
    out["head_bytes"] = fo.read(out["header_size"])
    return out
def open_head(fs, path, compression):
    
    from dask.bytes.core import OpenFile
    with OpenFile(fs, path, compression=compression) as f:
        head = read_header(f)
    size = fs.info(path)["size"]
    return head, size
def read_avro(urlpath, blocksize=100000000, storage_options=None, compression=None):
    
    from dask.utils import import_required
    from dask import delayed, compute
    from dask.bytes.core import open_files, get_fs_token_paths, OpenFile, tokenize
    from dask.bag import from_delayed
    import_required(
        "fastavro", "fastavro is a required dependency for using bag.read_avro()."
    )
    storage_options = storage_options or {}
    if blocksize is not None:
        fs, fs_token, paths = get_fs_token_paths(
            urlpath, mode="rb", storage_options=storage_options
        )
        dhead = delayed(open_head)
        out = compute(*[dhead(fs, path, compression) for path in paths])
        heads, sizes = zip(*out)
        dread = delayed(read_chunk)
        offsets = []
        lengths = []
        for size in sizes:
            off = list(range(0, size, blocksize))
            length = [blocksize] * len(off)
            offsets.append(off)
            lengths.append(length)
        out = []
        for path, offset, length, head in zip(paths, offsets, lengths, heads):
            delimiter = head["sync"]
            f = OpenFile(fs, path, compression=compression)
            token = tokenize(
                fs_token, delimiter, path, fs.ukey(path), compression, offset
            )
            keys = ["read-avro-%s-%s" % (o, token) for o in offset]
            values = [
                dread(f, o, l, head, dask_key_name=key)
                for o, key, l in zip(offset, keys, length)
            ]
            out.extend(values)
        return from_delayed(out)
    else:
        files = open_files(urlpath, compression=compression, **storage_options)
        dread = delayed(read_file)
        chunks = [dread(fo) for fo in files]
        return from_delayed(chunks)
def read_chunk(fobj, off, l, head):
    
    import fastavro
    from dask.bytes.core import read_block
    with fobj as f:
        chunk = read_block(f, off, l, head["sync"])
    head_bytes = head["head_bytes"]
    if not chunk.startswith(MAGIC):
        chunk = head_bytes + chunk
    i = io.BytesIO(chunk)
    return list(fastavro.iter_avro(i))
def read_file(fo):
    
    import fastavro
    with fo as f:
        return list(fastavro.iter_avro(f))
def to_avro(
    b,
    filename,
    schema,
    name_function=None,
    storage_options=None,
    codec="null",
    sync_interval=16000,
    metadata=None,
    compute=True,
    **kwargs
):
    
    # TODO infer schema from first partition of data
    from dask.utils import import_required
    from dask.bytes.core import open_files
    import_required(
        "fastavro", "fastavro is a required dependency for using bag.to_avro()."
    )
    _verify_schema(schema)
    storage_options = storage_options or {}
    files = open_files(
        filename,
        "wb",
        name_function=name_function,
        num=b.npartitions,
        **storage_options
    )
    name = "to-avro-" + uuid.uuid4().hex
    dsk = {
        (name, i): (
            _write_avro_part,
            (b.name, i),
            f,
            schema,
            codec,
            sync_interval,
            metadata,
        )
        for i, f in enumerate(files)
    }
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b])
    out = type(b)(graph, name, b.npartitions)
    if compute:
        out.compute(**kwargs)
        return [f.path for f in files]
    else:
        return out.to_delayed()
def _verify_schema(s):
    assert isinstance(s, dict), "Schema must be dictionary"
    for field in ["name", "type", "fields"]:
        assert field in s, "Schema missing '%s' field" % field
    assert s["type"] == "record", "Schema must be of type 'record'"
    assert isinstance(s["fields"], list), "Fields entry must be a list"
    for f in s["fields"]:
        assert "name" in f and "type" in f, "Field spec incomplete: %s" % f
def _write_avro_part(part, f, schema, codec, sync_interval, metadata):
    
    import fastavro
    with f as f:
        fastavro.writer(f, schema, part, codec, sync_interval, metadata)
