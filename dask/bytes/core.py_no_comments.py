import copy
from fsspec.core import (  # noqa: F401
    OpenFile,  # noqa: F401
    open_files,  # noqa: F401
    get_fs_token_paths,  # noqa: F401
    expand_paths_if_needed,  # noqa: F401
    _expand_paths,  # noqa: F401
    get_compression,  # noqa: F401
)
from fsspec.core import open as open_file  # noqa: F401
from fsspec.utils import (  # noqa: F401
    read_block,  # noqa: F401
    seek_delimiter,  # noqa: F401
    infer_storage_options,  # noqa: F401
    stringify_path,  # noqa: F401
    infer_compression,  # noqa: F401
)
from fsspec import get_mapper  # noqa: F401
from fsspec.compression import compr  # noqa: F401
from ..base import tokenize
from ..delayed import delayed
from ..utils import is_integer, parse_bytes
def read_bytes(
    urlpath,
    delimiter=None,
    not_zero=False,
    blocksize="128 MiB",
    sample="10 kiB",
    compression=None,
    include_path=False,
    **kwargs
):
    
    fs, fs_token, paths = get_fs_token_paths(urlpath, mode="rb", storage_options=kwargs)
    if len(paths) == 0:
        raise IOError("%s resolved to no files" % urlpath)
    if blocksize is not None:
        if isinstance(blocksize, str):
            blocksize = parse_bytes(blocksize)
        if not is_integer(blocksize):
            raise TypeError("blocksize must be an integer")
        blocksize = int(blocksize)
    if blocksize is None:
        offsets = [[0]] * len(paths)
        lengths = [[None]] * len(paths)
    else:
        offsets = []
        lengths = []
        for path in paths:
            if compression == "infer":
                comp = infer_compression(path)
            else:
                comp = compression
            if comp is not None:
                raise ValueError(
                    "Cannot do chunked reads on compressed files. "
                    "To read, set blocksize=None"
                )
            size = fs.info(path)["size"]
            if size is None:
                raise ValueError(
                    "Backing filesystem couldn't determine file size, cannot "
                    "do chunked reads. To read, set blocksize=None."
                )
            off = list(range(0, size, blocksize))
            length = [blocksize] * len(off)
            if not_zero:
                off[0] = 1
                length[0] -= 1
            offsets.append(off)
            lengths.append(length)
    delayed_read = delayed(read_block_from_file)
    out = []
    for path, offset, length in zip(paths, offsets, lengths):
        token = tokenize(fs_token, delimiter, path, fs.ukey(path), compression, offset)
        keys = ["read-block-%s-%s" % (o, token) for o in offset]
        values = [
            delayed_read(
                OpenFile(fs, path, compression=compression),
                o,
                l,
                delimiter,
                dask_key_name=key,
            )
            for o, key, l in zip(offset, keys, length)
        ]
        out.append(values)
    if sample:
        if sample is True:
            sample = "10 kiB"  # backwards compatibility
        if isinstance(sample, str):
            sample = parse_bytes(sample)
        with OpenFile(fs, paths[0], compression=compression) as f:
            # read block without seek (because we start at zero)
            if delimiter is None:
                sample = f.read(sample)
            else:
                sample_buff = f.read(sample)
                while True:
                    new = f.read(sample)
                    if not new:
                        break
                    if delimiter in new:
                        sample_buff = (
                            sample_buff + new.split(delimiter, 1)[0] + delimiter
                        )
                        break
                    sample_buff = sample_buff + new
                sample = sample_buff
    if include_path:
        return sample, out, paths
    return sample, out
def read_block_from_file(lazy_file, off, bs, delimiter):
    with copy.copy(lazy_file) as f:
        if off == 0 and bs is None:
            return f.read()
        return read_block(f, off, bs, delimiter)
