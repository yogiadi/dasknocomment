import io
import os
from toolz import concat
from ..utils import system_encoding, parse_bytes
from ..delayed import delayed
from ..bytes import open_files, read_bytes
from .core import from_delayed
delayed = delayed(pure=True)
def read_text(
    urlpath,
    blocksize=None,
    compression="infer",
    encoding=system_encoding,
    errors="strict",
    linedelimiter=os.linesep,
    collection=True,
    storage_options=None,
    files_per_partition=None,
):
    
    if blocksize is not None and files_per_partition is not None:
        raise ValueError("Only one of blocksize or files_per_partition can be set")
    if isinstance(blocksize, str):
        blocksize = parse_bytes(blocksize)
    files = open_files(
        urlpath,
        mode="rt",
        encoding=encoding,
        errors=errors,
        compression=compression,
        **(storage_options or {})
    )
    if blocksize is None:
        if files_per_partition is None:
            blocks = [delayed(list)(delayed(file_to_blocks)(fil)) for fil in files]
        else:
            blocks = []
            for start in range(0, len(files), files_per_partition):
                block_files = files[start : (start + files_per_partition)]
                block_lines = delayed(concat)(delayed(map)(file_to_blocks, block_files))
                blocks.append(block_lines)
    else:
        _, blocks = read_bytes(
            urlpath,
            delimiter=linedelimiter.encode(),
            blocksize=blocksize,
            sample=False,
            compression=compression,
            **(storage_options or {})
        )
        blocks = [delayed(decode)(b, encoding, errors) for b in concat(blocks)]
    if not blocks:
        raise ValueError("No files found", urlpath)
    if not collection:
        return blocks
    else:
        return from_delayed(blocks)
def file_to_blocks(lazy_file):
    with lazy_file as f:
        for line in f:
            yield line
def decode(block, encoding, errors):
    text = block.decode(encoding, errors)
    lines = io.StringIO(text)
    return list(lines)
