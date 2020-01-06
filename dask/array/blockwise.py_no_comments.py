import numbers
import warnings
import toolz
from .. import base, utils
from ..delayed import unpack_collections
from ..highlevelgraph import HighLevelGraph
from ..blockwise import blockwise as core_blockwise
def blockwise(
    func,
    out_ind,
    *args,
    name=None,
    token=None,
    dtype=None,
    adjust_chunks=None,
    new_axes=None,
    align_arrays=True,
    concatenate=None,
    meta=None,
    **kwargs
):
    
    out = name
    new_axes = new_axes or {}
    # Input Validation
    if len(set(out_ind)) != len(out_ind):
        raise ValueError(
            "Repeated elements not allowed in output index",
            [k for k, v in toolz.frequencies(out_ind).items() if v > 1],
        )
    new = (
        set(out_ind)
        - {a for arg in args[1::2] if arg is not None for a in arg}
        - set(new_axes or ())
    )
    if new:
        raise ValueError("Unknown dimension", new)
    from .core import Array, unify_chunks, normalize_arg
    if align_arrays:
        chunkss, arrays = unify_chunks(*args)
    else:
        arginds = [(a, i) for (a, i) in toolz.partition(2, args) if i is not None]
        if arginds:
            arg, ind = max(arginds, key=lambda ai: len(ai[1]))
            chunkss = dict(zip(ind, arg.chunks))
        else:
            chunkss = {}
        arrays = args[::2]
    for k, v in new_axes.items():
        if not isinstance(v, tuple):
            v = (v,)
        chunkss[k] = v
    arginds = list(zip(arrays, args[1::2]))
    for arg, ind in arginds:
        if hasattr(arg, "ndim") and hasattr(ind, "__len__") and arg.ndim != len(ind):
            raise ValueError(
                "Index string %s does not match array dimension %d" % (ind, arg.ndim)
            )
    numblocks = {a.name: a.numblocks for a, ind in arginds if ind is not None}
    dependencies = []
    arrays = []
    # Normalize arguments
    argindsstr = []
    for a, ind in arginds:
        if ind is None:
            a = normalize_arg(a)
            a, collections = unpack_collections(a)
            dependencies.extend(collections)
        else:
            arrays.append(a)
            a = a.name
        argindsstr.extend((a, ind))
    # Normalize keyword arguments
    kwargs2 = {}
    for k, v in kwargs.items():
        v = normalize_arg(v)
        v, collections = unpack_collections(v)
        dependencies.extend(collections)
        kwargs2[k] = v
    # Finish up the name
    if not out:
        out = "%s-%s" % (
            token or utils.funcname(func).strip("_"),
            base.tokenize(func, out_ind, argindsstr, dtype, **kwargs),
        )
    graph = core_blockwise(
        func,
        out,
        out_ind,
        *argindsstr,
        numblocks=numblocks,
        dependencies=dependencies,
        new_axes=new_axes,
        concatenate=concatenate,
        **kwargs2
    )
    graph = HighLevelGraph.from_collections(
        out, graph, dependencies=arrays + dependencies
    )
    chunks = [chunkss[i] for i in out_ind]
    if adjust_chunks:
        for i, ind in enumerate(out_ind):
            if ind in adjust_chunks:
                if callable(adjust_chunks[ind]):
                    chunks[i] = tuple(map(adjust_chunks[ind], chunks[i]))
                elif isinstance(adjust_chunks[ind], numbers.Integral):
                    chunks[i] = tuple(adjust_chunks[ind] for _ in chunks[i])
                elif isinstance(adjust_chunks[ind], (tuple, list)):
                    chunks[i] = tuple(adjust_chunks[ind])
                else:
                    raise NotImplementedError(
                        "adjust_chunks values must be callable, int, or tuple"
                    )
    chunks = tuple(chunks)
    if meta is None:
        from .utils import compute_meta
        meta = compute_meta(func, dtype, *args[::2], **kwargs)
    if meta is not None:
        return Array(graph, out, chunks, meta=meta)
    else:
        return Array(graph, out, chunks, dtype=dtype)
def atop(*args, **kwargs):
    warnings.warn("The da.atop function has moved to da.blockwise")
    return blockwise(*args, **kwargs)
