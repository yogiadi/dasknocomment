
import math
import heapq
from itertools import product, chain, count
from operator import getitem, add, mul, itemgetter
import numpy as np
import toolz
from toolz import accumulate, reduce
from ..base import tokenize
from ..highlevelgraph import HighLevelGraph
from ..utils import parse_bytes
from .core import concatenate3, Array, normalize_chunks
from .utils import validate_axis
from .wrap import empty
from .. import config
def cumdims_label(chunks, const):
    
    return [
        tuple(zip((const,) * (1 + len(bds)), accumulate(add, (0,) + bds)))
        for bds in chunks
    ]
def _breakpoints(cumold, cumnew):
    
    return tuple(sorted(cumold + cumnew, key=itemgetter(1)))
def _intersect_1d(breaks):
    
    start = 0
    last_end = 0
    old_idx = 0
    ret = []
    ret_next = []
    for idx in range(1, len(breaks)):
        label, br = breaks[idx]
        last_label, last_br = breaks[idx - 1]
        if last_label == "n":
            if ret_next:
                ret.append(ret_next)
                ret_next = []
        if last_label == "o":
            start = 0
        else:
            start = last_end
        end = br - last_br + start
        last_end = end
        if br == last_br:
            if label == "o":
                old_idx += 1
            continue
        ret_next.append((old_idx, slice(start, end)))
        if label == "o":
            old_idx += 1
            start = 0
    if ret_next:
        ret.append(ret_next)
    return ret
def _old_to_new(old_chunks, new_chunks):
    
    old_known = [x for x in old_chunks if not any(math.isnan(y) for y in x)]
    new_known = [x for x in new_chunks if not any(math.isnan(y) for y in x)]
    n_missing = [sum(math.isnan(y) for y in x) for x in old_chunks]
    n_missing2 = [sum(math.isnan(y) for y in x) for x in new_chunks]
    cmo = cumdims_label(old_known, "o")
    cmn = cumdims_label(new_known, "n")
    sums = [sum(o) for o in old_known]
    sums2 = [sum(n) for n in new_known]
    if not sums == sums2:
        raise ValueError("Cannot change dimensions from %r to %r" % (sums, sums2))
    if not n_missing == n_missing2:
        raise ValueError(
            "Chunks must be unchanging along unknown dimensions.\n\n"
            "A possible solution:\n  x.compute_chunk_sizes()"
        )
    old_to_new = [_intersect_1d(_breakpoints(cm[0], cm[1])) for cm in zip(cmo, cmn)]
    for idx, missing in enumerate(n_missing):
        if missing:
            # Missing dimensions are always unchanged, so old -> new is everything
            extra = [[(i, slice(0, None))] for i in range(missing)]
            old_to_new.insert(idx, extra)
    return old_to_new
def intersect_chunks(old_chunks, new_chunks):
    
    old_to_new = _old_to_new(old_chunks, new_chunks)
    cross1 = product(*old_to_new)
    cross = chain(tuple(product(*cr)) for cr in cross1)
    return cross
def rechunk(x, chunks="auto", threshold=None, block_size_limit=None):
    
    if isinstance(chunks, dict):
        chunks = {validate_axis(c, x.ndim): v for c, v in chunks.items()}
        for i in range(x.ndim):
            if i not in chunks:
                chunks[i] = x.chunks[i]
    if isinstance(chunks, (tuple, list)):
        chunks = tuple(lc if lc is not None else rc for lc, rc in zip(chunks, x.chunks))
    chunks = normalize_chunks(
        chunks, x.shape, limit=block_size_limit, dtype=x.dtype, previous_chunks=x.chunks
    )
    if chunks == x.chunks:
        return x
    ndim = x.ndim
    if not len(chunks) == ndim:
        raise ValueError("Provided chunks are not consistent with shape")
    new_shapes = tuple(map(sum, chunks))
    for new, old in zip(new_shapes, x.shape):
        if new != old and not math.isnan(old) and not math.isnan(new):
            raise ValueError("Provided chunks are not consistent with shape")
    steps = plan_rechunk(
        x.chunks, chunks, x.dtype.itemsize, threshold, block_size_limit
    )
    for c in steps:
        x = _compute_rechunk(x, c)
    return x
def _number_of_blocks(chunks):
    return reduce(mul, map(len, chunks))
def _largest_block_size(chunks):
    return reduce(mul, map(max, chunks))
def estimate_graph_size(old_chunks, new_chunks):
    
    # Estimate the number of intermediate blocks that will be produced
    # (we don't use intersect_chunks() which is much more expensive)
    crossed_size = reduce(
        mul, (len(oc) + len(nc) for oc, nc in zip(old_chunks, new_chunks))
    )
    return crossed_size
def divide_to_width(desired_chunks, max_width):
    
    chunks = []
    for c in desired_chunks:
        nb_divides = int(np.ceil(c / max_width))
        for i in range(nb_divides):
            n = c // (nb_divides - i)
            chunks.append(n)
            c -= n
        assert c == 0
    return tuple(chunks)
def merge_to_number(desired_chunks, max_number):
    
    if len(desired_chunks) <= max_number:
        return desired_chunks
    distinct = set(desired_chunks)
    if len(distinct) == 1:
        # Fast path for homogeneous target, also ensuring a regular result
        w = distinct.pop()
        n = len(desired_chunks)
        total = n * w
        desired_width = total // max_number
        width = w * (desired_width // w)
        adjust = (total - max_number * width) // w
        return (width + w,) * adjust + (width,) * (max_number - adjust)
    desired_width = sum(desired_chunks) // max_number
    nmerges = len(desired_chunks) - max_number
    heap = [
        (desired_chunks[i] + desired_chunks[i + 1], i, i + 1)
        for i in range(len(desired_chunks) - 1)
    ]
    heapq.heapify(heap)
    chunks = list(desired_chunks)
    while nmerges > 0:
        # Find smallest interval to merge
        width, i, j = heapq.heappop(heap)
        # If interval was made invalid by another merge, recompute
        # it, re-insert it and retry.
        if chunks[j] == 0:
            j += 1
            while chunks[j] == 0:
                j += 1
            heapq.heappush(heap, (chunks[i] + chunks[j], i, j))
            continue
        elif chunks[i] + chunks[j] != width:
            heapq.heappush(heap, (chunks[i] + chunks[j], i, j))
            continue
        # Merge
        assert chunks[i] != 0
        chunks[i] = 0  # mark deleted
        chunks[j] = width
        nmerges -= 1
    return tuple(filter(None, chunks))
def find_merge_rechunk(old_chunks, new_chunks, block_size_limit):
    
    ndim = len(old_chunks)
    old_largest_width = [max(c) for c in old_chunks]
    new_largest_width = [max(c) for c in new_chunks]
    graph_size_effect = {
        dim: len(nc) / len(oc)
        for dim, (oc, nc) in enumerate(zip(old_chunks, new_chunks))
    }
    block_size_effect = {
        dim: new_largest_width[dim] / (old_largest_width[dim] or 1)
        for dim in range(ndim)
    }
    # Our goal is to reduce the number of nodes in the rechunk graph
    # by merging some adjacent chunks, so consider dimensions where we can
    # reduce the # of chunks
    merge_candidates = [dim for dim in range(ndim) if graph_size_effect[dim] <= 1.0]
    # Merging along each dimension reduces the graph size by a certain factor
    # and increases memory largest block size by a certain factor.
    # We want to optimize the graph size while staying below the given
    # block_size_limit.  This is in effect a knapsack problem, except with
    # multiplicative values and weights.  Just use a greedy algorithm
    # by trying dimensions in decreasing value / weight order.
    def key(k):
        gse = graph_size_effect[k]
        bse = block_size_effect[k]
        if bse == 1:
            bse = 1 + 1e-9
        return (np.log(gse) / np.log(bse)) if bse > 0 else 0
    sorted_candidates = sorted(merge_candidates, key=key)
    largest_block_size = reduce(mul, old_largest_width)
    chunks = list(old_chunks)
    memory_limit_hit = False
    for dim in sorted_candidates:
        # Examine this dimension for possible graph reduction
        new_largest_block_size = (
            largest_block_size * new_largest_width[dim] // (old_largest_width[dim] or 1)
        )
        if new_largest_block_size <= block_size_limit:
            # Full replacement by new chunks is possible
            chunks[dim] = new_chunks[dim]
            largest_block_size = new_largest_block_size
        else:
            # Try a partial rechunk, dividing the new chunks into
            # smaller pieces
            largest_width = old_largest_width[dim]
            chunk_limit = int(block_size_limit * largest_width / largest_block_size)
            c = divide_to_width(new_chunks[dim], chunk_limit)
            if len(c) <= len(old_chunks[dim]):
                # We manage to reduce the number of blocks, so do it
                chunks[dim] = c
                largest_block_size = largest_block_size * max(c) // largest_width
            memory_limit_hit = True
    assert largest_block_size == _largest_block_size(chunks)
    assert largest_block_size <= block_size_limit
    return tuple(chunks), memory_limit_hit
def find_split_rechunk(old_chunks, new_chunks, graph_size_limit):
    
    ndim = len(old_chunks)
    chunks = list(old_chunks)
    for dim in range(ndim):
        graph_size = estimate_graph_size(chunks, new_chunks)
        if graph_size > graph_size_limit:
            break
        if len(old_chunks[dim]) > len(new_chunks[dim]):
            # It's not interesting to split
            continue
        # Merge the new chunks so as to stay within the graph size budget
        max_number = int(len(old_chunks[dim]) * graph_size_limit / graph_size)
        c = merge_to_number(new_chunks[dim], max_number)
        assert len(c) <= max_number
        # Consider the merge successful if its result has a greater length
        # and smaller max width than the old chunks
        if len(c) >= len(old_chunks[dim]) and max(c) <= max(old_chunks[dim]):
            chunks[dim] = c
    return tuple(chunks)
def plan_rechunk(
    old_chunks, new_chunks, itemsize, threshold=None, block_size_limit=None
):
    
    threshold = threshold or config.get("array.rechunk-threshold")
    block_size_limit = block_size_limit or config.get("array.chunk-size")
    if isinstance(block_size_limit, str):
        block_size_limit = parse_bytes(block_size_limit)
    ndim = len(new_chunks)
    steps = []
    has_nans = [any(math.isnan(y) for y in x) for x in old_chunks]
    if ndim <= 1 or not all(new_chunks) or any(has_nans):
        # Trivial array / unknown dim => no need / ability for an intermediate
        return steps + [new_chunks]
    # Make it a number ef elements
    block_size_limit /= itemsize
    # Fix block_size_limit if too small for either old_chunks or new_chunks
    largest_old_block = _largest_block_size(old_chunks)
    largest_new_block = _largest_block_size(new_chunks)
    block_size_limit = max([block_size_limit, largest_old_block, largest_new_block])
    # The graph size above which to optimize
    graph_size_threshold = threshold * (
        _number_of_blocks(old_chunks) + _number_of_blocks(new_chunks)
    )
    current_chunks = old_chunks
    first_pass = True
    while True:
        graph_size = estimate_graph_size(current_chunks, new_chunks)
        if graph_size < graph_size_threshold:
            break
        if first_pass:
            chunks = current_chunks
        else:
            # We hit the block_size_limit in a previous merge pass =>
            # accept a significant increase in graph size in exchange for
            # 1) getting nearer the goal 2) reducing the largest block size
            # to make place for the following merge.
            # To see this pass in action, make the block_size_limit very small.
            chunks = find_split_rechunk(
                current_chunks, new_chunks, graph_size * threshold
            )
        chunks, memory_limit_hit = find_merge_rechunk(
            chunks, new_chunks, block_size_limit
        )
        if (chunks == current_chunks and not first_pass) or chunks == new_chunks:
            break
        steps.append(chunks)
        current_chunks = chunks
        if not memory_limit_hit:
            break
        first_pass = False
    return steps + [new_chunks]
def _compute_rechunk(x, chunks):
    
    if x.size == 0:
        # Special case for empty array, as the algorithm below does not behave correctly
        return empty(x.shape, chunks=chunks, dtype=x.dtype)
    ndim = x.ndim
    crossed = intersect_chunks(x.chunks, chunks)
    x2 = dict()
    intermediates = dict()
    token = tokenize(x, chunks)
    merge_name = "rechunk-merge-" + token
    split_name = "rechunk-split-" + token
    split_name_suffixes = count()
    # Pre-allocate old block references, to allow re-use and reduce the
    # graph's memory footprint a bit.
    old_blocks = np.empty([len(c) for c in x.chunks], dtype="O")
    for index in np.ndindex(old_blocks.shape):
        old_blocks[index] = (x.name,) + index
    # Iterate over all new blocks
    new_index = product(*(range(len(c)) for c in chunks))
    for new_idx, cross1 in zip(new_index, crossed):
        key = (merge_name,) + new_idx
        old_block_indices = [[cr[i][0] for cr in cross1] for i in range(ndim)]
        subdims1 = [len(set(old_block_indices[i])) for i in range(ndim)]
        rec_cat_arg = np.empty(subdims1, dtype="O")
        rec_cat_arg_flat = rec_cat_arg.flat
        # Iterate over the old blocks required to build the new block
        for rec_cat_index, ind_slices in enumerate(cross1):
            old_block_index, slices = zip(*ind_slices)
            name = (split_name, next(split_name_suffixes))
            old_index = old_blocks[old_block_index][1:]
            if all(
                slc.start == 0 and slc.stop == x.chunks[i][ind]
                for i, (slc, ind) in enumerate(zip(slices, old_index))
            ):
                rec_cat_arg_flat[rec_cat_index] = old_blocks[old_block_index]
            else:
                intermediates[name] = (getitem, old_blocks[old_block_index], slices)
                rec_cat_arg_flat[rec_cat_index] = name
        assert rec_cat_index == rec_cat_arg.size - 1
        # New block is formed by concatenation of sliced old blocks
        if all(d == 1 for d in rec_cat_arg.shape):
            x2[key] = rec_cat_arg.flat[0]
        else:
            x2[key] = (concatenate3, rec_cat_arg.tolist())
    del old_blocks, new_index
    layer = toolz.merge(x2, intermediates)
    graph = HighLevelGraph.from_collections(merge_name, layer, dependencies=[x])
    return Array(graph, merge_name, chunks, meta=x)
class _PrettyBlocks(object):
    def __init__(self, blocks):
        self.blocks = blocks
    def __str__(self):
        runs = []
        run = []
        repeats = 0
        for c in self.blocks:
            if run and run[-1] == c:
                if repeats == 0 and len(run) > 1:
                    runs.append((None, run[:-1]))
                    run = run[-1:]
                repeats += 1
            else:
                if repeats > 0:
                    assert len(run) == 1
                    runs.append((repeats + 1, run[-1]))
                    run = []
                    repeats = 0
                run.append(c)
        if run:
            if repeats == 0:
                runs.append((None, run))
            else:
                assert len(run) == 1
                runs.append((repeats + 1, run[-1]))
        parts = []
        for repeats, run in runs:
            if repeats is None:
                parts.append(str(run))
            else:
                parts.append("%d*[%s]" % (repeats, run))
        return " | ".join(parts)
    __repr__ = __str__
def format_blocks(blocks):
    
    assert isinstance(blocks, tuple) and all(
        isinstance(x, int) or math.isnan(x) for x in blocks
    )
    return _PrettyBlocks(blocks)
def format_chunks(chunks):
    
    assert isinstance(chunks, tuple)
    return tuple(format_blocks(c) for c in chunks)
def format_plan(plan):
    
    return [format_chunks(c) for c in plan]
