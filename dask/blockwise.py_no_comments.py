import itertools
import warnings
from collections.abc import Mapping
import numpy as np
try:
    import cytoolz as toolz
except ImportError:
    import toolz
from .core import reverse_dict
from .delayed import to_task_dask
from .highlevelgraph import HighLevelGraph
from .optimization import SubgraphCallable, fuse
from .utils import ensure_dict, homogeneous_deepmap, apply
def subs(task, substitution):
    
    if isinstance(task, dict):
        return {k: subs(v, substitution) for k, v in task.items()}
    if type(task) in (tuple, list, set):
        return type(task)([subs(x, substitution) for x in task])
    try:
        return substitution[task]
    except (KeyError, TypeError):
        return task
def index_subs(ind, substitution):
    
    if ind is None:
        return ind
    else:
        return tuple([substitution.get(c, c) for c in ind])
def blockwise_token(i, prefix="_"):
    return prefix + "%d" % i
def blockwise(
    func,
    output,
    output_indices,
    *arrind_pairs,
    numblocks=None,
    concatenate=None,
    new_axes=None,
    dependencies=(),
    **kwargs
):
    
    new_axes = new_axes or {}
    arrind_pairs = list(arrind_pairs)
    # Transform indices to canonical elements
    # We use terms like _0, and _1 rather than provided index elements
    unique_indices = {
        i for ii in arrind_pairs[1::2] if ii is not None for i in ii
    } | set(output_indices)
    sub = {k: blockwise_token(i, ".") for i, k in enumerate(sorted(unique_indices))}
    output_indices = index_subs(tuple(output_indices), sub)
    arrind_pairs[1::2] = [tuple(a) if a is not None else a for a in arrind_pairs[1::2]]
    arrind_pairs[1::2] = [index_subs(a, sub) for a in arrind_pairs[1::2]]
    new_axes = {index_subs((k,), sub)[0]: v for k, v in new_axes.items()}
    # Unpack dask values in non-array arguments
    argpairs = list(toolz.partition(2, arrind_pairs))
    # separate argpairs into two separate tuples
    inputs = tuple([name for name, _ in argpairs])
    inputs_indices = tuple([index for _, index in argpairs])
    # Unpack delayed objects in kwargs
    new_keys = {n for c in dependencies for n in c.__dask_layers__()}
    if kwargs:
        # replace keys in kwargs with _0 tokens
        new_tokens = tuple(
            blockwise_token(i) for i in range(len(inputs), len(inputs) + len(new_keys))
        )
        sub = dict(zip(new_keys, new_tokens))
        inputs = inputs + tuple(new_keys)
        inputs_indices = inputs_indices + (None,) * len(new_keys)
        kwargs = subs(kwargs, sub)
    indices = [(k, v) for k, v in zip(inputs, inputs_indices)]
    keys = tuple(map(blockwise_token, range(len(inputs))))
    # Construct local graph
    if not kwargs:
        subgraph = {output: (func,) + keys}
    else:
        _keys = list(keys)
        if new_keys:
            _keys = _keys[: -len(new_keys)]
        kwargs2 = (dict, list(map(list, kwargs.items())))
        subgraph = {output: (apply, func, _keys, kwargs2)}
    # Construct final output
    subgraph = Blockwise(
        output,
        output_indices,
        subgraph,
        indices,
        numblocks=numblocks,
        concatenate=concatenate,
        new_axes=new_axes,
    )
    return subgraph
class Blockwise(Mapping):
    
    def __init__(
        self,
        output,
        output_indices,
        dsk,
        indices,
        numblocks,
        concatenate=None,
        new_axes=None,
    ):
        self.output = output
        self.output_indices = tuple(output_indices)
        self.dsk = dsk
        self.indices = tuple(
            (name, tuple(ind) if ind is not None else ind) for name, ind in indices
        )
        self.numblocks = numblocks
        self.concatenate = concatenate
        self.new_axes = new_axes or {}
    def __repr__(self):
        return "Blockwise<{} -> {}>".format(self.indices, self.output)
    @property
    def _dict(self):
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            keys = tuple(map(blockwise_token, range(len(self.indices))))
            func = SubgraphCallable(self.dsk, self.output, keys)
            self._cached_dict = make_blockwise_graph(
                func,
                self.output,
                self.output_indices,
                *list(toolz.concat(self.indices)),
                new_axes=self.new_axes,
                numblocks=self.numblocks,
                concatenate=self.concatenate
            )
        return self._cached_dict
    def __getitem__(self, key):
        return self._dict[key]
    def __iter__(self):
        return iter(self._dict)
    def __len__(self):
        return int(np.prod(list(self._out_numblocks().values())))
    def _out_numblocks(self):
        d = {}
        indices = {k: v for k, v in self.indices if v is not None}
        for k, v in self.numblocks.items():
            for a, b in zip(indices[k], v):
                d[a] = max(d.get(a, 0), b)
        return {k: v for k, v in d.items() if k in self.output_indices}
def make_blockwise_graph(func, output, out_indices, *arrind_pairs, **kwargs):
    
    numblocks = kwargs.pop("numblocks")
    concatenate = kwargs.pop("concatenate", None)
    new_axes = kwargs.pop("new_axes", {})
    argpairs = list(toolz.partition(2, arrind_pairs))
    if concatenate is True:
        from dask.array.core import concatenate_axes as concatenate
    assert set(numblocks) == {name for name, ind in argpairs if ind is not None}
    all_indices = {x for _, ind in argpairs if ind for x in ind}
    dummy_indices = all_indices - set(out_indices)
    # Dictionary mapping {i: 3, j: 4, ...} for i, j, ... the dimensions
    dims = broadcast_dimensions(argpairs, numblocks)
    for k, v in new_axes.items():
        dims[k] = len(v) if isinstance(v, tuple) else 1
    # (0, 0), (0, 1), (0, 2), (1, 0), ...
    keytups = list(itertools.product(*[range(dims[i]) for i in out_indices]))
    # {i: 0, j: 0}, {i: 0, j: 1}, ...
    keydicts = [dict(zip(out_indices, tup)) for tup in keytups]
    # {j: [1, 2, 3], ...}  For j a dummy index of dimension 3
    dummies = dict((i, list(range(dims[i]))) for i in dummy_indices)
    dsk = {}
    # Create argument lists
    valtups = []
    for kd in keydicts:
        args = []
        for arg, ind in argpairs:
            if ind is None:
                args.append(arg)
            else:
                tups = lol_tuples((arg,), ind, kd, dummies)
                if any(nb == 1 for nb in numblocks[arg]):
                    tups2 = zero_broadcast_dimensions(tups, numblocks[arg])
                else:
                    tups2 = tups
                if concatenate and isinstance(tups2, list):
                    axes = [n for n, i in enumerate(ind) if i in dummies]
                    tups2 = (concatenate, tups2, axes)
                args.append(tups2)
        valtups.append(args)
    if not kwargs:  # will not be used in an apply, should be a tuple
        valtups = [tuple(vt) for vt in valtups]
    # Add heads to tuples
    keys = [(output,) + kt for kt in keytups]
    # Unpack delayed objects in kwargs
    if kwargs:
        task, dsk2 = to_task_dask(kwargs)
        if dsk2:
            dsk.update(ensure_dict(dsk2))
            kwargs2 = task
        else:
            kwargs2 = kwargs
        vals = [(apply, func, vt, kwargs2) for vt in valtups]
    else:
        vals = [(func,) + vt for vt in valtups]
    dsk.update(dict(zip(keys, vals)))
    return dsk
def lol_tuples(head, ind, values, dummies):
    
    if not ind:
        return head
    if ind[0] not in dummies:
        return lol_tuples(head + (values[ind[0]],), ind[1:], values, dummies)
    else:
        return [
            lol_tuples(head + (v,), ind[1:], values, dummies) for v in dummies[ind[0]]
        ]
def optimize_blockwise(graph, keys=()):
    
    with warnings.catch_warnings():
        # In some cases, rewrite_blockwise (called internally) will do a bad
        # thing like `string in array[int].
        # See dask/array/tests/test_atop.py::test_blockwise_numpy_arg for
        # an example. NumPy currently raises a warning that 'a' == array([1, 2])
        # will change from returning `False` to `array([False, False])`.
        #
        # Users shouldn't see those warnings, so we filter them.
        # We set the filter here, rather than lower down, to avoid having to
        # create and remove the filter many times inside a tight loop.
        # https://github.com/dask/dask/pull/4805#discussion_r286545277 explains
        # why silencing this warning shouldn't cause issues.
        warnings.filterwarnings(
            "ignore", "elementwise comparison failed", Warning
        )  # FutureWarning or DeprecationWarning
        out = _optimize_blockwise(graph, keys=keys)
        while out.dependencies != graph.dependencies:
            graph = out
            out = _optimize_blockwise(graph, keys=keys)
    return out
def _optimize_blockwise(full_graph, keys=()):
    keep = {k[0] if type(k) is tuple else k for k in keys}
    layers = full_graph.dicts
    dependents = reverse_dict(full_graph.dependencies)
    roots = {k for k in full_graph.dicts if not dependents.get(k)}
    stack = list(roots)
    out = {}
    dependencies = {}
    seen = set()
    while stack:
        layer = stack.pop()
        if layer in seen or layer not in layers:
            continue
        seen.add(layer)
        # Outer loop walks through possible output Blockwise layers
        if isinstance(layers[layer], Blockwise):
            blockwise_layers = {layer}
            deps = set(blockwise_layers)
            while deps:  # we gather as many sub-layers as we can
                dep = deps.pop()
                if dep not in layers:
                    stack.append(dep)
                    continue
                if not isinstance(layers[dep], Blockwise):
                    stack.append(dep)
                    continue
                if dep != layer and dep in keep:
                    stack.append(dep)
                    continue
                if layers[dep].concatenate != layers[layer].concatenate:
                    stack.append(dep)
                    continue
                if (
                    sum(k == dep for k, ind in layers[layer].indices if ind is not None)
                    > 1
                ):
                    stack.append(dep)
                    continue
                # passed everything, proceed
                blockwise_layers.add(dep)
                # traverse further to this child's children
                for d in full_graph.dependencies.get(dep, ()):
                    # Don't allow reductions to proceed
                    output_indices = set(layers[dep].output_indices)
                    input_indices = {
                        i for _, ind in layers[dep].indices if ind for i in ind
                    }
                    if len(dependents[d]) <= 1 and output_indices.issuperset(
                        input_indices
                    ):
                        deps.add(d)
                    else:
                        stack.append(d)
            # Merge these Blockwise layers into one
            new_layer = rewrite_blockwise([layers[l] for l in blockwise_layers])
            out[layer] = new_layer
            dependencies[layer] = {k for k, v in new_layer.indices if v is not None}
        else:
            out[layer] = layers[layer]
            dependencies[layer] = full_graph.dependencies.get(layer, set())
            stack.extend(full_graph.dependencies.get(layer, ()))
    return HighLevelGraph(out, dependencies)
def rewrite_blockwise(inputs):
    
    inputs = {inp.output: inp for inp in inputs}
    dependencies = {
        inp.output: {d for d, v in inp.indices if v is not None and d in inputs}
        for inp in inputs.values()
    }
    dependents = reverse_dict(dependencies)
    new_index_iter = (
        c + (str(d) if d else "")  # A, B, ... A1, B1, ...
        for d in itertools.count()
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    [root] = [k for k, v in dependents.items() if not v]
    # Our final results.  These will change during fusion below
    indices = list(inputs[root].indices)
    new_axes = inputs[root].new_axes
    concatenate = inputs[root].concatenate
    dsk = dict(inputs[root].dsk)
    changed = True
    while changed:
        changed = False
        for i, (dep, ind) in enumerate(indices):
            if ind is None:
                continue
            if dep not in inputs:
                continue
            changed = True
            # Replace _n with dep name in existing tasks
            # (inc, _0) -> (inc, 'b')
            dsk = {k: subs(v, {blockwise_token(i): dep}) for k, v in dsk.items()}
            # Remove current input from input indices
            # [('a', 'i'), ('b', 'i')] -> [('a', 'i')]
            _, current_dep_indices = indices.pop(i)
            sub = {
                blockwise_token(i): blockwise_token(i - 1)
                for i in range(i + 1, len(indices) + 1)
            }
            dsk = subs(dsk, sub)
            # Change new input_indices to match give index from current computation
            # [('c', j')] -> [('c', 'i')]
            new_indices = inputs[dep].indices
            sub = dict(zip(inputs[dep].output_indices, current_dep_indices))
            contracted = {
                x
                for _, j in new_indices
                if j is not None
                for x in j
                if x not in inputs[dep].output_indices
            }
            extra = dict(zip(contracted, new_index_iter))
            sub.update(extra)
            new_indices = [(x, index_subs(j, sub)) for x, j in new_indices]
            # Update new_axes
            for k, v in inputs[dep].new_axes.items():
                new_axes[sub[k]] = v
            # Bump new inputs up in list
            sub = {}
            # Map from (id(key), inds or None) -> index in indices. Used to deduplicate indices.
            index_map = {(id(k), inds): n for n, (k, inds) in enumerate(indices)}
            for i, index in enumerate(new_indices):
                id_key = (id(index[0]), index[1])
                if id_key in index_map:  # use old inputs if available
                    sub[blockwise_token(i)] = blockwise_token(index_map[id_key])
                else:
                    index_map[id_key] = len(indices)
                    sub[blockwise_token(i)] = blockwise_token(len(indices))
                    indices.append(index)
            new_dsk = subs(inputs[dep].dsk, sub)
            # indices.extend(new_indices)
            dsk.update(new_dsk)
    # De-duplicate indices like [(a, ij), (b, i), (a, ij)] -> [(a, ij), (b, i)]
    # Make sure that we map everything else appropriately as we remove inputs
    new_indices = []
    seen = {}
    sub = {}  # like {_0: _0, _1: _0, _2: _1}
    for i, x in enumerate(indices):
        if x[1] is not None and x in seen:
            sub[i] = seen[x]
        else:
            if x[1] is not None:
                seen[x] = len(new_indices)
            sub[i] = len(new_indices)
            new_indices.append(x)
    sub = {blockwise_token(k): blockwise_token(v) for k, v in sub.items()}
    dsk = {k: subs(v, sub) for k, v in dsk.items()}
    indices_check = {k for k, v in indices if v is not None}
    numblocks = toolz.merge([inp.numblocks for inp in inputs.values()])
    numblocks = {k: v for k, v in numblocks.items() if v is None or k in indices_check}
    out = Blockwise(
        root,
        inputs[root].output_indices,
        dsk,
        new_indices,
        numblocks=numblocks,
        new_axes=new_axes,
        concatenate=concatenate,
    )
    return out
def zero_broadcast_dimensions(lol, nblocks):
    
    f = lambda t: (t[0],) + tuple(0 if d == 1 else i for i, d in zip(t[1:], nblocks))
    return homogeneous_deepmap(f, lol)
def broadcast_dimensions(argpairs, numblocks, sentinels=(1, (1,)), consolidate=None):
    
    # List like [('i', 2), ('j', 1), ('i', 1), ('j', 2)]
    argpairs2 = [(a, ind) for a, ind in argpairs if ind is not None]
    L = toolz.concat(
        [
            zip(inds, dims)
            for (x, inds), (x, dims) in toolz.join(
                toolz.first, argpairs2, toolz.first, numblocks.items()
            )
        ]
    )
    g = toolz.groupby(0, L)
    g = dict((k, set([d for i, d in v])) for k, v in g.items())
    g2 = dict((k, v - set(sentinels) if len(v) > 1 else v) for k, v in g.items())
    if consolidate:
        return toolz.valmap(consolidate, g2)
    if g2 and not set(map(len, g2.values())) == set([1]):
        raise ValueError("Shapes do not align %s" % g)
    return toolz.valmap(toolz.first, g2)
def fuse_roots(graph: HighLevelGraph, keys: list):
    
    layers = graph.layers.copy()
    dependencies = graph.dependencies.copy()
    dependents = reverse_dict(dependencies)
    for name, layer in graph.layers.items():
        deps = graph.dependencies[name]
        if (
            isinstance(layer, Blockwise)
            and len(deps) > 1
            and not any(dependencies[dep] for dep in deps)  # no need to fuse if 0 or 1
            and all(len(dependents[dep]) == 1 for dep in deps)
        ):
            new = toolz.merge(layer, *[layers[dep] for dep in deps])
            new, _ = fuse(new, keys, ave_width=len(deps))
            for dep in deps:
                del layers[dep]
            layers[name] = new
            dependencies[name] = set()
    return HighLevelGraph(layers, dependencies)
