from collections import defaultdict
from .utils_test import add, inc  # noqa: F401
no_default = "__no_default__"
def ishashable(x):
    
    try:
        hash(x)
        return True
    except TypeError:
        return False
def istask(x):
    
    return type(x) is tuple and x and callable(x[0])
def has_tasks(dsk, x):
    
    if istask(x):
        return True
    try:
        if x in dsk:
            return True
    except Exception:
        pass
    if isinstance(x, list):
        for i in x:
            if has_tasks(dsk, i):
                return True
    return False
def preorder_traversal(task):
    
    for item in task:
        if istask(item):
            for i in preorder_traversal(item):
                yield i
        elif isinstance(item, list):
            yield list
            for i in preorder_traversal(item):
                yield i
        else:
            yield item
def lists_to_tuples(res, keys):
    if isinstance(keys, list):
        return tuple(lists_to_tuples(r, k) for r, k in zip(res, keys))
    return res
def _execute_task(arg, cache, dsk=None):
    
    if isinstance(arg, list):
        return [_execute_task(a, cache) for a in arg]
    elif istask(arg):
        func, args = arg[0], arg[1:]
        args2 = [_execute_task(a, cache) for a in args]
        return func(*args2)
    elif not ishashable(arg):
        return arg
    elif arg in cache:
        return cache[arg]
    else:
        return arg
def get(dsk, out, cache=None):
    
    for k in flatten(out) if isinstance(out, list) else [out]:
        if k not in dsk:
            raise KeyError("{0} is not a key in the graph".format(k))
    if cache is None:
        cache = {}
    for key in toposort(dsk):
        task = dsk[key]
        result = _execute_task(task, cache)
        cache[key] = result
    result = _execute_task(out, cache)
    if isinstance(out, list):
        result = lists_to_tuples(result, out)
    return result
def get_dependencies(dsk, key=None, task=no_default, as_list=False):
    
    if key is not None:
        arg = dsk[key]
    elif task is not no_default:
        arg = task
    else:
        raise ValueError("Provide either key or task")
    result = []
    work = [arg]
    while work:
        new_work = []
        for w in work:
            typ = type(w)
            if typ is tuple and w and callable(w[0]):  # istask(w)
                new_work += w[1:]
            elif typ is list:
                new_work += w
            elif typ is dict:
                new_work += w.values()
            else:
                try:
                    if w in dsk:
                        result.append(w)
                except TypeError:  # not hashable
                    pass
        work = new_work
    return result if as_list else set(result)
def get_deps(dsk):
    
    dependencies = {k: get_dependencies(dsk, task=v) for k, v in dsk.items()}
    dependents = reverse_dict(dependencies)
    return dependencies, dependents
def flatten(seq, container=list):
    
    if isinstance(seq, str):
        yield seq
    else:
        for item in seq:
            if isinstance(item, container):
                for item2 in flatten(item, container=container):
                    yield item2
            else:
                yield item
def reverse_dict(d):
    
    result = defaultdict(set)
    _add = set.add
    for k, vals in d.items():
        result[k]
        for val in vals:
            _add(result[val], k)
    result.default_factory = None
    return result
def subs(task, key, val):
    
    type_task = type(task)
    if not (type_task is tuple and task and callable(task[0])):  # istask(task):
        try:
            if type_task is type(key) and task == key:
                return val
        except Exception:
            pass
        if type_task is list:
            return [subs(x, key, val) for x in task]
        return task
    newargs = []
    for arg in task[1:]:
        type_arg = type(arg)
        if type_arg is tuple and arg and callable(arg[0]):  # istask(task):
            arg = subs(arg, key, val)
        elif type_arg is list:
            arg = [subs(x, key, val) for x in arg]
        elif type_arg is type(key):
            try:
                # Can't do a simple equality check, since this may trigger
                # a FutureWarning from NumPy about array equality
                # https://github.com/dask/dask/pull/2457
                if len(arg) == len(key) and all(
                    type(aa) == type(bb) and aa == bb for aa, bb in zip(arg, key)
                ):
                    arg = val
            except (TypeError, AttributeError):
                # Handle keys which are not sized (len() fails), but are hashable
                if arg == key:
                    arg = val
        newargs.append(arg)
    return task[:1] + tuple(newargs)
def _toposort(dsk, keys=None, returncycle=False, dependencies=None):
    # Stack-based depth-first search traversal.  This is based on Tarjan's
    # method for topological sorting (see wikipedia for pseudocode)
    if keys is None:
        keys = dsk
    elif not isinstance(keys, list):
        keys = [keys]
    if not returncycle:
        ordered = []
    # Nodes whose descendents have been completely explored.
    # These nodes are guaranteed to not be part of a cycle.
    completed = set()
    # All nodes that have been visited in the current traversal.  Because
    # we are doing depth-first search, going "deeper" should never result
    # in visiting a node that has already been seen.  The `seen` and
    # `completed` sets are mutually exclusive; it is okay to visit a node
    # that has already been added to `completed`.
    seen = set()
    if dependencies is None:
        dependencies = dict((k, get_dependencies(dsk, k)) for k in dsk)
    for key in keys:
        if key in completed:
            continue
        nodes = [key]
        while nodes:
            # Keep current node on the stack until all descendants are visited
            cur = nodes[-1]
            if cur in completed:
                # Already fully traversed descendants of cur
                nodes.pop()
                continue
            seen.add(cur)
            # Add direct descendants of cur to nodes stack
            next_nodes = []
            for nxt in dependencies[cur]:
                if nxt not in completed:
                    if nxt in seen:
                        # Cycle detected!
                        cycle = [nxt]
                        while nodes[-1] != nxt:
                            cycle.append(nodes.pop())
                        cycle.append(nodes.pop())
                        cycle.reverse()
                        if returncycle:
                            return cycle
                        else:
                            cycle = "->".join(str(x) for x in cycle)
                            raise RuntimeError("Cycle detected in Dask: %s" % cycle)
                    next_nodes.append(nxt)
            if next_nodes:
                nodes.extend(next_nodes)
            else:
                # cur has no more descendants to explore, so we're done with it
                if not returncycle:
                    ordered.append(cur)
                completed.add(cur)
                seen.remove(cur)
                nodes.pop()
    if returncycle:
        return []
    return ordered
def toposort(dsk, dependencies=None):
    
    return _toposort(dsk, dependencies=dependencies)
def getcycle(d, keys):
    
    return _toposort(d, keys=keys, returncycle=True)
def isdag(d, keys):
    
    return not getcycle(d, keys)
class literal(object):
    
    __slots__ = ("data",)
    def __init__(self, data):
        self.data = data
    def __repr__(self):
        return "literal<type=%s>" % type(self.data).__name__
    def __reduce__(self):
        return (literal, (self.data,))
    def __call__(self):
        return self.data
def quote(x):
    
    if istask(x) or type(x) is list:
        return (literal(x),)
    return x
