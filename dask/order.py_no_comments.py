r
from .core import get_dependencies, reverse_dict, get_deps  # noqa: F401
from .utils_test import add, inc  # noqa: F401
def order(dsk, dependencies=None):
    
    if dependencies is None:
        dependencies = {k: get_dependencies(dsk, k) for k in dsk}
    for k, deps in dependencies.items():
        deps.discard(k)
    dependents = reverse_dict(dependencies)
    total_dependencies = ndependencies(dependencies, dependents)
    total_dependents, min_dependencies = ndependents(
        dependencies, dependents, total_dependencies
    )
    waiting = {k: set(v) for k, v in dependencies.items()}
    def dependencies_key(x):
        return total_dependencies.get(x, 0), ReverseStrComparable(x)
    def dependents_key(x):
        return (min_dependencies[x], -(total_dependents.get(x, 0)), StrComparable(x))
    result = dict()
    seen = set()  # tasks that should not be added again to the stack
    i = 0
    stack = [k for k, v in dependents.items() if not v]
    if len(stack) < 10000:
        stack = sorted(stack, key=dependencies_key)
    else:
        stack = stack[::-1]
    while stack:
        item = stack.pop()
        if item in result:
            continue
        deps = waiting[item]
        if deps:
            stack.append(item)
            seen.add(item)
            if len(deps) < 1000:
                deps = sorted(deps, key=dependencies_key)
            stack.extend(deps)
            continue
        result[item] = i
        i += 1
        for dep in dependents[item]:
            waiting[dep].discard(item)
        deps = [
            d
            for d in dependents[item]
            if d not in result and not (d in seen and len(waiting[d]) > 1)
        ]
        if len(deps) < 1000:
            deps = sorted(deps, key=dependents_key, reverse=True)
        stack.extend(deps)
    return result
def ndependents(dependencies, dependents, total_dependencies):
    
    result = dict()
    min_result = dict()
    num_needed = {k: len(v) for k, v in dependents.items()}
    current = {k for k, v in num_needed.items() if v == 0}
    while current:
        key = current.pop()
        result[key] = 1 + sum(result[parent] for parent in dependents[key])
        try:
            min_result[key] = min(min_result[parent] for parent in dependents[key])
        except ValueError:
            min_result[key] = total_dependencies[key]
        for child in dependencies[key]:
            num_needed[child] -= 1
            if num_needed[child] == 0:
                current.add(child)
    return result, min_result
def ndependencies(dependencies, dependents):
    
    result = dict()
    num_needed = {k: len(v) for k, v in dependencies.items()}
    current = {k for k, v in num_needed.items() if v == 0}
    while current:
        key = current.pop()
        result[key] = 1 + sum(result[child] for child in dependencies[key])
        for parent in dependents[key]:
            num_needed[parent] -= 1
            if num_needed[parent] == 0:
                current.add(parent)
    return result
class StrComparable(object):
    
    __slots__ = ("obj",)
    def __init__(self, obj):
        self.obj = obj
    def __lt__(self, other):
        try:
            return self.obj < other.obj
        except Exception:
            return str(self.obj) < str(other.obj)
class ReverseStrComparable(object):
    
    __slots__ = ("obj",)
    def __init__(self, obj):
        self.obj = obj
    def __lt__(self, other):
        try:
            return self.obj > other.obj
        except Exception:
            return str(self.obj) > str(other.obj)
