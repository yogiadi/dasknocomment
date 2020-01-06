
import os
from queue import Queue, Empty
from . import config
from .core import flatten, reverse_dict, get_dependencies, has_tasks, _execute_task
from .order import order
from .callbacks import unpack_callbacks, local_callbacks
from .utils_test import add, inc  # noqa: F401
if os.name == "nt":
    # Python 3 windows Queue.get doesn't handle interrupts properly. To
    # workaround this we poll at a sufficiently large interval that it
    # shouldn't affect performance, but small enough that users trying to kill
    # an application shouldn't care.
    def queue_get(q):
        while True:
            try:
                return q.get(block=True, timeout=0.1)
            except Empty:
                pass
else:
    def queue_get(q):
        return q.get()
DEBUG = False
def start_state_from_dask(dsk, cache=None, sortkey=None):
    
    if sortkey is None:
        sortkey = order(dsk).get
    if cache is None:
        cache = config.get("cache", None)
    if cache is None:
        cache = dict()
    data_keys = set()
    for k, v in dsk.items():
        if not has_tasks(dsk, v):
            cache[k] = v
            data_keys.add(k)
    dsk2 = dsk.copy()
    dsk2.update(cache)
    dependencies = {k: get_dependencies(dsk2, k) for k in dsk}
    waiting = {k: v.copy() for k, v in dependencies.items() if k not in data_keys}
    dependents = reverse_dict(dependencies)
    for a in cache:
        for b in dependents.get(a, ()):
            waiting[b].remove(a)
    waiting_data = dict((k, v.copy()) for k, v in dependents.items() if v)
    ready_set = set([k for k, v in waiting.items() if not v])
    ready = sorted(ready_set, key=sortkey, reverse=True)
    waiting = dict((k, v) for k, v in waiting.items() if v)
    state = {
        "dependencies": dependencies,
        "dependents": dependents,
        "waiting": waiting,
        "waiting_data": waiting_data,
        "cache": cache,
        "ready": ready,
        "running": set(),
        "finished": set(),
        "released": set(),
    }
    return state

def execute_task(key, task_info, dumps, loads, get_id, pack_exception):
    
    try:
        task, data = loads(task_info)
        result = _execute_task(task, data)
        id = get_id()
        result = dumps((result, id))
        failed = False
    except BaseException as e:
        result = pack_exception(e, dumps)
        failed = True
    return key, result, failed
def release_data(key, state, delete=True):
    
    if key in state["waiting_data"]:
        assert not state["waiting_data"][key]
        del state["waiting_data"][key]
    state["released"].add(key)
    if delete:
        del state["cache"][key]
def finish_task(
    dsk, key, state, results, sortkey, delete=True, release_data=release_data
):
    
    for dep in sorted(state["dependents"][key], key=sortkey, reverse=True):
        s = state["waiting"][dep]
        s.remove(key)
        if not s:
            del state["waiting"][dep]
            state["ready"].append(dep)
    for dep in state["dependencies"][key]:
        if dep in state["waiting_data"]:
            s = state["waiting_data"][dep]
            s.remove(key)
            if not s and dep not in results:
                if DEBUG:
                    from chest.core import nbytes
                    print(
                        "Key: %s\tDep: %s\t NBytes: %.2f\t Release"
                        % (key, dep, sum(map(nbytes, state["cache"].values()) / 1e6))
                    )
                release_data(dep, state, delete=delete)
        elif delete and dep not in results:
            release_data(dep, state, delete=delete)
    state["finished"].add(key)
    state["running"].remove(key)
    return state
def nested_get(ind, coll):
    
    if isinstance(ind, list):
        return tuple([nested_get(i, coll) for i in ind])
    else:
        return coll[ind]
def default_get_id():
    
    return None
def default_pack_exception(e, dumps):
    raise
def reraise(exc, tb=None):
    if exc.__traceback__ is not tb:
        raise exc.with_traceback(tb)
    raise exc
def identity(x):
    
    return x

def get_async(
    apply_async,
    num_workers,
    dsk,
    result,
    cache=None,
    get_id=default_get_id,
    rerun_exceptions_locally=None,
    pack_exception=default_pack_exception,
    raise_exception=reraise,
    callbacks=None,
    dumps=identity,
    loads=identity,
    **kwargs
):
    
    queue = Queue()
    if isinstance(result, list):
        result_flat = set(flatten(result))
    else:
        result_flat = set([result])
    results = set(result_flat)
    dsk = dict(dsk)
    with local_callbacks(callbacks) as callbacks:
        _, _, pretask_cbs, posttask_cbs, _ = unpack_callbacks(callbacks)
        started_cbs = []
        succeeded = False
        # if start_state_from_dask fails, we will have something
        # to pass to the final block.
        state = {}
        try:
            for cb in callbacks:
                if cb[0]:
                    cb[0](dsk)
                started_cbs.append(cb)
            keyorder = order(dsk)
            state = start_state_from_dask(dsk, cache=cache, sortkey=keyorder.get)
            for _, start_state, _, _, _ in callbacks:
                if start_state:
                    start_state(dsk, state)
            if rerun_exceptions_locally is None:
                rerun_exceptions_locally = config.get("rerun_exceptions_locally", False)
            if state["waiting"] and not state["ready"]:
                raise ValueError("Found no accessible jobs in dask")
            def fire_task():
                
                # Choose a good task to compute
                key = state["ready"].pop()
                state["running"].add(key)
                for f in pretask_cbs:
                    f(key, dsk, state)
                # Prep data to send
                data = dict(
                    (dep, state["cache"][dep]) for dep in get_dependencies(dsk, key)
                )
                # Submit
                apply_async(
                    execute_task,
                    args=(
                        key,
                        dumps((dsk[key], data)),
                        dumps,
                        loads,
                        get_id,
                        pack_exception,
                    ),
                    callback=queue.put,
                )
            # Seed initial tasks into the thread pool
            while state["ready"] and len(state["running"]) < num_workers:
                fire_task()
            # Main loop, wait on tasks to finish, insert new ones
            while state["waiting"] or state["ready"] or state["running"]:
                key, res_info, failed = queue_get(queue)
                if failed:
                    exc, tb = loads(res_info)
                    if rerun_exceptions_locally:
                        data = dict(
                            (dep, state["cache"][dep])
                            for dep in get_dependencies(dsk, key)
                        )
                        task = dsk[key]
                        _execute_task(task, data)  # Re-execute locally
                    else:
                        raise_exception(exc, tb)
                res, worker_id = loads(res_info)
                state["cache"][key] = res
                finish_task(dsk, key, state, results, keyorder.get)
                for f in posttask_cbs:
                    f(key, res, dsk, state, worker_id)
                while state["ready"] and len(state["running"]) < num_workers:
                    fire_task()
            succeeded = True
        finally:
            for _, _, _, _, finish in started_cbs:
                if finish:
                    finish(dsk, state, not succeeded)
    return nested_get(result, state["cache"])

def apply_sync(func, args=(), kwds={}, callback=None):
    
    res = func(*args, **kwds)
    if callback is not None:
        callback(res)
def get_sync(dsk, keys, **kwargs):
    
    kwargs.pop("num_workers", None)  # if num_workers present, remove it
    return get_async(apply_sync, 1, dsk, keys, **kwargs)
def sortkey(item):
    
    return (type(item).__name__, item)
