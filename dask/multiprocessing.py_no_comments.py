import copyreg
import multiprocessing
import pickle
import sys
import traceback
from functools import partial
from warnings import warn
from . import config
from .system import CPU_COUNT
from .local import reraise, get_async  # TODO: get better get
from .optimization import fuse, cull
def _reduce_method_descriptor(m):
    return getattr, (m.__objclass__, m.__name__)
# type(set.union) is used as a proxy to <class 'method_descriptor'>
copyreg.pickle(type(set.union), _reduce_method_descriptor)
try:
    import cloudpickle
    _dumps = partial(cloudpickle.dumps, protocol=pickle.HIGHEST_PROTOCOL)
    _loads = cloudpickle.loads
except ImportError:
    def _dumps(obj):
        try:
            return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, AttributeError) as exc:
            raise ModuleNotFoundError(
                "Please install cloudpickle to use the multiprocessing scheduler"
            ) from exc
    _loads = pickle.loads
def _process_get_id():
    return multiprocessing.current_process().ident
# -- Remote Exception Handling --
# By default, tracebacks can't be serialized using pickle. However, the
# `tblib` library can enable support for this. Since we don't mandate
# that tblib is installed, we do the following:
#
# - If tblib is installed, use it to serialize the traceback and reraise
#   in the scheduler process
# - Otherwise, use a ``RemoteException`` class to contain a serialized
#   version of the formatted traceback, which will then print in the
#   scheduler process.
#
# To enable testing of the ``RemoteException`` class even when tblib is
# installed, we don't wrap the class in the try block below
class RemoteException(Exception):
    
    def __init__(self, exception, traceback):
        self.exception = exception
        self.traceback = traceback
    def __str__(self):
        return str(self.exception) + "\n\nTraceback\n---------\n" + self.traceback
    def __dir__(self):
        return sorted(set(dir(type(self)) + list(self.__dict__) + dir(self.exception)))
    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            return getattr(self.exception, key)
exceptions = dict()
def remote_exception(exc, tb):
    
    if type(exc) in exceptions:
        typ = exceptions[type(exc)]
        return typ(exc, tb)
    else:
        try:
            typ = type(
                exc.__class__.__name__,
                (RemoteException, type(exc)),
                {"exception_type": type(exc)},
            )
            exceptions[type(exc)] = typ
            return typ(exc, tb)
        except TypeError:
            return exc
try:
    import tblib.pickling_support
    tblib.pickling_support.install()
    def _pack_traceback(tb):
        return tb
except ImportError:
    def _pack_traceback(tb):
        return "".join(traceback.format_tb(tb))
    def reraise(exc, tb):
        exc = remote_exception(exc, tb)
        raise exc
def pack_exception(e, dumps):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb = _pack_traceback(exc_traceback)
    try:
        result = dumps((e, tb))
    except BaseException as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = _pack_traceback(exc_traceback)
        result = dumps((e, tb))
    return result
_CONTEXT_UNSUPPORTED = 
def get_context():
    
    if sys.platform == "win32":
        # Just do the default, since we can't change it:
        if config.get("multiprocessing.context", None) is not None:
            warn(_CONTEXT_UNSUPPORTED, UserWarning)
        return multiprocessing
    context_name = config.get("multiprocessing.context", None)
    # The default context on OSX was switched from "fork" to "spawn" in
    # Python 3.8. We keep "fork" as the default here for backwards
    # compatibility and to avoid raising a RuntimeError when using the
    # multiprocessing scheduler in a Python script that is not in a
    # if __name__ == "__main__" block
    if context_name is None and sys.platform == "darwin":
        context_name = "fork"
    return multiprocessing.get_context(context_name)
def get(
    dsk,
    keys,
    num_workers=None,
    func_loads=None,
    func_dumps=None,
    optimize_graph=True,
    pool=None,
    **kwargs
):
    
    pool = pool or config.get("pool", None)
    num_workers = num_workers or config.get("num_workers", None) or CPU_COUNT
    if pool is None:
        context = get_context()
        pool = context.Pool(num_workers, initializer=initialize_worker_process)
        cleanup = True
    else:
        cleanup = False
    # Optimize Dask
    dsk2, dependencies = cull(dsk, keys)
    if optimize_graph:
        dsk3, dependencies = fuse(dsk2, keys, dependencies)
    else:
        dsk3 = dsk2
    # We specify marshalling functions in order to catch serialization
    # errors and report them to the user.
    loads = func_loads or config.get("func_loads", None) or _loads
    dumps = func_dumps or config.get("func_dumps", None) or _dumps
    # Note former versions used a multiprocessing Manager to share
    # a Queue between parent and workers, but this is fragile on Windows
    # (issue #1652).
    try:
        # Run
        result = get_async(
            pool.apply_async,
            len(pool._pool),
            dsk3,
            keys,
            get_id=_process_get_id,
            dumps=dumps,
            loads=loads,
            pack_exception=pack_exception,
            raise_exception=reraise,
            **kwargs
        )
    finally:
        if cleanup:
            pool.close()
    return result
def initialize_worker_process():
    
    # If Numpy is already imported, presumably its random state was
    # inherited from the parent => re-seed it.
    np = sys.modules.get("numpy")
    if np is not None:
        np.random.seed()
