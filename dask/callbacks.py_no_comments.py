from contextlib import contextmanager
__all__ = ["Callback", "add_callbacks"]
class Callback(object):
    
    active = set()
    def __init__(
        self, start=None, start_state=None, pretask=None, posttask=None, finish=None
    ):
        if start:
            self._start = start
        if start_state:
            self._start_state = start_state
        if pretask:
            self._pretask = pretask
        if posttask:
            self._posttask = posttask
        if finish:
            self._finish = finish
    @property
    def _callback(self):
        fields = ["_start", "_start_state", "_pretask", "_posttask", "_finish"]
        return tuple(getattr(self, i, None) for i in fields)
    def __enter__(self):
        self._cm = add_callbacks(self)
        self._cm.__enter__()
        return self
    def __exit__(self, *args):
        self._cm.__exit__(*args)
    def register(self):
        Callback.active.add(self._callback)
    def unregister(self):
        Callback.active.remove(self._callback)
def unpack_callbacks(cbs):
    
    if cbs:
        return [[i for i in f if i] for f in zip(*cbs)]
    else:
        return [(), (), (), (), ()]
@contextmanager
def local_callbacks(callbacks=None):
    
    global_callbacks = callbacks is None
    if global_callbacks:
        callbacks, Callback.active = Callback.active, set()
    try:
        yield callbacks or ()
    finally:
        if global_callbacks:
            Callback.active = callbacks
def normalize_callback(cb):
    
    if isinstance(cb, Callback):
        return cb._callback
    elif isinstance(cb, tuple):
        return cb
    else:
        raise TypeError("Callbacks must be either `Callback` or `tuple`")
class add_callbacks(object):
    
    def __init__(self, *callbacks):
        self.callbacks = [normalize_callback(c) for c in callbacks]
        Callback.active.update(self.callbacks)
    def __enter__(self):
        return
    def __exit__(self, type, value, traceback):
        for c in self.callbacks:
            Callback.active.discard(c)
