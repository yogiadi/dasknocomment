
import threading
from functools import partial
from . import config
_globals = config.config
thread_state = threading.local()
def globalmethod(default=None, key=None, falsey=None):
    
    if default is None:
        return partial(globalmethod, key=key, falsey=falsey)
    return GlobalMethod(default=default, key=key, falsey=falsey)
class GlobalMethod(object):
    def __init__(self, default, key, falsey=None):
        self._default = default
        self._key = key
        self._falsey = falsey
    def __get__(self, instance, owner=None):
        if self._key in _globals:
            if _globals[self._key]:
                return _globals[self._key]
            elif self._falsey is not None:
                return self._falsey
        return self._default
