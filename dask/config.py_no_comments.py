import ast
import builtins
import os
import sys
import threading
from collections.abc import Mapping
try:
    import yaml
except ImportError:
    yaml = None
no_default = "__no_default__"
paths = [
    os.getenv("DASK_ROOT_CONFIG", "/etc/dask"),
    os.path.join(sys.prefix, "etc", "dask"),
    os.path.join(os.path.expanduser("~"), ".config", "dask"),
    os.path.join(os.path.expanduser("~"), ".dask"),
]
if "DASK_CONFIG" in os.environ:
    PATH = os.environ["DASK_CONFIG"]
    paths.append(PATH)
else:
    PATH = os.path.join(os.path.expanduser("~"), ".config", "dask")
global_config = config = {}
config_lock = threading.Lock()
defaults = []
def canonical_name(k, config):
    
    try:
        if k in config:
            return k
    except TypeError:
        # config is not a mapping, return the same name as provided
        return k
    altk = k.replace("_", "-") if "_" in k else k.replace("-", "_")
    if altk in config:
        return altk
    return k
def update(old, new, priority="new"):
    
    for k, v in new.items():
        k = canonical_name(k, old)
        if isinstance(v, Mapping):
            if k not in old or old[k] is None:
                old[k] = {}
            update(old[k], v, priority=priority)
        else:
            if priority == "new" or k not in old:
                old[k] = v
    return old
def merge(*dicts):
    
    result = {}
    for d in dicts:
        update(result, d)
    return result
def collect_yaml(paths=paths):
    
    # Find all paths
    file_paths = []
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                try:
                    file_paths.extend(
                        sorted(
                            [
                                os.path.join(path, p)
                                for p in os.listdir(path)
                                if os.path.splitext(p)[1].lower()
                                in (".json", ".yaml", ".yml")
                            ]
                        )
                    )
                except OSError:
                    # Ignore permission errors
                    pass
            else:
                file_paths.append(path)
    configs = []
    # Parse yaml files
    for path in file_paths:
        try:
            with open(path) as f:
                data = yaml.safe_load(f.read()) or {}
                configs.append(data)
        except (OSError, IOError):
            # Ignore permission errors
            pass
    return configs
def collect_env(env=None):
    
    if env is None:
        env = os.environ
    d = {}
    for name, value in env.items():
        if name.startswith("DASK_"):
            varname = name[5:].lower().replace("__", ".")
            try:
                d[varname] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                d[varname] = value
    result = {}
    set(d, config=result)
    return result
def ensure_file(source, destination=None, comment=True):
    
    if destination is None:
        destination = PATH
    # destination is a file and already exists, never overwrite
    if os.path.isfile(destination):
        return
    # If destination is not an existing file, interpret as a directory,
    # use the source basename as the filename
    directory = destination
    destination = os.path.join(directory, os.path.basename(source))
    try:
        if not os.path.exists(destination):
            os.makedirs(directory, exist_ok=True)
            # Atomically create destination.  Parallel testing discovered
            # a race condition where a process can be busy creating the
            # destination while another process reads an empty config file.
            tmp = "%s.tmp.%d" % (destination, os.getpid())
            with open(source) as f:
                lines = list(f)
            if comment:
                lines = [
                    "# " + line if line.strip() and not line.startswith("#") else line
                    for line in lines
                ]
            with open(tmp, "w") as f:
                f.write("".join(lines))
            try:
                os.rename(tmp, destination)
            except OSError:
                os.remove(tmp)
    except (IOError, OSError):
        pass
class set(object):
    
    def __init__(self, arg=None, config=config, lock=config_lock, **kwargs):
        with lock:
            self.config = config
            self._record = []
            if arg is not None:
                for key, value in arg.items():
                    self._assign(key.split("."), value, config)
            if kwargs:
                for key, value in kwargs.items():
                    self._assign(key.split("__"), value, config)
    def __enter__(self):
        return self.config
    def __exit__(self, type, value, traceback):
        for op, path, value in reversed(self._record):
            d = self.config
            if op == "replace":
                for key in path[:-1]:
                    d = d.setdefault(key, {})
                d[path[-1]] = value
            else:  # insert
                for key in path[:-1]:
                    try:
                        d = d[key]
                    except KeyError:
                        break
                else:
                    d.pop(path[-1], None)
    def _assign(self, keys, value, d, path=(), record=True):
        
        key = canonical_name(keys[0], d)
        path = path + (key,)
        if len(keys) == 1:
            if record:
                if key in d:
                    self._record.append(("replace", path, d[key]))
                else:
                    self._record.append(("insert", path, None))
            d[key] = value
        else:
            if key not in d:
                if record:
                    self._record.append(("insert", path, None))
                d[key] = {}
                # No need to record subsequent operations after an insert
                record = False
            self._assign(keys[1:], value, d[key], path, record=record)
def collect(paths=paths, env=None):
    
    if env is None:
        env = os.environ
    configs = []
    if yaml:
        configs.extend(collect_yaml(paths=paths))
    configs.append(collect_env(env=env))
    return merge(*configs)
def refresh(config=config, defaults=defaults, **kwargs):
    
    config.clear()
    for d in defaults:
        update(config, d, priority="old")
    update(config, collect(**kwargs))
def get(key, default=no_default, config=config):
    
    keys = key.split(".")
    result = config
    for k in keys:
        k = canonical_name(k, result)
        try:
            result = result[k]
        except (TypeError, IndexError, KeyError):
            if default is not no_default:
                return default
            else:
                raise
    return result
def rename(aliases, config=config):
    
    old = []
    new = {}
    for o, n in aliases.items():
        value = get(o, None, config=config)
        if value is not None:
            old.append(o)
            new[n] = value
    for k in old:
        del config[canonical_name(k, config)]  # TODO: support nested keys
    set(new, config=config)
def update_defaults(new, config=config, defaults=defaults):
    
    defaults.append(new)
    update(config, new, priority="old")
def expand_environment_variables(config):
    
    if isinstance(config, Mapping):
        return {k: expand_environment_variables(v) for k, v in config.items()}
    elif isinstance(config, str):
        return os.path.expandvars(config)
    elif isinstance(config, (list, tuple, builtins.set)):
        return type(config)([expand_environment_variables(v) for v in config])
    else:
        return config
refresh()
if yaml:
    fn = os.path.join(os.path.dirname(__file__), "dask.yaml")
    ensure_file(source=fn)
    with open(fn) as f:
        _defaults = yaml.safe_load(f)
    update_defaults(_defaults)
    del fn, _defaults
