import re
class Engine:
    
    @staticmethod
    def read_metadata(
        fs,
        paths,
        categories=None,
        index=None,
        gather_statistics=None,
        filters=None,
        **kwargs
    ):
        
        raise NotImplementedError()
    @staticmethod
    def read_partition(fs, piece, columns, index, **kwargs):
        
        raise NotImplementedError()
    @staticmethod
    def initialize_write(
        df,
        fs,
        path,
        append=False,
        partition_on=None,
        ignore_divisions=False,
        division_info=None,
        **kwargs
    ):
        
        raise NotImplementedError
    @staticmethod
    def write_partition(
        df, path, fs, filename, partition_on, return_metadata, **kwargs
    ):
        
        raise NotImplementedError
    @staticmethod
    def write_metadata(parts, meta, fs, path, append=False, **kwargs):
        
        raise NotImplementedError()
def _parse_pandas_metadata(pandas_metadata):
    
    index_storage_names = [
        n["name"] if isinstance(n, dict) else n
        for n in pandas_metadata["index_columns"]
    ]
    index_name_xpr = re.compile(r"__index_level_\d+__")
    # older metadatas will not have a 'field_name' field so we fall back
    # to the 'name' field
    pairs = [
        (x.get("field_name", x["name"]), x["name"]) for x in pandas_metadata["columns"]
    ]
    # Need to reconcile storage and real names. These will differ for
    # pyarrow, which uses __index_leveL_d__ for the storage name of indexes.
    # The real name may be None (e.g. `df.index.name` is None).
    pairs2 = []
    for storage_name, real_name in pairs:
        if real_name and index_name_xpr.match(real_name):
            real_name = None
        pairs2.append((storage_name, real_name))
    index_names = [name for (storage_name, name) in pairs2 if name != storage_name]
    # column_indexes represents df.columns.name
    # It was added to the spec after pandas 0.21.0+, and implemented
    # in PyArrow 0.8. It was added to fastparquet in 0.3.1.
    column_index_names = pandas_metadata.get("column_indexes", [{"name": None}])
    column_index_names = [x["name"] for x in column_index_names]
    # Now we need to disambiguate between columns and index names. PyArrow
    # 0.8.0+ allows for duplicates between df.index.names and df.columns
    if not index_names:
        # For PyArrow < 0.8, Any fastparquet. This relies on the facts that
        # 1. Those versions used the real index name as the index storage name
        # 2. Those versions did not allow for duplicate index / column names
        # So we know that if a name is in index_storage_names, it must be an
        # index name
        if index_storage_names and isinstance(index_storage_names[0], dict):
            # Cannot handle dictionary case
            index_storage_names = []
        index_names = list(index_storage_names)  # make a copy
        index_storage_names2 = set(index_storage_names)
        column_names = [
            name for (storage_name, name) in pairs if name not in index_storage_names2
        ]
    else:
        # For newer PyArrows the storage names differ from the index names
        # iff it's an index level. Though this is a fragile assumption for
        # other systems...
        column_names = [name for (storage_name, name) in pairs2 if name == storage_name]
    storage_name_mapping = dict(pairs2)  # TODO: handle duplicates gracefully
    return index_names, column_names, storage_name_mapping, column_index_names
def _normalize_index_columns(user_columns, data_columns, user_index, data_index):
    
    specified_columns = user_columns is not None
    specified_index = user_index is not None
    if user_columns is None:
        user_columns = list(data_columns)
    elif isinstance(user_columns, str):
        user_columns = [user_columns]
    else:
        user_columns = list(user_columns)
    if user_index is None:
        user_index = data_index
    elif user_index is False:
        # When index is False, use no index and all fields should be treated as
        # columns (unless `columns` provided).
        user_index = []
        data_columns = data_index + data_columns
    elif isinstance(user_index, str):
        user_index = [user_index]
    else:
        user_index = list(user_index)
    if specified_index and not specified_columns:
        # Only `index` provided. Use specified index, and all column fields
        # that weren't specified as indices
        index_names = user_index
        column_names = [x for x in data_columns if x not in index_names]
    elif specified_columns and not specified_index:
        # Only `columns` provided. Use specified columns, and all index fields
        # that weren't specified as columns
        column_names = user_columns
        index_names = [x for x in data_index if x not in column_names]
    elif specified_index and specified_columns:
        # Both `index` and `columns` provided. Use as specified, but error if
        # they intersect.
        column_names = user_columns
        index_names = user_index
        if set(column_names).intersection(index_names):
            raise ValueError("Specified index and column names must not intersect")
    else:
        # Use default columns and index from the metadata
        column_names = data_columns
        index_names = data_index
    return column_names, index_names
def _analyze_paths(file_list, fs, root=False):
    
    def _join_path(*path):
        def _scrub(i, p):
            # Convert path to standard form
            # this means windows path separators are converted to linux
            p = p.replace(fs.sep, "/")
            if p == "":  # empty path is assumed to be a relative path
                return "."
            if p[-1] == "/":  # trailing slashes are not allowed
                p = p[:-1]
            if i > 0 and p[0] == "/":  # only the first path can start with /
                p = p[1:]
            return p
        abs_prefix = ""
        if path and path[0]:
            if path[0][0] == "/":
                abs_prefix = "/"
                path = list(path)
                path[0] = path[0][1:]
            elif fs.sep == "\\" and path[0][1:].startswith(":/"):
                # If windows, then look for the "c:/" prefix
                abs_prefix = path[0][0:3]
                path = list(path)
                path[0] = path[0][3:]
        _scrubbed = []
        for i, p in enumerate(path):
            _scrubbed.extend(_scrub(i, p).split("/"))
        simpler = []
        for s in _scrubbed:
            if s == ".":
                pass
            elif s == "..":
                if simpler:
                    if simpler[-1] == "..":
                        simpler.append(s)
                    else:
                        simpler.pop()
                elif abs_prefix:
                    raise Exception("can not get parent of root")
                else:
                    simpler.append(s)
            else:
                simpler.append(s)
        if not simpler:
            if abs_prefix:
                joined = abs_prefix
            else:
                joined = "."
        else:
            joined = abs_prefix + ("/".join(simpler))
        return joined
    path_parts_list = [_join_path(fn).split("/") for fn in file_list]
    if root is False:
        basepath = path_parts_list[0][:-1]
        for i, path_parts in enumerate(path_parts_list):
            j = len(path_parts) - 1
            for k, (base_part, path_part) in enumerate(zip(basepath, path_parts)):
                if base_part != path_part:
                    j = k
                    break
            basepath = basepath[:j]
        l = len(basepath)
    else:
        basepath = _join_path(root).split("/")
        l = len(basepath)
        assert all(
            p[:l] == basepath for p in path_parts_list
        ), "All paths must begin with the given root"
    l = len(basepath)
    out_list = []
    for path_parts in path_parts_list:
        out_list.append(
            "/".join(path_parts[l:])
        )  # use '/'.join() instead of _join_path to be consistent with split('/')
    return (
        "/".join(basepath),
        out_list,
    )  # use '/'.join() instead of _join_path to be consistent with split('/')
