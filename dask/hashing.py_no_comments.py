import binascii
import hashlib
hashers = []  # In decreasing performance order
# Timings on a largish array:
# - CityHash is 2x faster than MurmurHash
# - xxHash is slightly slower than CityHash
# - MurmurHash is 8x faster than SHA1
# - SHA1 is significantly faster than all other hashlib algorithms
try:
    import cityhash  # `pip install cityhash`
except ImportError:
    pass
else:
    # CityHash disabled unless the reference leak in
    # https://github.com/escherba/python-cityhash/pull/16
    # is fixed.
    if cityhash.__version__ >= "0.2.2":
        def _hash_cityhash(buf):
            
            h = cityhash.CityHash128(buf)
            return h.to_bytes(16, "little")
        hashers.append(_hash_cityhash)
try:
    import xxhash  # `pip install xxhash`
except ImportError:
    pass
else:
    def _hash_xxhash(buf):
        
        return xxhash.xxh64(buf).digest()
    hashers.append(_hash_xxhash)
try:
    import mmh3  # `pip install mmh3`
except ImportError:
    pass
else:
    def _hash_murmurhash(buf):
        
        return mmh3.hash_bytes(buf)
    hashers.append(_hash_murmurhash)
def _hash_sha1(buf):
    
    return hashlib.sha1(buf).digest()
hashers.append(_hash_sha1)
def hash_buffer(buf, hasher=None):
    
    if hasher is not None:
        try:
            return hasher(buf)
        except (TypeError, OverflowError):
            # Some hash libraries may have overly-strict type checking,
            # not accepting all buffers
            pass
    for hasher in hashers:
        try:
            return hasher(buf)
        except (TypeError, OverflowError):
            pass
    raise TypeError("unsupported type for hashing: %s" % (type(buf),))
def hash_buffer_hex(buf, hasher=None):
    
    h = hash_buffer(buf, hasher)
    s = binascii.b2a_hex(h)
    return s.decode()
