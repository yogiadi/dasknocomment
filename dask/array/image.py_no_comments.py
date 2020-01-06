from glob import glob
import os
try:
    from skimage.io import imread as sk_imread
except (AttributeError, ImportError):
    pass
from .core import Array
from ..base import tokenize
def add_leading_dimension(x):
    return x[None, ...]
def imread(filename, imread=None, preprocess=None):
    
    imread = imread or sk_imread
    filenames = sorted(glob(filename))
    if not filenames:
        raise ValueError("No files found under name %s" % filename)
    name = "imread-%s" % tokenize(filenames, map(os.path.getmtime, filenames))
    sample = imread(filenames[0])
    if preprocess:
        sample = preprocess(sample)
    keys = [(name, i) + (0,) * len(sample.shape) for i in range(len(filenames))]
    if preprocess:
        values = [
            (add_leading_dimension, (preprocess, (imread, fn))) for fn in filenames
        ]
    else:
        values = [(add_leading_dimension, (imread, fn)) for fn in filenames]
    dsk = dict(zip(keys, values))
    chunks = ((1,) * len(filenames),) + tuple((d,) for d in sample.shape)
    return Array(dsk, name, chunks, sample.dtype)
