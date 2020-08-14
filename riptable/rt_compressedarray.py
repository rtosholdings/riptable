import numpy as np
from .rt_fastarray import FastArray


class CompressedArray(FastArray):

    allowed_funcs = ['decompress', 'view']

    def __new__(cls, arr):
        if isinstance(arr, np.ndarray):
            arr.setflags(write=False)
        else:
            raise TypeError(f"Compressed array must be constructed from numpy array, not {type(arr)}")
        return arr.view(cls)

    def __init__(self, arr):
        pass

    def __getattribute__(self, attr):
        '''
        Block all FastArray operations. See allowed_funcs class global.
        '''
        if hasattr(FastArray, attr) and attr.startswith('_') is False and attr not in self.__class__.allowed_funcs:
            raise ValueError(f"{attr} cannot be accessed from compressed array.")
        return object.__getattribute__(self, attr)

    def decompress(self):
        pass

    def __repr__(self):
        return 'test repr'

    def __str__(self):
        return 'test str'

    def _build_string(self):
        # translate header information
        pass
