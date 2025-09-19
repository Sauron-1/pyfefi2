import numpy as np
import f90nml

class _Slice:
    """
    Proxy class for slicing ndarray
    """

    def convert_slice(self, s):
        """
        Convert a slice object to on that keep dims.
        """
        if np.isscalar(s):
            return slice(s, s+1)
        else:
            return s

    def __getitem__(self, args):
        """
        Return a `slice` object, which can be used to slice a np.ndarray.
        The difference is, the returned object will always keep dimension
        of the original ndarray.
        """
        #return tuple(self.convert_slice(s) for s in args)
        return args

    def squeezer(cls, slc):
        result = []
        for s in slc:
            if np.isscalar(s):
                result.append(0)
            else:
                result.append(slice(None))
        return tuple(result)

    def extender(cls, slc):
        result = []
        for s in slc:
            if np.isscalar(s):
                result.append(None)
            else:
                result.append(slice(None))
        return tuple(result)


class SlcTranspose:

    def __init__(self, slc):
        self.slc = slc

    def transpose(self, arr, *args):
        slc = self.slc
        if slc == Ellipsis:
            slc = (slice(None),) * len(args)
        if not isinstance(slc, tuple):
            slc = (slc,)
        slc = (slice(None),) * (len(args) - len(slc)) + slc

        fwd_slc = Slice.extender(slc)
        bwd_slc = Slice.squeezer(slc)
        bwd_slc = tuple(bwd_slc[i] for i in args)

        return arr[fwd_slc].transpose(*args)[bwd_slc]

    def __call__(self, arr, *args):
        return self.transpose(arr, *args)

Slice = _Slice()
