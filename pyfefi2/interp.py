import numpy as np

from .pyfefi_kernel import interp as interp_kernel
from .pyfefi_kernel import quick_stack

def _stack_last_trans(arrs):
    if len(arrs) == 1:
        return arrs[0]
    arrs = quick_stack(arrs)
    indices = list(range(1, arrs.ndim))
    indices.append(0)
    return np.transpose(arrs, indices)

def interp(coords_to, coords_from, var_list, order=2):
    """
    Interpolates data to new coordinates.

    Args:
        coords_to: The coordinates to interpolate to.
        coords_from: The original coordinates of the data.
        var_list: A list of variables to interpolate.
        order (int, optional): The order of interpolation. Defaults to 2.

    Returns:
        A list of interpolated variables.
    """
    var_shape = tuple(len(c) for c in coords_from)

    coords_to = tuple(coords_to)
    coords_from = tuple(coords_from)
    if len(coords_to) != len(coords_from):
        raise ValueError('Different number of coordinates provided for `coords_to` and `coords_from`')
    c_shape = coords_to[0].shape
    for c in coords_to[1:]:
        if c.shape != c_shape:
            raise ValueError('Shape of elements in `coords_to` must be the same')

    def convert(var):
        shape = var.shape
        if var.shape[:len(coords_from)] != var_shape:
            raise ValueError('Shape mismatch for interp data and coordinates')
        var = var.reshape(var.shape[:len(coords_from)] + (-1,))
        var_list = [var[..., i] for i in range(var.shape[len(coords_from)])]
        return var_list, shape

    if isinstance(var_list, np.ndarray):
        return_one = True
        var_list, shape = convert(var_list)
        shapes = [shape]
        index_ranges = [(0, len(var_list))]
    else:
        return_one = False
        _vl = []
        shapes = []
        index_ranges = []
        for var in var_list:
            vl, shape = convert(var)
            shapes.append(shape)
            index_ranges.append((len(_vl), len(_vl)+len(vl)))
            _vl.extend(vl)
        var_list = _vl

    dim = len(coords_to)
    dtype = coords_to[0].dtype
    if dtype == np.float64:
        dchar = 'd'
    else:
        dchar = 'f'
    fn_name = f'interp{dchar}{dim}d{order}'
    results = getattr(interp_kernel, fn_name)(coords_to, coords_from, var_list)

    ret = []
    for shape, index_range in zip(shapes, index_ranges):
        ret.append(
            _stack_last_trans(results[index_range[0]:index_range[1]])
        )
    if return_one:
        return ret[0]
    else:
        return ret
