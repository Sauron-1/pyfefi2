import numpy as np
import netCDF4
import math
import f90nml
import os
from scipy import constants as C
from collections import OrderedDict
import psutil
import numba as nb
import time as py_time

from typing import Union, Tuple, Optional, Dict

from .slices import SlcTranspose
from .config import Config
from .interp import interp
from .pyfefi_kernel import quick_stack

def timeit(fn):
    """
    For benchmarking.
    """
    def wrapper(*args, **kwargs):
        t0 = py_time.perf_counter()
        res = fn(*args, **kwargs)
        print("Time elapsed by {}: {:.3f} s".format(fn.__name__, py_time.perf_counter() - t0))
        return res
    return wrapper

class RegisteredFunction:

    def __init__(self, fn):
        self._var_names = fn.__code__.co_varnames
        fn_sigs = [dtype(*((dtype,)*len(self._var_names))) for dtype in [nb.f4, nb.f8]]
        if os.environ.get('PYFEFI_PERF', None) is not None:
            self._fn = timeit(nb.vectorize(fn_sigs, target='parallel')(fn))
        else:
            self._fn = nb.vectorize(fn_sigs, target='parallel')(fn)

    def __call__(self, obj, frame):
        var_list = [obj[frame, name] for name in self._var_names]
        return self._fn(*var_list)

class Data:

    registered_fns : Dict[str, RegisteredFunction] = {}

    def __init__(
            self,
            cfg: Union[Config, str],
            slices = None,
            max_memory: float = 0,
            use_cache: Optional[bool] = True,
            save_cache: Optional[bool] = False,
            max_open: int = 1):
        """
        Initializes the Data object for accessing simulation data.

        Args:
            cfg (Union[Config, str]): A `Config` object or a path to the configuration file.
            slices (optional): Slices for the data. Defaults to None.
            max_memory (float, optional): The maximum memory to use in bytes. Defaults to 1/3 of the system memory.
            use_cache (Optional[bool], optional): Whether to use cache. Defaults to True.
            save_cache (Optional[bool], optional): Whether to save cache. Defaults to False.
            max_open (int, optional): The maximum number of open files. Defaults to 1.
        """

        # initialize config object
        if isinstance(cfg, str):
            self.config = config.Config(cfg)
        else:
            self.config = cfg

        self.dtype = self.config.dtype
        self.coordinates = self.config.coordinates

        # initialize slices
        if slices is None or slices == Ellipsis:
            self.slices = tuple([slice(None)] * 3)
        else:
            self.slices = tuple(slices)
        self.slices = self.coordinates.convert(self.slices)
        self.slice_for_nc = tuple(self.slices[::-1])

        # initialize data path and cache
        self.path = self.config.path
        self.fn_prefix = self.config.get_prefix()
        self.trans_op = SlcTranspose(self.slice_for_nc)

        self.nx, self.ny, self.nz = self.coordinates.slice_size(self.slices)
        self.grid_size = (self.nx, self.ny, self.nz)
        if max_memory <= 0:
            max_memory = min(psutil.virtual_memory().total / 3, 2e10)
        self.max_memory = max_memory
        num_points = self.nx * self.ny * self.nz
        array_size = self.dtype.itemsize * num_points
        self.max_arrays = int(max_memory / array_size) - 3
        if self.max_arrays < 3:
            min_memory = array_size * 6
            raise ValueError(f'max_arrays must be greater than or equal to 1. Try increase `max_memory` to at least {min_memory}.')

        self.array_caches = OrderedDict()

        self.max_open = max_open
        self.ds_caches = OrderedDict()

        pqw = self.config.pqw()
        self.p = pqw[0][self.slices[0]]
        self.q = pqw[1][self.slices[1]]
        self.w = pqw[2][self.slices[2]]
        self.pqw = (self.p, self.q, self.w)

        self._coords_initialized = False
        if use_cache is None:
            use_cache = self.coordinates.prefer_cache
        self.use_cache = use_cache and self.coordinates.can_cache
        self.save_cache = save_cache and self.coordinates.can_cache

    def _calc_size(self):
        size = 0
        for k, v in self.array_caches.items():
            size += 1 if v.ndim == 3 else 3
        return size

    def clear(self):
        """Clears the array caches."""
        self.array_caches.clear()

    def _open(self, frame):
        if frame in self.ds_caches:
            self.ds_caches.move_to_end(frame, last=True)
            return self.ds_caches[frame]
        if len(self.ds_caches) >= self.max_open:
            self.ds_caches.popitem(last=False)

        data_fn = os.path.join(self.path, f'{self.fn_prefix}{frame:05d}.nc')
        if not os.path.exists(data_fn):
            raise RuntimeError(f"Data file {data_fn} does not exist.")
        ds = netCDF4.Dataset(data_fn)

        self.ds_caches[frame] = ds
        return ds

    def _load(self, frame, name):
        cache_key = f'{name}_{frame:05d}'
        if cache_key in self.array_caches:
            self.array_caches.move_to_end(cache_key, last=True)
            return self.array_caches[cache_key]
        
        if name in self.registered_fns:
            data = self.registered_fns[name](self, frame)
        else:
            if len(self.array_caches) >= self.max_arrays:
                self.array_caches.popitem(last=False)

            ds = self._open(frame)

            if not name in ds.variables:
                raise KeyError(f"Variable {name} not found in dataset")
            data = ds.variables[name][self.slice_for_nc].filled()
            data = self.trans_op(data, 2, 1, 0)

        self.array_caches[cache_key] = data
        return data
    
    def test_key(self, frame, name):
        """
        Tests if a variable exists in the data.

        Args:
            frame (int): The frame number.
            name (str): The name of the variable.

        Returns:
            bool: True if the variable exists, False otherwise.
        """
        cache_key = f'{name}_{frame:05d}'
        if cache_key in self.array_caches:
            return True
        if name in self.registered_fns:
            return True
        ds = self._open(frame)
        if name in ds.variables:
            return True
        return False

    def __getitem__(self, args) -> np.ndarray:
        frame, name = args
        data = None
        if name in ['mi', 'kB', 'e', 'mu0']:
            return 1
        if name in ['x', 'y', 'z']:
            return getattr(self, name)
        try:
            data = self._load(frame, name)
        except KeyError:
            data = [self._load(frame, name+ax_name) for ax_name in 'xyz']
            data = self.trans_op(quick_stack(data), 1, 2, 3, 0)
        return data

    def _calcuate_coordinates(self):
        P, Q, W = np.meshgrid(self.p, self.q, self.w, indexing='ij')
        x, y, z = self.coordinates.to_cartesian(P, Q, W)
        slcs = [slice(None), slice(None), slice(None)]
        for i, s in enumerate(self.slices):
            if np.isscalar(s):
                slcs[i] = 0
        slcs = tuple(slcs)
        return x[slcs], y[slcs], z[slcs]

    def _init_coordinates(self):
        if self._coords_initialized:
            return
        self._x, self._y, self._z = self._calcuate_coordinates()
        self._coords_initialized = True

    @property
    def x(self):
        self._init_coordinates()
        return self._x
    @property
    def y(self):
        self._init_coordinates()
        return self._y
    @property
    def z(self):
        self._init_coordinates()
        return self._z

    @classmethod
    def register(cls, name):
        def decorator(fn):
            reg_fn = RegisteredFunction(fn)
            cls.registered_fns[name] = reg_fn
            return fn
        return decorator

class InterpData:

    def __init__(self, data: Union[Data, Config], xs, ys=None, zs=None, max_memory=None, order=2):
        """
        Initializes the InterpData object for interpolating data.

        Args:
            data (Union[Data, Config]): A `Data` or `Config` object.
            xs: The x-coordinates for interpolation.
            ys (optional): The y-coordinates for interpolation. Defaults to None.
            zs (optional): The z-coordinates for interpolation. Defaults to None.
            max_memory (optional): The maximum memory to use in bytes. Defaults to None.
            order (int, optional): The order of interpolation. Defaults to 2.
        """
        if ys is None:
            xs, ys, zs = xs
        self.xs, self.ys, self.zs = xs, ys, zs
        self.interp_order = order

        if isinstance(data, Data):
            self.data = data
            self.config = data.config
        else:
            self.data = None
            self.config = data

        self._coords_initialized = False

        self.slices = tuple(0 if np.isscalar(c) else slice(None) for c in [xs, ys, zs])
        self.trans_op = SlcTranspose(self.slices)

        def get_len(c):
            if np.isscalar(c):
                return 1
            return len(c)
        array_size = get_len(xs)*get_len(ys)*get_len(zs) * data.dtype.itemsize
        if max_memory is None:
            if self.data is None:
                max_memory = min(psutil.virtual_memory().total / 3, 2e10)
            else:
                max_memory = self.data.max_memory
        self.max_memory = max_memory
        self.max_arrays = (max_memory / array_size) - 6
        self.array_caches = OrderedDict()

    def _init_coords(self):
        if self._coords_initialized:
            return

        x, y, z = np.meshgrid(self.xs, self.ys, self.zs, indexing='ij')
        self.x = x.copy('F')[self.slices]
        self.y = y.copy('F')[self.slices]
        self.z = z.copy('F')[self.slices]

        if self.data is None:
            slices = self.config.minimum_slice(self.x, self.y, self.z, extra=self.interp_order)
            self.data = Data(self.config, slices=slices, max_memory=self.max_memory)

        self.pqw = tuple(self.data.coordinates.from_cartesian(self.x, self.y, self.z))
        self.pqw_from = (self.data.p, self.data.q, self.data.w)

        self._coords_initialized = True

    def clear(self):
        self.array_caches.clear()

    def _load_cache(self, frame, name):
        cache_key = f'{name}_{frame:05d}'
        if cache_key in self.array_caches:
            self.array_caches.move_to_end(cache_key, last=True)
            return self.array_caches[cache_key]
        return None

    def _set_cache(self, frame, name, data):
        cache_key = f'{name}_{frame:05d}'
        if cache_key in self.array_caches:
            self.array_caches[cache_key] = data
            self.array_caches.move_to_end(cache_key, last=True)
        else:
            if len(self.array_caches) >= self.max_arrays:
                self.array_caches.popitem(last=False)
            self.array_caches[cache_key] = data

    def __getitem__(self, args):
        self._init_coords()
        frame, name = args

        if name in ['mi', 'kB', 'e', 'mu0']:
            return 1
        if name in ['x', 'y', 'z']:
            return getattr(self, name)

        if self.data.test_key(frame, name):
            cached = self._load_cache(frame, name)
            if cached is not None:
                return cached
            data = self.data[frame, name]
            data = interp(self.pqw, self.pqw_from, data, order=self.interp_order)
            self._set_cache(frame, name, data)
            return data
        if self.data.test_key(frame, name + 'x'):
            names = [name + ax_name for ax_name in 'xyz']
            cached = [self._load_cache(frame, n) for n in names]
            interp_query = []
            for n, c in zip(names, cached):
                if c is None:
                    interp_query.append(self.data[frame, n])
            interp_data = interp(self.pqw, self.pqw_from, interp_query, order=self.interp_order)
            for i in range(3):
                if cached[i] is None:
                    data = interp_data.pop(0)
                    self._set_cache(frame, names[i], data)
                    cached[i] = data
            return self.trans_op(quick_stack(cached), 1, 2, 3, 0)
