import numpy as np
from typing import Any

from .pyfefi_kernel import coords

class Coordinates:

    def __init__(self,
            config,
            idx: int = 0,
            version: int = 2,
            dtype: Any = np.float32):
        """
        Initializes the Coordinates object.

        Args:
            config: The configuration object.
            idx (int, optional): The diagnostic index. Defaults to 0.
            version (int, optional): The version of the coordinates. Defaults to 2.
            dtype (Any, optional): The data type. Defaults to np.float32.
        """
        cid = config['input_parameters']['coordinate']
        grid_type = config['changeparameters'][self._key_name(idx, 'idiagbox')] % 10

        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise RuntimeError(f'`dtype` must be a float type')
        self.dtype = dtype

        if version <= 1 or grid_type != 0:
            self.grid_size = np.array([
                config['changeparameters'][self._key_name(idx, 'nxd')],
                config['changeparameters'][self._key_name(idx, 'nyd')],
                config['changeparameters'][self._key_name(idx, 'nzd')]
            ])
            self.limits = np.array(config['changeparameters'][self._key_name(idx, 'ddomain')], dtype=self.dtype).reshape(3, 2)
        else:
            self.grid_size = np.array([
                config['input_parameters']['nx'] + 1,
                config['input_parameters']['ny'] + 1,
                config['input_parameters']['nz'] + 1
            ])
            self.limits = np.array(config['input_parameters']['domain'], dtype=self.dtype).reshape(3, 2)

        self.cuv_limits = self.limits
        self.can_cache = True
        if grid_type == 1:
            self._init_cartesian(config)
            self.prefer_cache = False
            self.can_cache = False
        else:
            match cid:
                case -13:
                    self._init_cartesian_mod(config)
                    self.prefer_cache = False
                case 16:
                    self._init_spehere_mod(config)
                    self.prefer_cache = True
                case _:
                    raise NotImplementedError(f'Coordinate of type {cid} is not implemented')

    def _key_name(self, idx: int, name: str):
        if idx > 0:
            return name + str(int(idx))
        return name

    def _init_cartesian(self, config):
        self.is_coords_independent = True
        def to_cartesian(p, q, w):
            return p, q, w
        def from_cartesian(x, y, z):
            return x, y, z
        self.to_cartesian = to_cartesian
        self.from_cartesian = from_cartesian

    def _bind_kernel(self, kernel):
        def to_cartesian(p, q, w):
            p, q, w = np.broadcast_arrays(p, q, w)
            return kernel.to_cartesian(p, q, w)
        def from_cartesian(x, y, z):
            x, y, z = np.broadcast_arrays(x, y, z)
            return kernel.from_cartesian(x, y, z)
        self.to_cartesian = to_cartesian
        self.from_cartesian = from_cartesian

    def _init_cartesian_mod(self, config):
        deltas = np.array([
            config['input_parameters']['dx0'],
            config['input_parameters']['dy0'],
            config['input_parameters']['dz0']
        ], dtype=self.dtype)

        if self.dtype.itemsize == 64:
            cls = coords.CartesianMod
        else:
            cls = coords.CartesianModf

        coord_args = config.get('pyfefiparams', {}).get('coord_args', None)
        if coord_args is None:
            kernel = cls(deltas, self.limits)
        else:
            kernel = cls(deltas, self.limits, np.array(coord_args).reshape(6, 4))

        self._bind_kernel(kernel)
        self.grid_size = np.array(kernel.grid_sizes())

        self.cuv_limits = np.array([(0, self.grid_size[0]),
                                    (0, self.grid_size[1]),
                                    (0, self.grid_size[2])])

    def _init_spehere_mod(self, config):
        if self.dtype.itemsize == 64:
            cls = coords.SphereMod
        else:
            cls = coords.SphereModf
        kernel = cls()
        self._bind_kernel(kernel)

        plim = [kernel.solve_grid(self.limits[0][0]), kernel.solve_grid(self.limits[0][1])]
        qlim = [lim*np.pi/180 for lim in self.limits[1]]
        wlim = [lim*np.pi/180 for lim in self.limits[2]]
        self.cuv_limits = np.array([plim, qlim, wlim])

    def slice_size(self, slices):
        """
        Calculates the size of a slice.

        Args:
            slices: The slice object.

        Returns:
            A list of the dimensions of the slice.
        """
        if slices == Ellipsis:
            slices = (slice(None),) * 3

        result = []
        for slc, n in zip(slices, self.grid_size):
            if np.isscalar(slc):
                result.append(1)
            else:
                s, e, st = slc.start, slc.stop, slc.step
                if s is None:
                    s = 0
                if e is None:
                    e = n
                if st is None:
                    st = 1
                result.append((e - s) // st)
        return result

    def convert(self, slc):
        """
        Converts a slice to be within the grid boundaries.

        Args:
            slc: The slice object.

        Returns:
            A tuple of slice objects.
        """
        if slc == Ellipsis:
            slc = (slice(None),) * 3
        result = []
        for s, n in zip(slc, self.grid_size):
            if np.isscalar(s):
                result.append(s)
            else:
                result.append(slice(
                    max(s.start, 0) if s.start is not None else 0,
                    min(s.stop, n) if s.stop is not None else n,
                    s.step))
        return tuple(result)
