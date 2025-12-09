import f90nml
import hashlib
import os
import warnings
from typing import Optional, Any
import numpy as np
from scipy import constants as C

from .coords import Coordinates
from .units import Units

class Config:

    def __init__(
            self,
            path: str,
            diag_idx: int = 0,
            data_type: str = "float32",
            cache_folder: Optional[str]=None,
            Re: Optional[float] = None):
        """
        Initializes the configuration for a simulation.

        Args:
            path (str): The path to the simulation data folder.
            diag_idx (int, optional): The diagnostic index. Defaults to 0.
            data_type (str, optional): The data type to be used for arrays. Defaults to "float32".
            cache_folder (Optional[str], optional): The folder to store cache files. Defaults to None.
            Re (float, optional): The Earth radius in meters. Defaults to 6371e3.
        """
        if os.path.isfile(path):
            raise RuntimeError(f'`path` must be a folder, got {path}')

        self.path = path

        conf_file = os.path.join(path, 'fefi.input')
        if not os.path.exists(conf_file):
            raise RuntimeError(f'{path} does not contain file `fefi.input`.')
        self._conf = f90nml.read(conf_file)

        self.__cache_folder_set = False
        self._cache_folder = cache_folder

        self.diag_idx = diag_idx
        self.dtype = np.dtype(data_type)
        self.coordinates = Coordinates(self, idx=self.diag_idx, dtype=self.dtype)
        self.grid_size = self.coordinates.grid_size

        self._Re = Re
        self._prefix = None

    def pqw(self, slices=None):
        """
        Returns the p, q, and w coordinates.

        Args:
            slices (optional): Slices for the coordinates. Defaults to None.

        Returns:
            tuple: A tuple containing the p, q, and w coordinates.
        """
        lims = self.coordinates.cuv_limits
        return tuple(np.linspace(*lims[i], self.grid_size[i], dtype=self.dtype) for i in range(3))

    def get_units(self, scaled=True, Re=None):
        """
        Returns a `Units` object.

        Args:
            scaled (bool, optional): Whether to use scaled units. Defaults to True.
            Re (float, optional): The Earth radius in meters. Defaults to None.

        Returns:
            Units: A `Units` object.
        """
        if Re is None:
            Re = self.Re
        return Units(self.path, scaled, Re)

    def _calc_run_id(self, path: str) -> str:
        abs_path = os.path.abspath(path)
        run_id = hashlib.sha256(abs_path.encode('utf-8')).hexdigest()[:8]
        return run_id

    def get_run_id(self, path: str) -> str:
        """
        Returns the run ID for the simulation.

        Args:
            path (str): The path to the simulation data folder.

        Returns:
            str: The run ID.
        """
        id_file = os.path.join(path, 'runid')
        if os.path.exists(id_file):
            with open(id_file, 'r') as f:
                return f.read()
        else:
            run_id = self._calc_run_id(path)
            try:
                with open(id_file, 'w') as f:
                    f.write(run_id)
            except (IOError, OSError) as e:
                warnings.warn(f'RunID file not written. Failed to write {id_file}: {e}')
            return run_id

    @property
    def cache_folder(self) -> Optional[str]:
        if not self.__cache_folder_set:
            cache_folder = self._cache_folder
            if cache_folder is None:
                cache_folder = os.environ.get('PYFEFI_CACHE_FOLDER', None)
            if cache_folder is not None:
                run_id = self.get_run_id(os.path.abspath(self.path))
                real_cache_folder = os.path.join(cache_folder, run_id)
                os.makedirs(real_cache_folder, exist_ok=True)
            self._cache_folder = real_cache_folder
            self.__cache_folder_set = True

        return self._cache_folder

    def cache_file(self, name: str) -> Optional[str]:
        """
        Returns the path to a cache file.

        Args:
            name (str): The name of the cache file.

        Returns:
            Optional[str]: The path to the cache file, or None if no cache folder is specified.
        """

        if self.cache_folder is None:
            raise RuntimeError('No cache folder specified. Please pass the path to the constructor, or set "PYFEFI_CACHE_FOLDER" environment variable.')
        return os.path.join(self.cache_folder, name)

    def __getitem__(self, key: str) -> Any:
        return self._conf[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._conf.get(key, default)

    def get_prefix(self):
        """
        Returns the file prefix.

        Returns:
            str: The file prefix.
        """
        if self._prefix is not None:
            return self._prefix

        prefixes = ['fieldds', 'fieldns', 'fieldmp', 'fieldeq']
        prefix = prefixes[self.diag_idx]

        if self.diag_idx > 0:
            self._prefix = prefix
        elif os.path.exists(os.path.join(self.path, prefix + '%05d.nc' % 1)):
            if self._Re is None:
                self._Re = 2440e3
            self._prefix = prefix
        elif os.path.exists(os.path.join(self.path, 'field' + '%05d.nc' % 1)):
            if self._Re is None:
                self._Re = 6371e3
            self._prefix = 'field'
        else:
            # pass two: list all files
            fns = os.listdir(self.path)
            for fn in fns:
                if fn.startswith(prefix):
                    self._Re = 2440e3
                    self._prefix = prefix
                    break
            else:
                self._Re = 6371e3
                self._prefix = 'field'
        return self._prefix

    @property
    def Re(self):
        if self._Re is not None:
            return self._Re
        self.get_prefix()
        if self._Re is None:
            raise RuntimeError('Cannot automatically detect Re. Provide it when construct the config object')
        return self._Re

    @property
    def mu0(self):
        if not hasattr(self, '_mu0'):
            R_lambda = self['input_parameters']['roverL']
            self._mu0 = R_lambda**2 * C.mu_0
        return self._mu0

    def minimum_slice(self, xs, ys=None, zs=None, extra=None):
        """
        Calculates the minimum slice required to cover the given coordinates.

        Args:
            xs: The x-coordinates.
            ys (optional): The y-coordinates. Defaults to None.
            zs (optional): The z-coordinates. Defaults to None.
            extra (optional): Extra buffer to add to the slice. Defaults to None.

        Returns:
            tuple: A tuple of slice objects.
        """
        if zs is None:
            xs, ys, zs = xs
            if ys is not None and extra is None:
                extra = ys
        if extra is None:
            extra = 0

        xs = np.array(xs, dtype=self.dtype)
        ys = np.array(ys, dtype=self.dtype)
        zs = np.array(zs, dtype=self.dtype)
        pqw = self.coordinates.from_cartesian(xs, ys, zs)
        lims = tuple((c.min(), c.max()) for c in pqw)
        idx_lims = (
                (np.searchsorted(c, l[0], side='left') - 1,
                 np.searchsorted(c, l[1]))
                for c, l in zip(self.pqw(), lims)
            )
        lims = [(max(lim[0]-extra, 0), min(lim[1]+extra, gs))
                for lim, gs in zip(idx_lims, self.grid_size)]
        return tuple(slice(s, e) for s, e in lims)
