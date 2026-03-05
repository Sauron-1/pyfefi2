from abc import abstractmethod
import os
import netCDF4
import requests
import numpy as np

from .pyfefi_kernel import compress
from .datafolder import DataFolder, Auth

def encode_index(items):
    """Encodes a tuple of slices/integers into a string."""
    if isinstance(items, int):
        items = (items,)
    encoded = []
    for item in items:
        if isinstance(item, slice):
            # Format: start:stop:step (None is kept as empty)
            parts = [str(p) if p is not None else "" for p in (item.start, item.stop, item.step)]
            encoded.append(":".join(parts))
        else:
            encoded.append(str(item))
    return ",".join(encoded)

class Dataset:

    @abstractmethod
    def has_array(self, name):
        raise NotImplementedError()

    @abstractmethod
    def get(self, name, slices):
        raise NotImplementedError()

    def __getitem__(self, args):
        name, slices = args
        if not self.has_array(name):
            raise KeyError(f"Variable {name} not found in dataset")
        return self.get(name, slices)

class NcDataset(Dataset):

    def __init__(self, fn):
        self.ds = netCDF4.Dataset(fn)

    def has_array(self, name):
        return name in self.ds.variables

    def get(self, name, slices):
        return self.ds.variables[name][slices].filled()


class SZ3Dataset(Dataset):

    def __init__(self, fn):
        self.ds = compress.CompressedFile(fn, mode='r')

    def has_array(self, name):
        return self.ds.has_array(name)

    def get(self, name, slices):
        return self.ds[name][slices]

class HttpDataset(Dataset):

    def __init__(self, url: str, auth: Auth):
        self.url = url
        self.auth = auth

    def has_array(self, name):
        resp = requests.get(
            self.url,
            headers = self.auth.get_headers(),
            params = dict(q='has', name=name)
        )
        return resp.text == 'T'

    def get(self, name, slices):
        resp = requests.get(
            self.url,
            headers = self.auth.get_headers(),
            params = dict(
                q = 'get',
                name = name,
                slc = encode_index(slices)
            )
        )
        if name == 'param':
            result = np.frombuffer(resp.content, dtype=np.float32)
            if np.isscalar(slices):
                return result[0]
        if len(resp.content) <= 8: # 0-d array
            return np.frombuffer(resp.content, dtype=np.float32)[0]
        return compress.decompress_array(resp.content)

def open_dataset(folder: DataFolder, filename: str, allow_remote=True):
    if folder.type() == 'local':
        path = folder.path
        data_fn = os.path.join(path, filename + '.sz3')
        if os.path.exists(data_fn):
            return SZ3Dataset(data_fn)
        else:
            data_fn = os.path.join(path, filename + '.nc')
            return NcDataset(data_fn)
    else:
        if not allow_remote:
            raise RuntimeError("Opening remote dataset is not allowed")
        url = folder.path + filename
        return HttpDataset(url, folder.auth)
