from abc import abstractmethod
import os
import netCDF4
from .pyfefi_kernel import compress

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


def open_dataset(path, filename):
    data_fn = os.path.join(path, filename + '.sz3')
    if os.path.exists(data_fn):
        return SZ3Dataset(data_fn)
    else:
        data_fn = os.path.join(path, filename + '.nc')
        return NcDataset(data_fn)
