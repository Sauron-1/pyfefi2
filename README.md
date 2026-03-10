# PyFefi2
Post processor for Prof. Wang's fefi code, mainly used for global simulations. This lib is meant to work with curvilinear netCDF outputs, especially the pnetcdf version.

## Building
A C++20 compatible compiler with OpenMP is required. The library also requires [pybind11](https://github.com/pybind/pybind11) and [xsimd](https://github.com/xtensor-stack/xsimd) to build. You can manually install the libraries so that cmake can find them, or clone the Git repos into "thirdparty" folder, or let cmake to download them from GitHub.

## Usage

### Basic usage

Here is a simple example code:

```python
import numpy as np
import matplotlib.pyplot as plt

from pyfefi2 import Data, Config, Slice, InterpData

path = '/path/to/fefi.input'
cfg = Config(path)
units = cfg.get_units()
frame = 400

def example1():
    # If you'd like (Nx, 1, Nz) arrays, use Slice[:, [cfg.grid_size[1]//2], :].
    slc = Slice[:, cfg.grid_size[1]//2, :]

    data = Data(cfg, slices=slc)

    # The read data will be cached in memory.
    # Default maximun cache size is 1/3 of the system RAM.
    Ni = data[frame, 'Ni'] * units.N

    plt.figure()
    plt.pcolormesh(data.x, data.z, Ni)
    plt.gca().set_aspect(1)
    plt.title('Example 1')
    plt.xlabel('$x$ ($R_E$)')
    plt.ylabel('$z$ ($R_E$)')

def example2():
    xs = np.linspace(5, 20, 101)
    zs = np.linspace(-10, 10, 101)

    data = InterpData(cfg, xs, 0, zs)
    # Or, if you want to reuse raw data cache,
    # data_raw = Data(cfg, slices=cfg.minimum_slice(*np.meshgrid(xs, 0, zs), extra=2))
    # data = InterpData(data_raw, xs, 0, zs)

    # Interpolated data will also be cached, and the cache size
    # is calculated independently.
    Ni = data[frame, 'Ni'] * units.N

    plt.figure()
    plt.pcolormesh(data.x, data.z, Ni)
    plt.gca().set_aspect(1)
    plt.title('Example 2')
    plt.xlabel('$x$ ($R_E$)')
    plt.ylabel('$z$ ($R_E$)')

if __name__ == '__main__':
    example1()
    example2()
    plt.show()
```

### Remote Data Access
PyFefi2 allows you to directly process simulation data hosted on a remote server without downloading the raw files manually. You'll need a credential to access a remote server. Run this command in your Python interpreter to generate the credential:
```python
>>> from pyfefi2 import Auth
>>> auth = Auth() # load or generate credential
>>> auth.print()  # print username and public key
```

You'll need to provide your username and public key to the maintainer of the remote dataset. To use the remote dataset, simply do:
```python
>>> from pyfefi2 import Config
>>> cfg = Config('https://your-domain.com/path/to/data/')
```

Data accessed via this method is compressed with the [SZ3](https://github.com/szcompressor/SZ3) lossy compression library. The default error is `1e-3`, which should be sufficiently accurate for most use cases.
