import netCDF4
import os

from .pyfefi_kernel import compress

def compress_file(src, dst_dir, skip=True, **kwargs):
    ds = netCDF4.Dataset(src)

    src_fn = os.path.basename(src)
    dst_fn = src_fn[:-2] + 'sz3'
    dst = os.path.join(dst_dir, dst_fn)

    if os.path.exists(dst):
        if skip:
            return
        os.remove(dst)

    default_cfg = dict(
        err_mode = 'abs_and_rel',
        rel_err = 1e-3,
        abs_err = 1e-3
    )
    default_cfg.update(kwargs)

    cf = compress.CompressedFile(
        filename = dst,
        mode = 'w',
        **default_cfg
    )
    cf.put_array('param', ds['param'][...], compressed=False)
    for name, var in ds.variables.items():
        if len(var.dimensions) == 3:
            cf.put_array(name, var[...])
