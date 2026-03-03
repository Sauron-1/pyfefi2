from .config import Config
from .slices import Slice
from .coords import Coordinates
from .data import Data, InterpData

from .fns.register import register

from .interp import interp
from .pyfefi_kernel import compress, tracer as line_tracer

from .compress_file import compress_file
