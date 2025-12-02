import math
from ..data import Data

@Data.register('Pth')
def thermal_pressure(Ti1, Ni):
    return Ni*Ti1

@Data.register('Pd')
def dynamic_pressure(Ni, Upx, Upy, Upz):
    return 0.5 * Ni * (Upx**2 + Upy**2 + Upz**2)
