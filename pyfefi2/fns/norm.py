import math
from ..data import Data

@Data.register('B_norm')
def norm_B(Bx, By, Bz):
    return math.sqrt(Bx**2 + By**2 + Bz**2)

@Data.register('Up_norm')
def norm_V(Upx, Upy, Upz):
    return math.sqrt(Upx**2 + Upy**2 + Upz**2)

@Data.register('Jp_norm')
def norm_J(Jx, Jy, Jz):
    return math.sqrt(Jx**2 + Jy**2 + Jy**2)

@Data.register('E_norm')
def norm_J(Ex, Ey, Ez):
    return math.sqrt(Ex**2 + Ey**2 + Ey**2)

@Data.register('Ti1')
def total_T(Ti1par, Ti1per):
    return (1/3)*(Ti1par + Ti1per*2)
