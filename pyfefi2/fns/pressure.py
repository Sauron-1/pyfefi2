import math
from .register import register

@register('Pth')
def thermal_pressure(kB, Ni, Ti1par, Ti1per):
    return (1/3)*kB*Ni*(Ti1par + 2*Ti1per)

@register('Pd')
def dynamic_pressure(Ni, Upx, Upy, Upz, mi):
    return 0.5 * mi*Ni * (Upx**2 + Upy**2 + Upz**2)

@register('Pb')
def magnetic_pressure(Bx, By, Bz, mu0):
    return (0.5/mu0) * (Bx**2 + By**2 + Bz**2)
