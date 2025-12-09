from ..data import Data
from ..units import Units

def register(name):
    def decorator(fn):
        Data.register(name)(fn)
        Units.register(name)(fn)
        return fn
    return decorator
