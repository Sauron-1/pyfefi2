from ..data import Data
from ..units import Units

def register(name, vectorize=False):
    def decorator(fn):
        Data.register(name, vectorize)(fn)
        Units.register(name)(fn)
        return fn
    return decorator
