import numpy as np
import f90nml
import os
from scipy import constants as C

class RegisteredFunction:

    def __init__(self, fn):
        self._var_names = fn.__code__.co_varnames
        self._fn = fn

    def __call__(self, obj):
        var_list = [getattr(obj, name) for name in self._var_names]
        return self._fn(*var_list)

class Units:

    registered_fns = {}
    _basic_reg_functions = {}

    @classmethod
    def register(cls, name):
        def decorator(fn):
            reg_fn = RegisteredFunction(fn)
            cls.registered_fns[name] = reg_fn
            return fn
        return decorator

    def __init__(self, path, scaled=True, Re=6371e3):
        """
        Initializes the Units object.

        Args:
            path (str): The path to the simulation data folder.
            scaled (bool, optional): Whether to use scaled units. Defaults to True.
            Re (float, optional): The Earth radius in meters. Defaults to 6371e3.
        """
        if os.path.isfile(path):
            conf_fn = path
        else:
            conf_fn = os.path.join(path, 'fefi.input')
        nml = f90nml.read(conf_fn)

        simtype = nml['input_parameters']['simtype']

        if simtype[0] not in [6, 7] or (simtype[1] < 90 or simtype[1] >= 100):
            raise Exception('simtype %d, %d not supported' % (simtype[0], simtype[1]))

        sw_info = nml['input_parameters']['bnvt_sw']
        eq_info = nml['input_parameters']['bnvt_eq']
        B0 = np.linalg.norm(sw_info[:3]) * 1e-9
        Vsw = sw_info[3:6]
        N0 = sw_info[6]*1e6
        Tsw = sw_info[7] * (C.e/C.k)

        R_lambda = nml['input_parameters']['roverL']
        omega_i = B0 * (C.e / C.m_p)
        Va = B0 / np.sqrt(C.mu_0*N0*C.m_p)
        di = Va / omega_i

        self._units = {}
        self.scale_factor = Re / (di / R_lambda)

        # Basic units
        self['m'] = C.m_p
        self['B'] = B0
        self['N'] = N0

        if scaled:
            self['L'] = Re
            self['t'] = self.scale_factor / omega_i
        else:
            self['L'] = di * R_lambda
            self['t'] = 1 / omega_i

        # Derived units
        self['v'] = self.L / self.t
        self['E'] = self.v * self.B
        self['q'] = self.m / (self.t * self.B)
        self['J'] = self.q * self.v * self.N
        self['T'] = self.m * self.v**2 / C.k

        # Solarwind params
        self.Re = Re
        self.Neq = eq_info[3]
        self.Nsw = N0
        self.Tsw = Tsw
        self.Vsw = np.array(Vsw) * 1e3
        self.Ma = -np.linalg.norm(self.Vsw) / Va * 1e3
        self.Bsw = np.array(sw_info[:3]) * 1e-9
        self.Va = Va

        self.__init_alias()

    def __init_alias(self):
        self['R'] = self['L']
        self['V'] = self['v']
        self['Up'] = self['v']
        self['Ni'] = self['N']
        for axis in 'xyz':
            for name in ['B', 'E', 'J', 'Up']:
                self[f'{name}{axis}'] = self[name]
        for name in ['Ti1', 'Ti1par', 'Ti1per']:
            self[name] = self['T']

    def __getitem__(self, name):
        if name in self._units:
            return self._units[name]
        elif name in self.registered_fns:
            result = self.registered_fns[name](self)
            self[name] = result
            return result
        else:
            raise KeyError(f"Undefined unit name {name}")

    def __setitem__(self, name, value):
        self._units[name] = value

    def __getattr__(self, name):
        return self[name]
