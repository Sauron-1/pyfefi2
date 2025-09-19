import numpy as np
import f90nml
import os
from scipy import constants as C

class Units:

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

        omega_i = B0 * (C.e / C.m_p)
        Va = B0 / np.sqrt(C.mu_0*N0*C.m_p)
        di = Va / omega_i

        R_lambda = nml['input_parameters']['roverL']

        self.scale_factor = Re / (di / R_lambda)

        self.m = C.m_p
        self.B = B0
        if scaled:
            # Keep resulting speed the same
            self.L = Re
            self.t = self.scale_factor / omega_i
        else:
            self.L = di * R_lambda
            self.t = 1 / omega_i

        self.v = self.L / self.t
        self.V = self.v

        self.N = N0

        self.E = self.v * self.B  # E ~ v \times B
        #self.J = self.B / self.L / C.mu_0  # J ~ 1/mu_0 * \nabla B
        self.q = self.m / (self.t * self.B)  # m dv/dt ~ q v \times B
        self.J = self.q * self.v * self.N

        self.T = self.m * self.v**2 / C.k

        self.Re = Re

        self.Neq = eq_info[3]

        self.Vsw = np.linalg.norm(Vsw) * 1e3
        self.Nsw = N0
        self.Tsw = Tsw
        self.Ma = -self.Vsw / Va * 1e3
        self.Bsw = np.array(sw_info[:3])*1e-9
        self.Va = Va
