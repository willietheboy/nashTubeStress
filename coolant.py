#!/usr/bin/env python3

import sys, time, os
from math import exp, log, sqrt, pi, asin
import numpy as np
from printers import *

#################################### CLASSES ###################################

class LiquidSodium:
    """
    Usage: thermo = LiquidSodium()
           thermo.update(T) # T in [K]

    Fink, J. K. and Leibowitz, L., 1995. Thermodynamic and transport properties
    of sodium liquid and vapor. Technical Report ANL/RE-95/2,
    Reactor Engineering Division, Argonne National Laboratory,
    Chicago. doi:10.2172/94649.
    """
    def __init__ (self, verbose=False):
        self.verbose = verbose
        if verbose:
            headerprint('Liquid Sodium', ' ')

    def update (self, T):
        self.T = T
        T_c = 2503.7 # K
        rho_c = 219. # kg/m^3
        self.rho = rho_c + 275.32*(1 - self.T/T_c) + \
                   511.58*sqrt(1 - self.T/T_c) # kg/m^3
        self.Cp = (1.6582 - 8.4790e-4*self.T + \
                   4.4541e-7*pow(self.T, 2) - \
                   2992.6*pow(self.T, -2) ) *1e3 # m^2/s^2/K
        self.mu = exp(-6.4406 - 0.3958*log(self.T) + \
                      556.835/self.T)# kg/m/s
        self.kappa = 124.67 - 0.11381*self.T + \
                     5.5226e-5*pow(self.T, 2) - \
                     1.1842e-8*pow(self.T, 3) # kg*m/s^3/K
        self.nu = self.mu / self.rho
        self.alpha = self.kappa / (self.rho * self.Cp)
        self.Pr = self.nu / self.alpha
        if self.verbose:
            valprint('T', self.T, 'K')
            valprint('rho', self.rho, 'kg/m^3')
            valprint('Cp', self.Cp, 'm^2/s^2/K')
            valprint('mu', self.mu*1e6, 'x1e6 kg/m/s')
            valprint('kappa', self.kappa, 'kg*m/s^3/K')
            valprint('Pr', self.Pr)

class NitrateSalt:
    """
    Usage: thermo = NitrateSalt()
           thermo.update(T) # T in [K]

    Zavoico, A. B., 2001. Solar power tower design basis document.
    Technical Report SAND2001-2100, Sandia National Laboratories,
    Albuquerque, NM. doi:10.2172/786629.
    """
    def __init__ (self, verbose=False):
        self.verbose = verbose
        if verbose:
            headerprint('Nitrate Salt', ' ')

    def update (self, T):
        self.T = min(T, 873.15) # K
        self.rho = 2263.7234 - 0.636*self.T # kg/m^3
        self.Cp = 1396.0182 + 0.172*self.T # m^2/s^2/K
        self.mu = (-0.0001474*pow(self.T, 3) + \
                   0.348886926471821*pow(self.T, 2) \
                   - 277.603979928015*self.T + \
                   75514.7595133316) *1e-6 # kg/m/s
        self.kappa = 0.00019*self.T + 0.3911015 # kg*m/s^3/K
        self.nu = self.mu / self.rho
        self.alpha = self.kappa / (self.rho * self.Cp)
        self.Pr = self.nu / self.alpha
        if self.verbose:
            valprint('T', self.T, 'K')
            valprint('rho', self.rho, 'kg/m^3')
            valprint('Cp', self.Cp, 'm^2/s^2/K')
            valprint('mu', self.mu*1e6, 'x1e6 kg/m/s')
            valprint('kappa', self.kappa, 'kg*m/s^3/K')
            valprint('Pr', self.Pr)

class ChlorideSalt:
    """
    Usage: thermo = ChlorideSalt()
           thermo.update(T) # T in [K]

    NREL (Janna Martinek mailto:janna.martinek@nrel.gov)
    """
    def __init__ (self, verbose=False):
        self.verbose = verbose
        if verbose:
            headerprint('Ternary Chloride Salt', ' ')

    def update (self, T):
        self.T = T # K
        self.rho = 2124 - 0.579*self.T # kg/m^3
        self.Cp = 1412 - 0.448*self.T # m^2/s^2/K
        self.mu = 6.891e-4 * exp(1225/self.T) # kg/m/s
        self.kappa = 1.16 - 0.00146*self.T + \
                     7.15e-7*pow(self.T,2) # kg*m/s^3/K
        self.nu = self.mu / self.rho
        self.alpha = self.kappa / (self.rho * self.Cp)
        self.Pr = self.nu / self.alpha
        if self.verbose:
            valprint('T', self.T, 'K')
            valprint('rho', self.rho, 'kg/m^3')
            valprint('Cp', self.Cp, 'm^2/s^2/K')
            valprint('mu', self.mu*1e6, 'x1e6 kg/m/s')
            valprint('kappa', self.kappa, 'kg*m/s^3/K')
            valprint('Pr', self.Pr)

################################### FUNCTIONS ##################################

def heat_transfer_coeff(thermo, ri, ro, correlation, mode, arg, verbose=False):
    """
    Inputs:
        thermo :         LiquidSodium, NitrateSalt, ChlorideSalt
        ri :              tube inner radius (m)
        ro :              tube outer radius (m)
        correlation :    'Dittus', 'Skupinski', 'Sleicher', ...
        mode :           'velocity', 'dotm', 'heatCapRate' (m/s, kg/s, J/K/s)
        arg :            either velocity, mass-flow or heat capacity rate
        verbose :        (default:False)
    Return:
        h :              heat transfer coefficient (W/(m^2.K))
    """
    di = ri*2 # inner pipe diameter (m)
    t = ro - ri # tube wall thickness (m)
    A_di = pi * pow(di/2., 2) # cross sectional area of pipe flow
    if mode=='velocity':
        U = arg # m/s
        dotm = U * (A_di * thermo.rho)
        hcr = dotm * thermo.Cp
    elif mode=='dotm':
        dotm = arg # kg/s
        U = dotm / (A_di * thermo.rho)
        hcr = dotm * thermo.Cp
    elif mode=='heatCapRate':
        hcr = arg # J/K/s
        dotm = hcr / thermo.Cp
        U = dotm / (A_di * thermo.rho)
    else: sys.exit('HTC mode: {} not recognised'.format(mode))
    Re = U * di / thermo.nu # Reynolds
    Pe = Re * thermo.Pr # Peclet
    f = pow(1.82 * np.log10(Re) - 1.64, -2)
    DP_f = f * (0.5 * thermo.rho * pow(U, 2)) \
              / di # kg/m/s^2 for a metre of pipe!
    # if isinstance(thermo, liquidSodium):
    if correlation == 'Dittus':
        # Dittus-Boelter (Holman p286):
        Nu = 0.023 * pow(Re, 0.8) * pow(thermo.Pr, 0.4)
    elif correlation == 'Skupinski':
        # Skupinski, Tortel and Vautrey
        # https://doi.org/10.1016/0017-9310(65)90077-3
        Nu = 4.82 + 0.0185 * pow(Pe, 0.827)
    elif correlation == 'Notter':
        # Notter and Schleicher
        # https://doi.org/10.1016/0009-2509(72)87065-9
        Nu = 6.3 + 0.0167 * pow(Pe, 0.85) * pow(thermo.Pr, 0.08)
    elif correlation == 'Chen':
        # Chen and Chiou
        # https://doi.org/10.1016/0017-9310(81)90167-8
        Nu = 5.6 + 0.0165 * pow(Pe, 0.85) * pow(thermo.Pr, 0.01)
    elif correlation == 'Petukhov':
        # Petukhov
        # https://doi.org/10.1016/S0065-2717(08)70153-9
        Nu = sys.exit("Correlation not yet implemented - try Gnielinski!")
    elif correlation == 'Gnielinski':
        # V. Gnielinski, New Equations for Heat and Mass Transfer
        # in Turbulent Pipe and Channel Flow,
        # International Chemical Engineering,
        # Vol. 16, No. 2, 1976, pp. 359-68.
        if thermo.Pr < 1.5:
            Nu = 0.0214*(pow(Re, 0.8) - 100)*pow(thermo.Pr, 0.4)
        else:
            Nu = 0.012*(pow(Re, 0.87) - 280)*pow(thermo.Pr, 0.4)
    else: sys.exit('Correlation: {} not recognised'.format(correlation))
    htc = Nu * thermo.kappa / di
    if verbose:
        valprint('U', U, 'm/s')
        valprint('dotm', dotm, 'kg/s')
        valprint('Re', Re)
        valprint('Pe', Pe)
        valprint('deltaP', DP_f, 'Pa/m')
        valprint('HCR', hcr, 'J/K/s')
        valprint('htc', htc, 'W/m^2/K')
    return htc
