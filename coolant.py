#!/usr/bin/env python3
# Copyright (C) 2021 William R. Logie

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
nashTubeStress.py
 -- steady-state temperature distribution (Gauss-Seidel iteration)
 -- biharmonic thermoelastic stress

See also:
 -- Solar Energy 160 (2018) 368-379
 -- https://doi.org/10.1016/j.solener.2017.12.003
"""

import sys, time, os
from math import exp, log, sqrt, pi, ceil, floor, asin

import numpy as np
from pint import UnitRegistry
UR_ = UnitRegistry()
Q_ = UR_.Quantity

#################################### CLASSES ###################################

class liquidSodium:
    """
    Usage: thermo = liquidSodium()
           thermo.update(T) # T in [K]

    Fink, J. K. and Leibowitz, L., 1995. Thermodynamic and transport properties
    of sodium liquid and vapor. Technical Report ANL/RE-95/2,
    Reactor Engineering Division, Argonne National Laboratory,
    Chicago. doi:10.2172/94649.
    """
    def __init__ (self, debug):
        self.debug = debug
        if debug==True:
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
        if self.debug==True:
            valprint('T', self.T, 'K')
            valprint('rho', self.rho, 'kg/m^3')
            valprint('Cp', self.Cp, 'm^2/s^2/K')
            valprint('mu', self.mu*1e6, 'x1e6 kg/m/s')
            valprint('kappa', self.kappa, 'kg*m/s^3/K')
            valprint('Pr', self.Pr)

class nitrateSalt:
    """
    Usage: thermo = nitrateSalt()
           thermo.update(T) # T in [K]

    Zavoico, A. B., 2001. Solar power tower design basis document.
    Technical Report SAND2001-2100, Sandia National Laboratories,
    Albuquerque, NM. doi:10.2172/786629.
    """
    def __init__ (self, debug):
        self.debug = debug
        if debug==True:
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
        if self.debug==True:
            valprint('T', self.T, 'K')
            valprint('rho', self.rho, 'kg/m^3')
            valprint('Cp', self.Cp, 'm^2/s^2/K')
            valprint('mu', self.mu*1e6, 'x1e6 kg/m/s')
            valprint('kappa', self.kappa, 'kg*m/s^3/K')
            valprint('Pr', self.Pr)

class chlorideSalt:
    """
    Usage: thermo = chlorideSalt()
           thermo.update(T) # T in [K]

    NREL (Janna Martinek mailto:janna.martinek@nrel.gov)
    """
    def __init__ (self, debug):
        self.debug = debug
        if debug==True:
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
        if self.debug==True:
            valprint('T', self.T, 'K')
            valprint('rho', self.rho, 'kg/m^3')
            valprint('Cp', self.Cp, 'm^2/s^2/K')
            valprint('mu', self.mu*1e6, 'x1e6 kg/m/s')
            valprint('kappa', self.kappa, 'kg*m/s^3/K')
            valprint('Pr', self.Pr)

################################### FUNCTIONS ##################################

def headerprint(string, mychar='='):
    """ Prints a centered string to divide output sections. """
    mywidth = 64
    numspaces = mywidth - len(string)
    before = int(ceil(float(mywidth-len(string))/2))
    after  = int(floor(float(mywidth-len(string))/2))
    print("\n"+before*mychar+string+after*mychar+"\n")

def valprint(string, value, unit='-'):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .4f} {2}".format(string, value, unit))

def valeprint(string, value, unit='-'):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .4e} {2}".format(string, value, unit))

def matprint(string, value):
    """ Ensure uniform formatting of matrix value outputs. """
    print("{0}:".format(string))
    print(value)

def HTC(debug, thermo, a, b, k, correlation, mode, arg):
    """
    Inputs:
        debug : (default:False)
        thermo : liquidSodium, nitrateSalt, chlorideSalt
        a : tube inner diameter (m)
        b : tube outer diameter (m)
        k : tube thermal conductivity (W/(m.K))
        correlation : 'Dittus', 'Skupinski', 'Sleicher', ...
        mode : 'velocity', 'mdot', 'heatCapRate' (m/s, kg/s, J/K/s)
        arg : either velocity, mass-flow or heat capacity rate
    Return:
        h : heat transfer coefficient (W/(m^2.K))
        DP_f : pressure drop (Pa/m)
    """
    d_i = a*2 # inner pipe diameter [m]
    t = b - a # tube wall thickness [m]
    A_d_i = pi * pow(d_i/2., 2) # cross sectional area of pipe flow
    if mode=='velocity':
        U = arg # m/s
        mdot = U * (A_d_i * thermo.rho)
        hcr = mdot * thermo.Cp
    elif mode=='mdot':
        mdot = arg # kg/s
        U = mdot / (A_d_i * thermo.rho)
        hcr = mdot * thermo.Cp
    elif mode=='heatCapRate':
        hcr = arg #
        mdot = hcr / thermo.Cp
        U = mdot / (A_d_i * thermo.rho)
    else: sys.exit('Mode: {} not recognised'.format(mode))
    Re = U * d_i / thermo.nu # Reynolds
    Pe = Re * thermo.Pr # Peclet
    f = pow(1.82 * np.log10(Re) - 1.64, -2)
    DP_f = f * (0.5 * thermo.rho * pow(U, 2)) \
              / d_i # kg/m/s^2 for a metre of pipe!
    # if isinstance(thermo, liquidSodium):
    if correlation == 'Dittus':
        # Dittus-Boelter (Holman p286):
        Nu = 0.023 * pow(Re, 0.8) * pow(thermo.Pr, 0.4)
    elif correlation == 'Skupinski':
        # # Skupinski, Tortel and Vautrey:
        # # https://doi.org/10.1016/0017-9310(65)90077-3
        Nu = 4.82 + 0.0185 * pow(Pe, 0.827)
    elif correlation == 'Notter':
        # Notter and Schleicher:
        # https://doi.org/10.1016/0009-2509(72)87065-9
        Nu = 6.3 + 0.0167 * pow(Pe, 0.85) * pow(thermo.Pr, 0.08)
    elif correlation == 'Chen':
        # Chen and Chiou:
        # https://doi.org/10.1016/0017-9310(81)90167-8
        Nu = 5.6 + 0.0165 * pow(Pe, 0.85) * pow(thermo.Pr, 0.01)
    elif correlation == 'Petukhov':
        # Petukhov:
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
    h = Nu * thermo.kappa / d_i
    Bi = (t * h) / k
    if debug==True:
        valprint('U', U, 'm/s')
        valprint('mdot', mdot, 'kg/s')
        valprint('Re', Re)
        valprint('Pe', Pe)
        valprint('deltaP', DP_f, 'Pa/m')
        valprint('HCR', hcr, 'J/K/s')
        valprint('h_int', h, 'W/m^2/K')
        valprint('Bi', Bi)
    return [h, DP_f]
