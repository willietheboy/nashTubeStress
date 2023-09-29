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
import scipy.optimize as opt
from pint import UnitRegistry
UR_ = UnitRegistry()
Q_ = UR_.Quantity

import nashTubeStress as nts
import coolant
from printers import *

import matplotlib as mpl
import matplotlib.pyplot as plt

################################### FUNCTIONS ##################################

def findFluxStress(flux, s, f, i, point):
    """
    Helper for finding optimum flux for certain stress condition
    """
    s.CG = flux

    ret = s.solve(eps=1e-6)
    s.postProcessing()

    if point=='max':
        # T_max|sigmaEqMax:
        sigmaEqMax = np.interp(np.max(s.T), f[:,0], f[:,i])
        return sigmaEqMax - np.max(s.sigmaEq)
    elif point=='inside':
        # T_i:
        sigmaEqMax = np.interp(s.T[0,0], f[:,0], f[:,i])
        return sigmaEqMax - s.sigmaEq[0,0]
    elif point=='outside':
        # T_o
        sigmaEqMax = np.interp(s.T[0,-1], f[:,0], f[:,i])
        return sigmaEqMax - s.sigmaEq[0,-1]
    else: sys.exit('Variable point {} not recognised'.format(point))

##################################### MAIN #####################################

if __name__ == "__main__":
    """
    Determine conservative AFD using twice yield method
    Falcone (1986), A Handbook for Solar Central Receiver Design
    """

    salt = coolant.NitrateSalt(); sodium = coolant.LiquidSodium()
    nr=30; nt=91
    h_ext=30            # convective loss due to wind W/(m2.K)

    OD = 19.05          # mm
    WT = 1.2446         # mm
    b = OD/2e3          # outside tube radius [mm->m]
    a = (b-WT*1e-3)     # inside tube radius [mm->m]
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b) # nr, nt -> resolution
    for mat in ['316H', 'P91']:
        headerprint('Allowable flux density: {}'.format(mat), ' ')
        if mat == '316H':
            k = 21; alpha=20e-6; E = 165e9; nu = 0.31
        if mat == 'P91':
            k = 27.5; alpha=14e-6; E = 183e9; nu = 0.3
        s = nts.Solver(
            g, debug=False, CG=0.85e6, k=k, T_int=723.15, R_f=0,
            A=0.968, epsilon=0.87, T_ext=293.15, h_ext=h_ext,
            P_i=0e5, alpha=alpha, E=E, nu=nu, n=1, bend=False
        )
        s.extBC = s.extTubeHalfCosFluxRadConv
        s.intBC = s.intTubeConv
        fv = np.genfromtxt(os.path.join('mats', mat), delimiter=';')
        fv[:,0] += 273.15 # degC to K
        fv[:,2] *= 3e6 # apply 3S criteria to S_o and convert MPa->Pa
        T_int = np.linspace(290, 565, 12)+273.15
        TSod_met = np.zeros(len(T_int))
        fluxSod = np.zeros(len(T_int))
        TSalt_met = np.zeros(len(T_int))
        fluxSalt = np.zeros(len(T_int))
        t = time.perf_counter()
        for i in range(len(T_int)):
            s.T_int = T_int[i]
            sodium.update(T_int[i])
            s.h_int = coolant.heat_transfer_coeff(
                sodium, a, b, 'Chen', 'velocity', 4
            )
            fluxSod[i] = opt.newton(
                findFluxStress, 1e6,
                args=(s, fv, 2, 'outside'),
                maxiter=1000, tol=1e-1
            )
            TSod_met[i] = np.max(s.T)
            salt.update(T_int[i])
            s.h_int = coolant.heat_transfer_coeff(
                salt, a, b, 'Dittus', 'velocity', 4
            )
            fluxSalt[i] = opt.newton(
                findFluxStress, 5e5,
                args=(s, fv, 2, 'outside'),
                maxiter=1000, tol=1e-1
            )
            TSalt_met[i] = np.max(s.T)
        valprint('Time taken', time.perf_counter() - t, 'sec')

        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)
        ax.plot(T_int-273.15,fluxSod*1e-6, label=r'Sodium')
        ax.plot(T_int-273.15,fluxSalt*1e-6, label=r'Nitrate salt')
        ax.set_xlabel(r'\textsc{fluid temperature}, '+\
                      '$T_\mathrm{f}$ (\si{\celsius})')
        ax.set_ylabel(
            r'\textsc{incident flux}, $\phi_\mathrm{q}$ '+\
            '(\si{\mega\watt\per\meter\squared})'
        )
        #ax.set_ylim(0.2, 1.6)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig('{0}_OD{1:.2f}_WT{2:.2f}_peakFlux.pdf'.format(mat, OD, WT),
                    transparent=True)
        fig.savefig('{0}_OD{1:.2f}_WT{2:.2f}_peakFlux.png'.format(mat, OD, WT),
                    dpi=150)
        plt.close(fig)
        ## Dump peak flux results to CSV file:
        csv = np.c_[T_int,
                    TSod_met, fluxSod,
                    TSalt_met, fluxSalt,
        ]
        np.savetxt('{0}_OD{1:.2f}_WT{2:.2f}_peakFlux.csv'.format(mat, OD, WT),
                   csv, delimiter=',', header='T_int(K),'+\
                   'TSod_metal(K),fluxSod(W/(m^2.K))'+\
                   'TSalt_metal(K),fluxSalt(W/(m^2.K))'
        )
