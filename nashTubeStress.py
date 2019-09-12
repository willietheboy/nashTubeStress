#!/usr/bin/env python
# Copyright (C) 2018 William R. Logie

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
 -- tested 09/07/2019 with Python 2.7.15+ and pip packages 

See also:
 -- Solar Energy 160 (2018) 368-379
 -- https://doi.org/10.1016/j.solener.2017.12.003
"""

import sys, time, os
from math import exp, log, sqrt, pi, ceil, floor, asin
import numpy as np # version 1.15.4
from numpy import ma
import weave # version 0.17.0
from weave import converters
import scipy.optimize as opt # version 1.1.0
from pint import UnitRegistry # version 0.8.1
UR_ = UnitRegistry()
Q_ = UR_.Quantity

# Plotting:
import matplotlib as mpl # version 2.2.3
import matplotlib.pyplot as plt
#params = {'text.latex.preamble': [r'\usepackage{mathptmx,txfonts,siunitx}']}
#plt.rcParams.update(params)
mpl.rc('figure.subplot', bottom=0.13, top=0.95)
mpl.rc('figure.subplot', left=0.15, right=0.95)
mpl.rc('xtick', labelsize='medium')
mpl.rc('ytick', labelsize='medium')
mpl.rc('axes', labelsize='large')
mpl.rc('axes', titlesize='large')
mpl.rc('legend', fontsize='medium')
mpl.rc('lines', markersize=4)
mpl.rc('lines', linewidth=0.5)
from matplotlib import colors, ticker, cm
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.projections import PolarAxes
import matplotlib.transforms as mtransforms
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist.grid_finder import \
    (FixedLocator, MaxNLocator, DictFormatter)
# if you're matplotlib is older than version 2:
#import colormaps as cmaps # magma, inferno, plasma, viridis

#################################### CLASSES ###################################

class liquidSodium:
    """
    Usage: thermo = liquidSodium()
           thermo.update(T) # T in [K]
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

class Grid:
    
    """ A cylindrical coordinate (theta, r) grid class """
    
    def __init__(self, nr=6, nt=61, rMin=0.5, rMax=0.7,
                 thetaMin=0, thetaMax=np.radians(180)):
        self.nr, self.nt = nr, nt
        self.a, self.b = rMin, rMax
        r = np.linspace(rMin, rMax, nr)
        theta = np.linspace(thetaMin, thetaMax, nt)
        self.r, self.theta = np.meshgrid(r, theta)
        self.dr = float(rMax-rMin)/(nr-1)
        dTheta = float(thetaMax-thetaMin)/(nt-1)
        # face surface (sf) areas:
        self.sfRmin = (np.ones(nt) * pi * rMin) / (nt - 1)
        self.sfRmin[0] *= 0.5; self.sfRmin[-1] *= 0.5
        self.sfRmax = (np.ones(nt) * pi * rMax) / (nt - 1)
        self.sfRmax[0] *= 0.5; self.sfRmax[-1] *= 0.5
        # create 'ghost' elements for symmetry BCs:
        theta = np.insert(theta, 0, thetaMin-dTheta)
        theta = np.append(theta, thetaMax+dTheta)
        self.meshR, self.meshTheta = np.meshgrid(r, theta)
        # create constants for use in iterations:
        self.twoDrR = 2 * self.dr * self.meshR[1:-1,1:-1]
        self.dr2, self.dTheta2 = self.dr**2, dTheta**2
        self.dTheta2R2 = self.dTheta2 * self.meshR[1:-1,1:-1]**2
        self.dnr = (2. / self.dr2 + 2. / self.dTheta2R2)
        # create 'mask array' for front-side collimated flux logic:
        self.cosTheta = np.cos(theta)
        self.sinTheta = np.sin(theta)
        self.tubeFront = np.ones(len(theta))
        self.tubeFront[self.cosTheta<0] = 0.0

class Solver:
    
    """  A Laplacian solver for steady-state conduction in cylinders 
         -- Gauss-Seidel iteration of T(r, theta) 
         -- bi-harmonic thermoelastic stress post-processing """

    # Constants:
    sigma = 5.67e-8    # Stefan-Boltzmann
    
    def __init__(self, grid, debug=False, it='inline', CG=8.5e5, 
                 k=20, T_int=723.15, h_int=10e3, U=4.0, R_f=0., A=0.968, 
                 epsilon=0.87, T_ext=293.15, h_ext=30., P_i=0e5, 
                 alpha=18.5e-6, E=165e9, nu=0.3, n=1, bend=False):
        self.debug = debug
        # Class constants and variables (default UNS S31600 @ 450degC):
        self.g = grid
        self.setIterator(it)
        self.CG = CG            # concentration (C) x solar constant (G)
        self.k = k              # thermal conductivity of tube
        self.T_int = T_int      # temperature of heat transfer fluid
        self.h_int = h_int      # constant int convection coefficient
        self.R_f = R_f          # internal fouling coefficient
        self.A = A              # tube external surface absorptance
        self.epsilon = epsilon  # tube external emmissivity
        self.T_ext = T_ext      # ambient temperature
        self.h_ext = h_ext      # ext convection coefficient (with wind)
        self.P_i = P_i          # internal pipe pressure
        self.alpha = alpha      # thermal expansion coefficienct of tube
        self.E = E              # Modulus of elasticity
        self.nu = nu            # Poisson's coefficient
        self.n = n              # Number of Fourier 'frequencies'
        self.bend = bend        # switch to allow tube bending
        self.meshT = np.ones((grid.nt+2, grid.nr), 'd') * T_int
        self.T = self.meshT[1:-1,:] # remove symm for post-processing

    def computeError(self):        
        """ Computes absolute error using an L2 norm for the solution.
        This requires that self.T and self.old_T must be appropriately
        setup - only used for numpyStep """        
        v = (self.meshT - self.old_T).flat
        return np.sqrt(np.dot(v,v))

    def numpyStep(self):
        """ Gauss-Seidel iteration using numpy expression. """
        self.old_T = self.meshT.copy()
        # "Driving" BCs (heat flux, radiation and convection)
        self.extBC()
        self.intBC()
        # Numpy iteration
        self.meshT[1:-1,1:-1] = ( 
            ( self.meshT[1:-1,2:] - self.meshT[1:-1,:-2] ) 
            / self.g.twoDrR +
            ( self.meshT[1:-1,:-2] + self.meshT[1:-1,2:] ) 
            / self.g.dr2 +
            ( self.meshT[:-2,1:-1] + self.meshT[2:,1:-1] ) 
            / self.g.dTheta2R2
        ) / self.g.dnr
        # Symmetry boundary conditions
        self.symmetryBC()
        return self.computeError()

    def blitzStep(self):
        """ Gauss-Seidel iteration using numpy expression
        that has been blitzed using weave """
        self.old_T = self.meshT.copy()
        # "Driving" BCs (heat flux, radiation and convection)
        self.extBC()
        self.intBC()
        # Prepare constants and arrays for blitz
        T = self.meshT
        twoDrR = self.g.twoDrR
        dr2 = self.g.dr2
        dTheta2R2 = self.g.dTheta2R2
        dnr = self.g.dnr
        expr = "T[1:-1,1:-1] = ("\
            "( T[1:-1,2:] - T[1:-1,:-2] ) / twoDrR +"\
            "( T[1:-1,:-2] + T[1:-1,2:] ) / dr2 +"\
            "( T[:-2,1:-1] + T[2:,1:-1] ) / dTheta2R2"\
            ") / dnr"
        weave.blitz(expr, check_size=0)
        # Transfer result back to mesh/grid
        self.meshT = T
        # Symmetry boundary conditions
        self.symmetryBC()
        return self.computeError()

    def inlineStep(self):
        """ Gauss-Seidel iteration using an inline C code """
        # "Driving" BCs (heat flux, radiation and convection)
        self.extBC()
        self.intBC()
        # Prepare constants and arrays for blitz
        T = self.meshT
        nt, nr = self.meshT.shape
        twoDrR = self.g.twoDrR
        dr2 = self.g.dr2
        dTheta2R2 = self.g.dTheta2R2
        dnr = self.g.dnr
        code = """
               #line 000 "nashTubeStress.py"
               double tmp, err, diff;
               err = 0.0;
               for (int i=1; i<nt-1; ++i) {
                   for (int j=1; j<nr-1; ++j) {
                       tmp = T(i,j);
                       T(i,j) = ((T(i,j+1) - T(i,j-1))/twoDrR(i-1,j-1) +
                                 (T(i,j-1) + T(i,j+1))/dr2 +
                                 (T(i-1,j) + T(i+1,j))/dTheta2R2(i-1,j-1)
                                ) / dnr(i-1,j-1);
                       diff = T(i,j) - tmp;
                       err += diff*diff;
                   }
               }
               return_val = sqrt(err);
               """
        err = weave.inline(code,
                           ['nr', 'nt', 'T', 'twoDrR', 
                            'dr2', 'dTheta2R2', 'dnr'],
                           type_converters=converters.blitz,
                           compiler = 'gcc')
        # Transfer result back to mesh/grid
        self.meshT = T
        # Symmetry boundary conditions
        self.symmetryBC()
        return err

    def setIterator(self, iterator='numpy'):        
        """ Sets the iteration scheme to be used while solving given a
        string which should be one of ['numpy', 'blitz', 'inline']. """        
        if iterator == 'numpy':
            self.iterate = self.numpyStep
        elif iterator == 'blitz':
            self.iterate = self.blitzStep
        elif iterator == 'inline':
            self.iterate = self.inlineStep
        else:
            self.iterate = self.numpyStep            
                
    def solve(self, n_iter=0, eps=1.0e-16):        
        """ Solves the equation given:
        - an error precision -- eps
        - a maximum number of iterations -- n_iter """
        err = self.iterate()
        count = 1
        while err > eps:
            if n_iter and count >= n_iter:
                return err
            err = self.iterate()
            count = count + 1
        self.T = self.meshT[1:-1,:]
        return count

    def postProcessing(self):
        self.stress()
        return

    ############################ BOUNDARY CONDITIONS ###########################

    def symmetryBC(self):        
        """ Sets the left and right symmetry BCs """       
        self.meshT[0, 1:-1] = self.meshT[2, 1:-1]
        self.meshT[-1, 1:-1] = self.meshT[-3, 1:-1]

    def tubeExtTemp(self):
        """ fixedValue boundary condition """        
        self.meshT[:,-1] = self.T_ext

    def tubeExtConv(self):
        """ Convective boundary condition """        
        self.meshT[:, -1] = (self.meshT[:,-2] + \
                             ((self.g.dr * self.h_ext / 
                               self.k) * self.T_ext)) \
            / (1 + (self.g.dr * self.h_ext / self.k))

    def tubeExtFlux(self):        
        """ Heat flux boundary condition """
        self.meshT[:,-1] = ((self.g.dr * self.CG) / 
                            self.k) + self.meshT[:, -2]

    def tubeExtCosFlux(self): 
        """ 100% absorbed cosine flux boundary condition """
        self.heatFluxInc = (self.g.tubeFront * \
                            self.CG * self.g.cosTheta)
        heatFluxAbs = self.heatFluxInc
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (heatFluxAbs * self.g.dr / self.k)

    def tubeExtCosFluxRadConv(self): 
        """ Heat flux, re-radiation and convection boundary condition """
        self.heatFluxInc = (self.g.tubeFront * \
                            self.CG * self.g.cosTheta)
        heatFluxAbs = self.heatFluxInc * self.A \
                      - (self.sigma * self.epsilon \
                         * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                      - (self.h_ext * (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (heatFluxAbs * self.g.dr / self.k)

    def tubeExtCosFluxRadConvAdiabaticBack(self): 
        """ Heat flux, re-radiation and convection boundary condition """
        self.heatFluxInc = (self.g.tubeFront * \
                            self.CG * self.g.cosTheta)
        heatFluxAbs = self.heatFluxInc * self.A \
                      - (self.g.tubeFront * self.sigma * self.epsilon \
                         * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                      - (self.h_ext * self.g.tubeFront * 
                         (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (heatFluxAbs * self.g.dr / self.k)

    def tubeIntTemp(self):
        """ fixedValue boundary condition """        
        self.meshT[:,0] = self.T_int

    def tubeIntFlux(self):        
        """ Heat flux boundary condition """        
        self.meshT[:,0] = ((self.g.dr * self.CG) / 
                           self.k) + self.meshT[:, 1]

    def tubeIntConv(self):
        """ Convective boundary condition to tube flow with fouling """
        U = 1 / (self.R_f + (1 / self.h_int))
        self.meshT[:,0] = (self.meshT[:,1] + \
                           ((self.g.dr * U / self.k) * \
                            self.T_int)) \
            / (1 + (self.g.dr * U / self.k))

    ############################## POST-PROCESSING #############################

    def stress(self):
        """ The Timoshenko & Goodier approach (dating back to 1937):
        -- S. Timoshenko and J. N. Goodier. Theory of Elasticity. 
           p432, 1951.
        -- J. N. Goodier, Thermal Stresses and Deformation, 
           J. Applied Mechanics, Trans ASME, vol. 24(3), 
           p467-474, 1957. """
        # smaller names for equations below:
        a, b = self.g.a, self.g.b
        P_i, alpha, E, nu = self.P_i, self.alpha, self.E, self.nu
        # create a local 'harmonic' cylinder of T, theta and r:
        T = np.insert(self.T, 0, self.T[1:,:][::-1], axis=0)
        theta = np.linspace(np.radians(-180), np.radians(180), 
                            (self.g.nt*2)-1)
        r = np.linspace(a, b, self.g.nr)
        meshR, meshTheta = np.meshgrid(r, theta)
        # local time-saving variables:
        meshR2 = meshR**2; meshR4 = meshR**4
        a2 = a**2; a4 = a**4
        b2 = b**2; b4 = b**4
        # 'guess' of coefficients for curve_fit function:
        p0 = [1.0] * (1 + (self.n * 2))
        # inside:
        popt1, pcov1 = opt.curve_fit(fourierTheta, theta, T[:,0], p0)
        Tbar_i = popt1[0]; BP = popt1[1]; DP = popt1[2];
        # outside:
        popt2, pcov2 = opt.curve_fit(fourierTheta, theta, T[:,-1], p0)
        Tbar_o = popt2[0]; BPP = popt2[1]; DPP = popt2[2];
        kappa_theta = (( (((BP * b) - (BPP * a)) / (b2 + a2)) 
                         * np.cos(meshTheta)) + \
                       ( (((DP * b) - (DPP * a)) / (b2 + a2)) 
                         * np.sin(meshTheta))) * \
                         (meshR * a * b) / (b2 - a2)
        kappa_tau = (( (((BP * b) - (BPP * a)) / (b2 + a2)) 
                       * np.sin(meshTheta)) - \
                     ( (((DP * b) - (DPP * a)) / (b2 + a2)) 
                       * np.cos(meshTheta))) * \
                       (meshR * a * b) / (b2 - a2)
        if self.bend:
            kappa_noM = meshR * ((((BP * a) + (BPP * b)) / (b2 + a2) * \
                                  np.cos(meshTheta)) \
                                 + (((DP * a) + (DPP * b)) / (b2 + a2) * \
                                    np.sin(meshTheta)))
        else: kappa_noM = 0.0
        C = (alpha * E) / (2 * (1 - nu))
        # Axisymmetrical thermal stress component:
        kappa = (Tbar_i - Tbar_o) / np.log(b/a)
        QR = kappa * C * (- np.log(b/meshR) - \
                          (a2/(b2 - a2) * (1 - b2/meshR2) * np.log(b/a)))
        QTheta = kappa * C * (1 - np.log(b/meshR) - \
                              (a2/(b2 - a2) * (1 + b2/meshR2) * np.log(b/a)))
        QZ = kappa * C * (1 - (2*np.log(b/meshR)) - (2 * a2 / (b2 - a2) * \
                                                     np.log(b/a)))
        # Nonaxisymmetrical T:
        T_theta = T - ((Tbar_i - Tbar_o) * \
                       np.log(b / meshR) / np.log(b / a)) - Tbar_o
        self.T_theta = T_theta
        # Nonaxisymmetric thermal stress component:
        QR += C * kappa_theta * (1 - (a2 / meshR2)) * (1 - (b2 / meshR2))
        QTheta += C * kappa_theta * (3 - ((a2 + b2) / meshR2) - \
                                     ((a2 * b2) / meshR4))
        QZ += alpha * E * ((kappa_theta * (nu / (1 - nu)) * \
                            (2 - ((a2 + b2) / meshR2))) + \
                           kappa_noM - T_theta)
        QRTheta = C * kappa_tau * (1 - (a2 / meshR2)) * (1 - (b2 / meshR2))
        QEq = np.sqrt(0.5 * ((QR - QTheta)**2 + \
                             (QTheta - QZ)**2 + \
                             (QZ - QR)**2) + \
                      6 * (QRTheta**2))
        # Pressure stress component:
        PR = ((a2 * self.P_i) / (b2 - a2)) * (1 - (b2 / meshR2))
        PTheta = ((a2 * self.P_i) / (b2 - a2)) * (1 + (b2 / meshR2))
        PZ = 0 #(a2 * self.P_i) / (b2 - a2)
        PEq = np.sqrt(0.5 * ((PR - PTheta)**2 + \
                             (PTheta - PZ)**2 + \
                             (PZ - PR)**2))
        sigmaR = QR + PR
        sigmaTheta = QTheta + PTheta
        sigmaZ = QZ + PZ
        sigmaRTheta = QRTheta
        # Equivalent/vM stress:
        sigmaEq = np.sqrt(0.5 * ((sigmaR - sigmaTheta)**2 + \
                                 (sigmaTheta - sigmaZ)**2 + \
                                 (sigmaZ - sigmaR)**2) + \
                          6 * (sigmaRTheta**2))
        self.popt1 = popt1
        self.popt2 = popt2
        self.sigmaR = sigmaR[self.g.nt-1:,:]
        self.sigmaRTheta = sigmaRTheta[self.g.nt-1:,:]
        self.sigmaTheta = sigmaTheta[self.g.nt-1:,:]
        self.sigmaZ = sigmaZ[self.g.nt-1:,:]
        self.sigmaEq = sigmaEq[self.g.nt-1:,:]
        if self.debug:
            valprint('Tbar_i', Tbar_i, 'K')
            valprint('B\'_1', BP, 'K')
            valprint('D\'_1', DP, 'K')
            valprint('Tbar_o', Tbar_o, 'K')
            valprint('B\'\'_1', BPP, 'K')
            valprint('D\'\'_1', DPP, 'K')
            headerprint('Stress at outside tube crown:', ' ')
            valprint('sigma_r', self.sigmaR[0,-1]*1e-6, 'MPa')
            valprint('sigma_rTheta', self.sigmaRTheta[0,-1]*1e-6, 'MPa')
            valprint('sigma_theta', self.sigmaTheta[0,-1]*1e-6, 'MPa')
            valprint('sigma_z', self.sigmaZ[0,-1]*1e-6, 'MPa')
            valprint('sigma_Eq', self.sigmaEq[0,-1]*1e-6, 'MPa')

################################### PLOTTING ###################################

def plotStress(theta, r, sigma, sigmaMin, sigmaMax, filename):
    fig = plt.figure(figsize=(2.5, 3))    
    fig.subplots_adjust(left=-1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90))
    cmap = cm.get_cmap('magma')
    #cmap = cmaps.magma
    levels = ticker.MaxNLocator(nbins=10).tick_values(
        sigmaMin*1e-6, sigmaMax*1e-6
    )
    cf = ax.contourf(theta, r, sigma*1e-6, levels=levels, cmap=cmap)
    ax.set_rmin(0)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label('$\sigma$ [MPa]')
    ax.patch.set_visible(False)
    ax.spines['polar'].set_visible(False)
    gridlines = ax.get_xgridlines()
    ticklabels = ax.get_xticklabels()
    for i in range(5, len(gridlines)):
        gridlines[i].set_visible(False)
        ticklabels[i].set_visible(False)
    ax.grid(axis='y', linewidth=0)
    ax.grid(axis='x', linewidth=0.2)
    plt.setp(ax.get_yticklabels(), visible=False)
    #fig.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotFEA(r, sigma, fc, i, filename, loc, ylabel):
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    a = r[0,0]
    ax.plot(r[0,:]*1e3, sigma[0,:]*1e-6, '-', color='C0', 
            label=r'$\theta=0^\circ$')
    ax.plot((a+fc[0][:,0])*1e3, fc[0][:,i], 'o', color='C0', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[20,:]*1e-6, '-', color='C1', 
            label=r'$\theta=60^\circ$')
    ax.plot((a+fc[1][:,0])*1e3, fc[1][:,i], '^', color='C1', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[40,:]*1e-6, '-', color='C2', 
            label=r'$\theta=120^\circ$')
    ax.plot((a+fc[2][:,0])*1e3, fc[2][:,i], 'v', color='C2', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[60,:]*1e-6, '-', color='C3', 
            label=r'$\theta=180^\circ$')
    ax.plot((a+fc[3][:,0])*1e3, fc[3][:,i], 's', color='C3', markevery=1)
    ax.set_xlabel('$r$ (mm)')
    #ax.set_xlim((a*1e3)-10,(b*1e3)+10)
    ax.set_ylabel(ylabel+' (MPa)')
    c0line = Line2D([], [], color='C0', marker='o',
                   label=r'$\theta=0^\circ$')
    c1line = Line2D([], [], color='C1', marker='^',
                   label=r'$\theta=60^\circ$')
    c2line = Line2D([], [], color='C2', marker='v',
                   label=r'$\theta=120^\circ$')
    c3line = Line2D([], [], color='C3', marker='s',
                   label=r'$\theta=180^\circ$')
    handles=[c0line, c1line, c2line, c3line]
    labels = [h.get_label() for h in handles]
    ax.legend([handle for i,handle in enumerate(handles)],
              [label for i,label in enumerate(labels)], loc=loc)
    fig.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)

################################### FUNCTIONS ##################################

def fourierTheta(theta, a0, *c):
    """ Timoshenko & Goodier equation """
    ret = a0 + np.zeros(len(theta))
    for i, n in zip(range(0,len(c),2), range(1,(len(c)/2)+1)):
        ret += (c[i] * np.cos(n * theta)) + (c[i+1] * np.sin(n * theta))
    return ret

def HTC(debug, thermo, a, b, k, correlation, mode, arg):
    """
    Inputs:
        debug : (default:False)
        thermo : liquidSodium, nitrateSalt, chlorideSalt
        a : tube inner diameter (m)
        b : tube outer diameter (m)
        k : tube thermal conductivity (W/(m.K))
        corrlation : 'Dittus', 'Skupinski', 'Sleicher'
        mode : 'velocity','mdot','heatCapRate' (m/s,kg/s,???)
        arg : either velocity, mass-flow or heat capacity rate
    Return:
        h : heat transfer coefficient (W/(m^2.K))
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
    f = pow(0.790 * np.log(Re) - 1.64, -2)
    DP_f = -f * (0.5 * thermo.rho * pow(U, 2)) \
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
    return h

def headerprint(string, mychar='='):
    """ Prints a centered string to divide output sections. """
    mywidth = 64
    numspaces = mywidth - len(string)
    before = int(ceil(float(mywidth-len(string))/2))
    after  = int(floor(float(mywidth-len(string))/2))
    print("\n"+before*mychar+string+after*mychar+"\n")

def valprint(string, value, unit='-'):
    """ Ensure uniform formatting of scalar value outputs. """
    print("{0:>30}: {1: .4f} ({2})".format(string, value, unit))

def matprint(string, value):
    """ Ensure uniform formatting of matrix value outputs. """
    print("{0}:".format(string))
    print(value)

################################### ROUTINES ###################################

def SE6413():
    """ 
    Reproduction of results from Solar Energy 160 (2018) 368-379
    https://doi.org/10.1016/j.solener.2017.12.003
    """
    
    headerprint(' NPS Sch. 5S 1" SS316 at 450degC ')

    nr=12; nt=91
    a = 30.098/2e3     # inside tube radius [mm->m]
    b = 33.4/2e3       # outside tube radius [mm->m]

    """ Create instance of Grid: """
    g = Grid(nr=nr, nt=nt, rMin=a, rMax=b) # nr, nt -> resolution

    """ Create instance of LaplaceSolver: """
    s = Solver(g, debug=True, CG=0.85e6, k=20, T_int=723.15, R_f=0,
               A=0.968, epsilon=0.87, T_ext=293.15, h_ext=20., 
               P_i=0e5, alpha=18.5e-6, E=165e9, nu=0.31, n=1,
               bend=False)

    """ Any of the properties defined above can be changed, e.g.: """
    # s.CG = 1.2e5 ...

    """ External BC: """
    #s.extBC = s.tubeExtTemp
    #s.extBC = s.tubeExtFlux
    #s.extBC = s.tubeExtConv
    #s.extBC = s.tubeExtCosFluxRadConv
    s.extBC = s.tubeExtCosFluxRadConvAdiabaticBack

    """ Internal BC: """
    #s.intBC = s.tubeIntTemp
    #s.intBC = s.tubeIntFlux
    s.intBC = s.tubeIntConv

    headerprint(' HTC: 10e3 W/m^s/K ', ' ')
    s.h_int = 10e3
    t = time.clock(); ret = s.solve(eps=1e-6); s.postProcessing()
    valprint('Time', time.clock() - t, 'sec')

    """ To access the temperature distribution: """
    #     s.T[theta,radius] using indexes set by nr and nt
    """ e.g. s.T[0,-1] is outer tube front """

    """ Same goes for stress fields: """
    #     s.sigmaR[theta,radius]
    #     s.sigmaTheta[theta,radius]
    #     s.sigmaZ[theta,radius]
    #     s.sigmaEq[theta,radius]

    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(), 
               'NPS5S1_316H_htc10_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'NPS5S1_316H_htc10_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'NPS5S1_316H_htc10_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'NPS5S1_316H_htc10_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'NPS5S1_316H_htc10_sigmaEq.pdf')

    headerprint(' HTC: 40e3 W/m^s/K ', ' ')
    s.h_int = 40e3
    t = time.clock(); ret = s.solve(eps=1e-6); s.postProcessing()
    valprint('Time', time.clock() - t, 'sec')

    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(), 
               'NPS5S1_316H_htc40_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'NPS5S1_316H_htc40_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'NPS5S1_316H_htc40_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'NPS5S1_316H_htc40_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'NPS5S1_316H_htc40_sigmaEq.pdf')

def Holms1952():
    """
    Holms, A.G., 1952. A biharmonic relaxation method for calculating thermal
    stress in cooled irregular cylinders. Technical Report NACA-TR-1059.
    Lewis Flight Propulsion Lab. Cleveland. 
    URL: https://ntrs.nasa.gov/search.jsp?R=19930092105 .
    """
    headerprint(' Holms (1952), NACA-TR-1059 ')

    nr=34; nt=61
    a = 101.6/1e3      # inside tube radius [mm->m]
    b = 304.8/1e3      # outside tube radius [mm->m]

    c0 = Q_(0,'degF').to('K').magnitude
    c1 = Q_(1000,'degF').to('K').magnitude
    c2 = Q_(500,'degF').to('K').magnitude
    E = Q_(17.5e6, 'psi').to('Pa').magnitude
    valprint('E', E*1e-9, 'GPa')
    alpha = Q_(8e-6, 'degF^-1').to('K^-1').magnitude
    valprint('alpha', alpha*1e6, 'x1e6 K^-1')
    
    # Create instance of Grid and Solver to use stress post-processor:
    g = Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    s = Solver(g, debug=True, alpha=alpha, 
               E=E, nu=0.3, n=1)
    s.T = (((c1-c0) * b) / (b**2 - a**2)) * \
          ((g.r**2 - a**2) / g.r) * np.cos(g.theta) + \
          (c2-c0) * (1 - (np.log(b / g.r) / np.log(b / a)))
    s.postProcessing()

    headerprint('Table II, p89: radius vs. tangential (hoop) stress', ' ')
    radius_inch = np.linspace(4,12,9)
    radius = Q_(radius_inch, 'inch').to('m')
    sigmaTheta = Q_(np.interp(radius, s.g.r[0,:], s.sigmaTheta[0,:]), 'Pa').to('psi')
    for r, sig_t in zip(radius_inch, sigmaTheta):
        valprint('{} (in.)'.format(r), sig_t.magnitude, 'psi')

    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(), 
               'NACA-TR-1059_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'NACA-TR-1059_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'NACA-TR-1059_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'NACA-TR-1059_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'NACA-TR-1059_sigmaEq.pdf')

def ASTRI2():
    """
    Peak flux reference point in advanced CSP prototypes using Inco625
    """
    headerprint(' NPS Sch. 5S 3/4" Inco625 at 650 degC ')

    ## Material constants
    b = 25.4e-3/2.     # inside tube radius (mm->m)
    valprint('b', b*1e3, 'mm')
    a = b - 1.65e-3    # outside tube radius (mm->m)
    valprint('a', a*1e3, 'mm')
    k = 19.1           # thermal conductivity (kg*m/s^3/K)
    valprint('k', k, 'kg*m/s^3/K')
    alpha = 18.2e-6  # thermal dilation (K^-1)
    valprint('alpha', alpha*1e6, 'x1e6 K^-1')
    E = 169e9          # Youngs modulus (Pa)
    valprint('E', E*1e-9, 'GPa')
    nu = 0.31          # Poisson
    valprint('nu', nu)

    ## Thermal constants
    CG = 7.50e5        # absorbed flux (W/m^2)
    valprint('CG', CG*1e-3, 'kW/m^2')
    mdot = 0.1         # mass flow (kg/s)
    valprint('mdot', mdot, 'kg/s')
    T_int = 888        # bulk sodium temperature (K)
    sodium = liquidSodium(True); sodium.update(T_int)
    #h_int = HTC(True, sodium, a, b, k, 'Skupinski', 'velocity', 4.0)
    #h_int = HTC(True, sodium, a, b, k, 'Skupinski', 'heatCapRate', 5000)
    #h_int = HTC(True, sodium, a, b, k, 'Skupinski', 'mdot', mdot)
    #h_int = HTC(True, sodium, a, b, k, 'Notter', 'mdot', mdot)
    h_int = HTC(True, sodium, a, b, k, 'Chen', 'mdot', mdot)

    """ Create instance of Grid: """
    nr = 17; nt = 61
    g = Grid(nr=nr, nt=nt, rMin=a, rMax=b) # nr, nt -> resolution

    """ Create instance of LaplaceSolver: """
    s = Solver(g, debug=True, CG=CG, k=k, T_int=T_int, R_f=0,
               h_int=h_int, P_i=0e5, alpha=alpha, E=E, nu=nu, n=1,
               bend=False)

    """ External BC: """
    #s.extBC = s.tubeExtTemp
    s.extBC = s.tubeExtCosFlux
    #s.extBC = s.tubeExtConv
    #s.extBC = s.tubeExtCosFluxRadConv
    #s.extBC = s.tubeExtCosFluxRadConvAdiabaticBack

    """ Internal BC: """
    #s.intBC = s.tubeIntTemp
    #s.intBC = s.tubeIntFlux
    s.intBC = s.tubeIntConv
    
    ## Generalised plane strain:
    t = time.clock(); ret = s.solve(eps=1e-6)
    headerprint('Generalised plane strain', ' ')
    s.postProcessing()
    valprint('Time', time.clock() - t, 'sec')

    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(), 
               'NPS5S34_Inco625_GPS_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'NPS5S34_Inco625_GPS_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'NPS5S34_Inco625_GPS_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'NPS5S34_Inco625_GPS_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'NPS5S34_Inco625_GPS_sigmaEq.pdf')

    # Comparison with FEA -- code_aster MECA_STATIQUE [U4.51.01]:
    fc = [None]*4
    for i, theta in enumerate([0, 60, 120, 180]):
        fn = 'NPS5S34_Inco625_theta{}_TSOD615_HTCSOD17394_FLUX750.dat'.format(theta) 
        fc[i] = np.genfromtxt('aster/'+fn, skip_header=5)
    plotFEA(g.r, s.sigmaTheta, fc, 4, 'NPS5S34_Inco625_FEA-GPS_sigmaTheta.pdf',
            'best', r'$\sigma_\theta$')
    plotFEA(g.r, s.sigmaR, fc, 5, 'NPS5S34_Inco625_FEA-GPS_sigmaR.pdf',
            'best', r'$\sigma_r$')
    plotFEA(g.r, s.sigmaZ, fc, 6, 'NPS5S34_Inco625_FEA-GPS_sigmaZ.pdf',
            'best', r'$\sigma_z$')
    plotFEA(g.r, s.sigmaRTheta, fc, 7, 'NPS5S34_Inco625_FEA-GPS_sigmaRTheta.pdf',
            'best', r'$\sigma_{r\theta}$')

    ## Generalised plane strain with annulled bending:
    s.bend = True
    headerprint('Generalised plane strain with annulled bending moment', ' ')
    s.postProcessing()

    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(), 
               'NPS5S34_Inco625_GPS-AB_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'NPS5S34_Inco625_GPS-AB_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'NPS5S34_Inco625_GPS-AB_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'NPS5S34_Inco625_GPS-AB_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'NPS5S34_Inco625_GPS-AB_sigmaEq.pdf')

    ## Sensitivity of heat transfer coefficient and peak stress to mass-flow
    s.bend = False
    s.debug = False
    mdot = np.linspace(0.05, 0.2)
    h_int = np.zeros(len(mdot))
    sig_eq = np.zeros(len(h_int))
    for i, m in enumerate(mdot):
        #h_int[i] = HTC(False, sodium, a, b, k, 'Skupinski', 'mdot', m)
        #h_int[i] = HTC(False, sodium, a, b, k, 'Notter', 'mdot', m)
        h_int[i] = HTC(False, sodium, a, b, k, 'Chen', 'mdot', m)
        s.h_int = h_int[i]
        ret = s.solve(eps=1e-6)
        s.postProcessing()
        sig_eq[i] = s.sigmaEq[0,-1]
    ## plot of mdot vs internal convection coefficient
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(mdot, h_int*1e-3, '-')
    ax.set_xlabel(r'$\dot{m}$ (\si{\kilo\gram\per\second})')
    ax.set_ylabel(r'$h_\mathrm{int}$ (\si{\kilo\watt\per\meter\squared\per\kelvin})')
    fig.tight_layout()
    fig.savefig('NPS5S34_Inco625_mdot-intConv.pdf')
    #fig.savefig('NPS5S34_Inco625_mdot-intConv.png', dpi=150)
    plt.close(fig)
    ## plot of mdot vs maximum equivalent stress
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(mdot, sig_eq*1e-6, '-')
    ax.set_xlabel(r'$\dot{m}$ (\si{\kilo\gram\per\second})')
    ax.set_ylabel(r'$\max(\sigma_\mathrm{Eq})$ (MPa)')
    fig.tight_layout()
    fig.savefig('NPS5S34_Inco625_mdot-sigmaEq.pdf')
    #fig.savefig('NPS5S34_Inco625_mdot-sigmaEq.png', dpi=150)
    plt.close(fig)

""" __________________________ USAGE (MAIN) ___________________________ """

if __name__ == "__main__":

    # Holms1952()
    # SE6413()
    ASTRI2()
