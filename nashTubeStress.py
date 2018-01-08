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

See also:
 -- Solar Energy 160 (2018) 368-379
 -- https://doi.org/10.1016/j.solener.2017.12.003
"""

import sys, time, os
from math import exp, log, sqrt, pi
import numpy as np
from numpy import ma
from scipy import weave
from scipy.weave import converters
import scipy.optimize as opt

# Plotting:
import matplotlib as mpl
import matplotlib.pyplot as plt
params = {'text.latex.preamble': [r'\usepackage{mathptmx,txfonts}']}
plt.rcParams.update(params)
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
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator, DictFormatter)
import colormaps as cmaps # magma, inferno, plasma, viridis

""" ________________________ PLOTTING FUNCTIONS _______________________ """

def plotStress(theta, r, sigma, sigmaMin, sigmaMax, filename):
    fig = plt.figure(figsize=(2.5, 3))    
    fig.subplots_adjust(left=-1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90))
    #cmap = cm.get_cmap('jet')
    cmap = cmaps.magma # magma, inferno, plasma, viridis...
    levels = ticker.MaxNLocator(nbins=10).tick_values(sigmaMin*1e-6, sigmaMax*1e-6)
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
    fig.savefig(filename, transparent=True)
    plt.close(fig)

""" ____________________________ FUNCTIONS ____________________________ """

def fourierTheta(theta, a0, *c):
    """ Timoshenko & Goodier equation """
    ret = a0 + np.zeros(len(theta))
    for i, n in zip(range(0,len(c),2), range(1,(len(c)/2)+1)):
        ret += (c[i] * np.cos(n * theta)) + (c[i+1] * np.sin(n * theta))
    return ret

""" ________________________ CLASS DEFINITIONS ________________________ """

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
               #line 000 "laplacianCylinder.py"
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

    """ _____________________ BOUNDARY CONDITIONS _____________________ """

    def symmetryBC(self):        
        """ Sets the left and right symmetry BCs """       
        self.meshT[0, 1:-1] = self.meshT[2, 1:-1]
        self.meshT[-1, 1:-1] = self.meshT[-3, 1:-1]

    def tubeExtTemp(self):
        """ fixedValue boundary condition """        
        self.meshT[:,-1] = self.T_ext

    def tubeExtFlux(self):        
        """ Heat flux boundary condition """
        self.meshT[:,-1] = ((self.g.dr * self.CG) / 
                            self.k) + self.meshT[:, -2]

    def tubeExtConv(self):
        """ Convective boundary condition """        
        self.meshT[:, -1] = (self.meshT[:,-2] + \
                             ((self.g.dr * self.h_ext / 
                               self.k) * self.T_ext)) \
            / (1 + (self.g.dr * self.h_ext / self.k))

    def tubeExtFluxRadConv(self): 
        """ Heat flux, re-radiation and convection boundary condition """
        self.heatFluxInc = (self.g.tubeFront * \
                            self.CG * self.g.cosTheta)
        heatFluxAbs = self.heatFluxInc * self.A \
                      - (self.sigma * self.epsilon \
                         * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                      - (self.h_ext * (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (heatFluxAbs * self.g.dr / self.k)

    def tubeExtFluxRadConvAdiabaticBack(self): 
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

    """ _______________________ POST-PROCESSING _______________________ """

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
        p0 = [1.0] * (1 + (s.n * 2))
        # inside:
        popt1, pcov1 = opt.curve_fit(fourierTheta, theta, T[:,0], p0)
        B0 = popt1[0]; BP = popt1[1]; BPP = popt1[2];
        # outside:
        popt2, pcov2 = opt.curve_fit(fourierTheta, theta, T[:,-1], p0)
        D0 = popt2[0]; DP = popt2[1]; DPP = popt2[2];
        kappa = (( (((BP * b) - (DP * a)) / (b2 + a2)) 
                   * np.cos(meshTheta)) + \
                 ( (((BPP * b) - (DPP * a)) / (b2 + a2)) 
                   * np.sin(meshTheta))) * \
            (meshR * a * b) / (b2 - a2)
        kappa_ = (( (((BP * b) - (DP * a)) / (b2 + a2)) 
                   * np.sin(meshTheta)) - \
                 ( (((BPP * b) - (DPP * a)) / (b2 + a2)) 
                   * np.cos(meshTheta))) * \
            (meshR * a * b) / (b2 - a2)
        if self.bend:
            kappaP = meshR * ((((BP * a) + (DP * b)) / (b2 + a2) * \
                               np.cos(meshTheta)) \
                              + (((BPP * a) + (DPP * b)) / (b2 + a2) * \
                              np.sin(meshTheta)))
        else: kappaP = 0.0
        # Axisymmetrical thermal stress component:
        C0 = ((alpha * E * (B0 - D0)) / (2*(1 - nu)*np.log(b/a)))
        QR = C0 * (- np.log(b/meshR) - \
                      (a2/(b2 - a2) * (1 - b2/meshR2) * np.log(b/a)))
        QTheta = C0 * (1 - np.log(b/meshR) - \
                          (a2/(b2 - a2) * (1 + b2/meshR2) * np.log(b/a)))
        QZ = C0 * (1 - (2*np.log(b/meshR)) - (2 * a2 / (b2 - a2) * \
                                                  np.log(b/a)))
        # Nonaxisymmetrical T:
        T_theta = T - ((B0 - D0) * np.log(b / meshR) / np.log(b / a)) - D0
        self.T_theta = T_theta
        # Nonaxisymmetric thermal stress component:
        C1 = (alpha * E) / (2 * (1 - nu))
        QR += C1 * kappa * (1 - (a2 / meshR2)) * (1 - (b2 / meshR2))
        QTheta += C1 * kappa * (3 - ((a2 + b2) / meshR2) - \
                                    ((a2 * b2) / meshR4))
        QZ += alpha * E * ((kappa * (nu / (1 - nu)) * \
                                (2 - ((a2 + b2) / meshR2))) + \
                               kappaP - T_theta)
        QRTheta = C1 * kappa_ * (1 - (a2 / meshR2)) * (1 - (b2 / meshR2))
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
            print 'Timoshenko & Goodier method:'
            print '\tB0 : {}'.format(B0)
            print '\tB\'1 : {}'.format(BP)
            print '\tB\'\'1 : {}'.format(BPP)
            print '\tD0 : {}'.format(D0)
            print '\tD\'1 : {}'.format(DP)
            print '\tD\'\'1 : {}'.format(DPP)
            print '\tsigmaR : {0:g} [MPa]'.format(self.sigmaR[0,-1]*1e-6)
            print '\tsigmaRTheta : {0:g} [MPa]'.format(self.sigmaRTheta[0,-1]*1e-6)
            print '\tsigmaTheta : {0:g} [MPa]'.format(self.sigmaTheta[0,-1]*1e-6)
            print '\tsigmaZ : {0:g} [MPa]'.format(self.sigmaZ[0,-1]*1e-6)
            print '\tmax(sigmaEq) : {0:g} [MPa]'.format(np.max(self.sigmaEq)*1e-6)

""" __________________________ USAGE (MAIN) ___________________________ """

if __name__ == "__main__":

    """ NPS Sch. 5S 1" SS316 at 450degC """
    a = 30.098/2e3     # inside tube radius [mm->m]
    b = 33.4/2e3       # outside tube radius [mm->m]

    """ Create instance of Grid: """
    g = Grid(nr=10, nt=91, rMin=a, rMax=b) # nr, nt -> resolution

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
    #s.extBC = s.tubeExtFluxRadConv
    s.extBC = s.tubeExtFluxRadConvAdiabaticBack

    """ Internal BC: """
    #s.intBC = s.tubeIntTemp
    #s.intBC = s.tubeIntFlux
    s.intBC = s.tubeIntConv

    print('Nitrate salt:')
    s.h_int = 10e3
    t = time.clock(); ret = s.solve(eps=1e-6); s.postProcessing()
    print 'Time : {0:g} [sec]'.format(time.clock() - t)

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
               'nitrateSalt_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'nitrateSalt_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'nitrateSalt_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'nitrateSalt_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'nitrateSalt_sigmaEq.pdf')

    print('\nLiquid Sodium:')
    s.h_int = 40e3
    t = time.clock(); ret = s.solve(eps=1e-6); s.postProcessing()
    print 'Time : {0:g} [sec]'.format(time.clock() - t)

    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(), 
               'liquidSodium_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(), 
               'liquidSodium_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta, 
               s.sigmaRTheta.min(), s.sigmaRTheta.max(), 
               'liquidSodium_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ, 
               s.sigmaZ.min(), s.sigmaZ.max(), 
               'liquidSodium_sigmaZ.pdf')
    plotStress(g.theta, g.r, s.sigmaEq, 
               s.sigmaEq.min(), s.sigmaEq.max(),
               'liquidSodium_sigmaEq.pdf')
