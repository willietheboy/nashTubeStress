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

from math import exp, log, sqrt, pi, ceil, floor, asin
import numpy as np
from numpy import ma
import scipy.optimize as opt

#################################### CLASSES ###################################

class Grid:

    """
    A cylindrical coordinate (theta, r) grid class
    -- units are m, sec, K, W, Pa
    """

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
        # create some useful BC variables
        self.cosTheta = np.cos(theta)
        self.sinTheta = np.sin(theta)
        self.fullTube = np.ones(len(theta))
        self.halfTube = np.ones(len(theta))
        self.halfTube[self.cosTheta<0] = 0.0

class Solver:

    """  A Laplacian solver for steady-state conduction in cylinders
         -- Gauss-Seidel iteration of T(r, theta)
         -- bi-harmonic thermoelastic stress post-processing """

    # Constants:
    sigma = 5.67e-8    # Stefan-Boltzmann

    def __init__(self, grid, debug=False, it='numpy', CG=8.5e5,
                 k=21, T_int=723.15, h_int=10e3, U=4.0, R_f=0., A=0.968,
                 epsilon=0.87, T_ext=293.15, h_ext=30., P_i=0e5,
                 T_0=0., alpha=20e-6, E=165e9, nu=0.31, n=1,
                 bend=False, GPS=True):
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
        self.T_0 = T_0          # stress free temperature
        self.alpha = alpha      # thermal expansion coefficienct of tube
        self.E = E              # Modulus of elasticity
        self.nu = nu            # Poisson's coefficient
        self.n = n              # Number of Fourier 'frequencies'
        self.bend = bend        # switch to allow tube bending
        self.GPS = GPS          # switch to turn off generalised plane strain
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
        """ DEPRECATED blitz using weave """
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
        """ DEPRECATED inline C code """
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
        # elif iterator == 'blitz':
        #     self.iterate = self.blitzStep
        # elif iterator == 'inline':
        #     self.iterate = self.inlineStep
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
        # self.heatFluxBalance()
        self.linearElasticStress()
        # self.babcockAndWilcoxStress()
        return

    ############################ BOUNDARY CONDITIONS ###########################

    def symmetryBC(self):
        """ Sets the left and right symmetry BCs """
        self.meshT[0, 1:-1] = self.meshT[2, 1:-1]
        self.meshT[-1, 1:-1] = self.meshT[-3, 1:-1]

    def extTubeHalfTemp(self):
        """ fixedValue boundary condition """
        self.meshT[:,-1] = self.T_ext

    def extTubeHalfConv(self):
        """ Convective boundary condition """
        self.meshT[:, -1] = (self.meshT[:,-2] + \
                             ((self.g.dr * self.h_ext /
                               self.k) * self.T_ext)) \
            / (1 + (self.g.dr * self.h_ext / self.k))

    def extTubeHalfFlux(self):
        """ Heat flux boundary condition """
        self.meshT[:,-1] = ((self.g.dr * self.CG) /
                            self.k) + self.meshT[:, -2]

    def extTubeHalfCosFlux(self):
        """ 100% absorbed cosine flux boundary condition """
        self.phi_inc = (self.g.halfTube * \
                            self.CG * self.g.cosTheta)
        phi_t = self.phi_inc
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (phi_t * self.g.dr / self.k)

    def extTubeHalfCosFluxRadConv(self):
        """ Heat flux, re-radiation and convection boundary condition """
        self.phi_inc = (self.g.halfTube * \
                            self.CG * self.g.cosTheta)
        phi_t = self.phi_inc * self.A \
                      - (self.sigma * self.epsilon \
                         * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                      - (self.h_ext * (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (phi_t * self.g.dr / self.k)

    def extTubeFullCosFluxRadConv(self):
        """ Heat flux, re-radiation and convection boundary condition """
        self.phi_inc = (self.CG * np.abs(self.g.cosTheta))
        phi_t = self.phi_inc * self.A \
                      - (self.sigma * self.epsilon \
                         * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                      - (self.h_ext * (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (phi_t * self.g.dr / self.k)

    def extTubeFluxProfileRadConv(self):
        """ Heat flux profile, re-radiation and convection boundary condition """
        phi_t = self.phi_inc * self.A \
                      - (self.sigma * self.epsilon \
                         * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                      - (self.h_ext * (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (phi_t * self.g.dr / self.k)

    def extTubeHalfCosFluxRadConvAdiabaticBack(self):
        """ Heat flux, re-radiation and convection boundary condition """
        self.phi_inc = (self.g.halfTube * \
                            self.CG * self.g.cosTheta)
        phi_t = self.phi_inc * self.A \
                      - (self.g.halfTube * self.sigma * self.epsilon \
                         * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                      - (self.h_ext * self.g.halfTube *
                         (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (phi_t * self.g.dr / self.k)

    def extTubeFWFluxRadConv(self):
        """ Heat flux, re-radiation and convection boundary condition
        -- Heat flux profile taken from Foster Wheeler report U.S. DOE
           Cooperative Agreement No. DE-EE0003596 """
        self.phi_inc = ((self.g.halfTube * self.CG) * \
                        (3 - 3*self.g.sinTheta + 2*self.g.cosTheta * \
                         np.sqrt(1 - self.g.sinTheta)) / \
                        (5 - 4*self.g.sinTheta))
        phi_t = self.phi_inc * self.A \
                - (self.g.halfTube * self.sigma * self.epsilon \
                   * (self.meshT[:,-1]**4 - self.T_ext**4)) \
                   - (self.h_ext * self.g.halfTube * \
                      (self.meshT[:,-1] - self.T_ext))
        self.meshT[:,-1] = self.meshT[:,-2] + \
                           (phi_t * self.g.dr / self.k)

    def intTubeTemp(self):
        """ fixedValue boundary condition """
        self.meshT[:,0] = self.T_int

    def intTubeFlux(self):
        """ Heat flux boundary condition """
        self.meshT[:,0] = ((self.g.dr * self.CG) /
                           self.k) + self.meshT[:, 1]

    def intTubeConv(self):
        """ Convective boundary condition to tube flow with fouling """
        U = 1 / (self.R_f + (1 / self.h_int))
        self.meshT[:,0] = (self.meshT[:,1] + \
                           ((self.g.dr * U / self.k) * \
                            self.T_int)) \
            / (1 + (self.g.dr * U / self.k))

    ############################## POST-PROCESSING #############################

    def heatFluxBalance(self):
        """
        Calculate the heat flux for inner and outer tube surfaces:
        -- currently nashTubeStress is configured for HALF A TUBE,
           so the integral of flux of surface area needs multiplied
           by two (2x) for a whole tube circumference
        """
        self.phi_int = self.k * (self.T[:,0] - self.T[:,1]) \
                        / self.g.dr
        self.q_int = - self.g.sfRmin * self.phi_int
        self.phi_ext = self.k * (self.T[:,-1] - self.T[:,-2]) \
                        / self.g.dr
        self.q_ext = self.g.sfRmax * self.phi_ext
        # tube efficiency eta_tube:
        #self.phi_inc = - self.phi_inc[1:-1]
        self.q_inc = self.g.sfRmax * -self.phi_inc[1:-1]
        self.eta_tube = abs(np.sum(self.q_int) / np.sum(self.q_inc))
        if self.debug:
            headerprint('Half tube heat balance:', ' ')
            valprint('sum(phi_q->to)', np.sum(self.q_inc)*1e-3, 'kW/m')
            valprint('sum(phi_q,to->ti)', np.sum(self.q_ext)*1e-3, 'kW/m')
            valprint('sum(phi_q,ti->f)', np.sum(self.q_int)*1e-3, 'kW/m')
            valprint('eta_t', self.eta_tube*1e2, '%')

    def linearElasticStress(self):
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
        # Plane strain override (needs T_0 defined):
        if not self.GPS:
            QZ = (nu*(QR + QTheta)) - (E*alpha*(T-self.T_0))
        QRTheta = C * kappa_tau * (1 - (a2 / meshR2)) * (1 - (b2 / meshR2))
        QEq = np.sqrt(0.5 * ((QR - QTheta)**2 + \
                             (QTheta - QZ)**2 + \
                             (QZ - QR)**2) + \
                      6 * (QRTheta**2))
        # Pressure stress component:
        PR = ((a2 * self.P_i) / (b2 - a2)) * (1 - (b2 / meshR2))
        PTheta = ((a2 * self.P_i) / (b2 - a2)) * (1 + (b2 / meshR2))
        ## generalised plane strain:
        PZ =  (a2 * self.P_i) / (b2 - a2)
        if not self.GPS:
            PZ =  2*nu*(a2 * self.P_i) / (b2 - a2)
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
            headerprint('Tube temperatures', ' ')
            valprint('max T', np.max(self.T)-273.15, 'C')
            valprint('min T', np.min(self.T)-273.15, 'C')
            headerprint('Biharmonic coefficients:', ' ')
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

    def babcockAndWilcoxStress(self):
        """ Kistler (1987), Kolb (2011) from Babcock & Wilcox (1984)
        -- 'G' refers to Goodier (1937a,b)
        """
        A, CG = self.A, self.CG
        T_int, k, h_int = self.T_int, self.k, np.average(self.h_int)
        a, b = self.g.a, self.g.b
        alpha, E, nu = self.alpha, self.E, self.nu
        ## Assumes you have run the linearElasticStress():
        Tbar_i = self.popt1[0]; BP = self.popt1[1]; DP = self.popt1[2];
        Tbar_o = self.popt2[0]; BPP = self.popt2[1]; DPP = self.popt2[2];
        T_ci = T_int + ((A*CG*(b/a)) / h_int)
        T_co = T_ci + A*CG * (b/k) * log(b/a)
        T_mc = (T_ci + T_co) / 2.
        T_m = T_int + (1 / np.pi) * (T_mc - T_int)
        sigmaR = 0.0
        sigmaTheta = alpha * E * ((T_co - T_ci) / (2*(1-nu)))
        sigmaThetaG = - ((alpha * E) / (2*(1-nu))) \
                      * ((Tbar_o - Tbar_i) \
                         + np.sqrt((BPP - BP)**2 + (DPP - DP)**2))
        sigmaZ = alpha * E * (T_mc - T_m)
        sigmaZG = alpha * E * (- 0.5*(T_ci + T_co) \
                               + 0.5*(T_ci - T_co) + 0.5*(Tbar_i + Tbar_o))# \
        sigmaBnW = sigmaZ + sigmaTheta
        sigmaEq = np.sqrt(0.5 * ((sigmaR - sigmaTheta)**2 + \
                                 (sigmaTheta - sigmaZ)**2 + \
                                 (sigmaZ - sigmaR)**2))
        sigmaEqG = np.sqrt(0.5 * ((sigmaR - sigmaThetaG)**2 + \
                                 (sigmaThetaG - sigmaZG)**2 + \
                                 (sigmaZG - sigmaR)**2))
        sigmaCheck = alpha * E * (0.954*(T_co - T_int) + \
                                  0.523*(T_co - T_ci)) / 1.4
        if self.debug:
            headerprint(r'Babcock & Wilcox (1984), SAND82-8178:', ' ')
            valprint('T_m', T_m, 'K')
            valprint('T_mc', T_mc, 'K')
            valprint('T_co', T_co, 'K')
            valprint('T_ci', T_ci, 'K')
            valprint('T_co - T_ci', T_co-T_ci, 'dT')
            valprint('h_int', h_int*1e-3, 'kW/m^2')
            valprint('sigmaR', sigmaR*1e-6, 'MPa')
            valprint('sigmaTheta', sigmaTheta*1e-6, 'MPa')
            valprint('sigmaZ', sigmaZ*1e-6, 'MPa')
            valprint('sigmaEq', sigmaEq*1e-6, 'MPa')
            valprint('sigmaB\&W', sigmaBnW*1e-6, 'MPa')
            valprint('sigmaCheck', sigmaCheck*1e-6, 'MPa')
            valprint('sigmaTheta_G', sigmaThetaG*1e-6, 'MPa')
            valprint('sigmaZ_G', sigmaZG*1e-6, 'MPa')
            valprint('sigmaEq_G', sigmaEqG*1e-6, 'MPa')

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

def fourierTheta(theta, a0, *c):
    """ Timoshenko & Goodier equation """
    ret = a0 + np.zeros(len(theta))
    for i, n in [*zip(range(0,len(c),2), range(1,int(len(c)/2)+1))]:
        ret += (c[i] * np.cos(n * theta)) + \
            (c[i+1] * np.sin(n * theta))
    return ret
