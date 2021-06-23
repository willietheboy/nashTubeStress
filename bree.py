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

import nashTubeStress as nts

################################### PLOTTING ###################################

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

def plotComponentStress(r, sigmaR, sigmaTheta, sigmaZ,
                        sigmaEq, filename, i, loc):
    a = r[0,0]; b = r[0,-1]
    trX = Q_(1, 'inch').to('mm').magnitude
    trY = Q_(1, 'ksi').to('MPa').magnitude
    trans = mtransforms.Affine2D().scale(trX,trY)
    fig = plt.figure(figsize=(4, 3.5))
    ax = SubplotHost(fig, 1, 1, 1)
    axa = ax.twin(trans)
    axa.set_viewlim_mode("transform")
    axa.axis["top"].set_label(r'\textsc{radius}, $r$ (in.)')
    axa.axis["top"].label.set_visible(True)
    axa.axis["right"].set_label(r'\textsc{stress}, $\sigma$ (ksi)')
    axa.axis["right"].label.set_visible(True)
    ax = fig.add_subplot(ax)
    ax.plot(r[i,:]*1e3, sigmaR[i,:]*1e-6, '^-',
            label='$\sigma_r$')
    ax.plot(r[i,:]*1e3, sigmaTheta[i,:]*1e-6, 'o-',
            label=r'$\sigma_\theta$')
    ax.plot(r[i,:]*1e3, sigmaZ[i,:]*1e-6, 'v-',
            label='$\sigma_z$')
    # ax.fill_between(r[i,:]*1e3, sigmaTheta[i,:]*1e-6,
    #                 sigmaR[i,:]*1e-6, color='C3', alpha=0.3)
    # ax.fill_between(r[i,:]*1e3, sigmaZ[i,:]*1e-6, color='C2',
    #                 alpha=0.3)
    ax.plot(r[i,:]*1e3, sigmaEq[i,:]*1e-6, 's-',
            label='$\sigma_\mathrm{eq}$')
    ax.set_xlabel(r'\textsc{radius}, $r$ (mm)')
    ax.set_xlim((a*1e3)-0.1,(b*1e3)+0.1)
    ax.set_ylabel(r'\textsc{stress}, $\sigma$ (MPa)')
    ax.legend(loc=loc)
    #labels = ax.get_xticklabels()
    #plt.setp(labels, rotation=30)
    fig.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotPrincipalStress(r, P1, P2, P3, TG, filename, i, loc):
    a = r[0,0]; b = r[0,-1]
    trX = Q_(1, 'inch').to('mm').magnitude
    trY = Q_(1, 'ksi').to('MPa').magnitude
    trans = mtransforms.Affine2D().scale(trX,trY)
    fig = plt.figure(figsize=(4, 3.5))
    ax = SubplotHost(fig, 1, 1, 1)
    axa = ax.twin(trans)
    axa.set_viewlim_mode("transform")
    axa.axis["top"].set_label(r'\textsc{radius}, $r$ (in.)')
    axa.axis["top"].label.set_visible(True)
    axa.axis["right"].set_label(r'\textsc{stress}, $\sigma$ (ksi)')
    axa.axis["right"].label.set_visible(True)
    ax = fig.add_subplot(ax)
    ax.plot(r[i,:]*1e3, P1[i,:]*1e-6, '^-',
            label=r'$\sigma_1$')
    ax.plot(r[i,:]*1e3, P2[i,:]*1e-6, 'o-',
            label=r'$\sigma_2$')
    ax.plot(r[i,:]*1e3, P3[i,:]*1e-6, 'v-',
            label=r'$\sigma_3$')
    ax.fill_between(r[i,:]*1e3, P1[i,:]*1e-6,
                    P3[i,:]*1e-6, color='k', alpha=0.15)
    # ax.fill_between(r[i,:]*1e3, sigmaZ[i,:]*1e-6, color='C2',
    #                 alpha=0.3)
    ax.plot(r[i,:]*1e3, TG[i,:]*1e-6, 's-',
            label=r'$\sigma_\mathrm{\textsc{mss}}$')
    ax.set_xlabel(r'\textsc{radius}, $r$ (mm)')
    ax.set_xlim((a*1e3)-0.1,(b*1e3)+0.1)
    ax.set_ylabel(r'\textsc{stress}, $\sigma$ (MPa)')
    ax.legend(loc=loc)
    #labels = ax.get_xticklabels()
    #plt.setp(labels, rotation=30)
    fig.tight_layout()
    fig.savefig(filename, transparent=True)
    plt.close(fig)

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

def cauchy(info, sixx, siyy, sizz, sixy, sixz, siyz):
    """
    Input: sig11 sig22 sig33 sig12 sig13 sig23
    Output: various Cauchy stress information if info>0
    Return: Max Shear Stress (Tresca)
    """
    # load the stresses into a matrix and compute the
    # deviatoric and isotropic stress matricies
    sigma = np.array([[sixx,sixy,sixz],
                      [sixy,siyy,siyz],
                      [sixz,siyz,sizz]])

    sigma_iso = 1.0/3.0*np.trace(sigma)*np.eye(3)
    sigma_dev = sigma - sigma_iso

    # compute principal stresses
    eigvals = list(np.linalg.eigvalsh(sigma))
    eigvals.sort()
    eigvals.reverse()

    # compute max shear stress (Tresca)
    tresca = (max(eigvals)-min(eigvals))
    maxshear = tresca/2.0

    # compute the stress invariants
    I1 = np.trace(sigma)
    J2 = 1.0/2.0*np.trace(np.dot(sigma_dev,sigma_dev))
    J3 = 1.0/3.0*np.trace(\
         np.dot(sigma_dev,np.dot(sigma_dev,sigma_dev)))

    # compute other common stress measures
    mean_stress = 1.0/3.0*I1
    eq_stress  = math.sqrt(3.0*J2)

    # compute lode coordinates
    lode_r = math.sqrt(2.0*J2)
    lode_z = I1/math.sqrt(3.0)

    dum = 3.0*math.sqrt(6.0)*np.linalg.det(sigma_dev/lode_r)
    lode_theta = 1.0/3.0*math.asin(dum)

    # compute the stress triaxiality
    triaxiality = mean_stress/eq_stress

    # Print out what we've found if requested
    if info>0:
        headerprint(" Stress Info  ")
        matprint("Input Stress",sigma)
        headerprint(" Component Matricies ")
        matprint("Isotropic Stress",sigma_iso)
        matprint("Deviatoric Stress",sigma_dev)
        #matprint("Eigenvalues",eigvals)
        headerprint(" Scalar Values ")
        valprint("P1",eigvals[0])
        valprint("P2",eigvals[1])
        valprint("P3",eigvals[2])
        valprint("Mean Stress",mean_stress)
        valprint("Max Shear Stress",maxshear)
        valprint("Tresca Stress",tresca)
        valprint("Equivalent (von Mises) Stress",eq_stress)
        valprint("I1",I1)
        valprint("J2",J2)
        valprint("J3",J3)
        valprint("Lode z",lode_z)
        valprint("Lode r",lode_r)
        valprint("Lode theta (rad)",lode_theta)
        valprint("Lode theta (deg)",math.degrees(lode_theta))
        valprint("Triaxiality",triaxiality)
        headerprint(" End Info ")

    return tresca

def tresca(info, sixx, siyy, sizz, sixy, sixz, siyz):
    """
    Input: sig11 sig22 sig33 sig12 sig13 sig23
    Return: Max Shear Stress (Tresca)
    """
    sigma = np.array([[sixx,sixy,sixz],
                      [sixy,siyy,siyz],
                      [sixz,siyz,sizz]])

    # compute principal stresses
    eigvals = list(np.linalg.eigvalsh(sigma))
    eigvals.sort()
    eigvals.reverse()

    # compute max shear stress (Tresca)
    tresca = (max(eigvals)-min(eigvals))
    maxshear = tresca/2.0

    # Print out what we've found if requested
    if info>0:
        headerprint(" Stress Info  ")
        matprint("Input Stress",sigma)
        headerprint(" Scalar Values ")
        valprint("P1",eigvals[0])
        valprint("P2",eigvals[1])
        valprint("P3",eigvals[2])
        valprint("Max Shear Stress",maxshear)
        valprint("Tresca Stress",tresca)
        headerprint(" End Info ")

    return (eigvals[0], eigvals[1], eigvals[2], tresca)

def iterateTensor(func, xx, yy, zz, xy):
    """
    Iterate over tensors for scalar quantities in GPS (no z-shear)
    """
    i, j = xx.shape
    p1 = np.zeros([i,j])
    p2 = np.zeros([i,j])
    p3 = np.zeros([i,j])
    tg = np.zeros([i,j])
    for k in range(i):
        for l in range(j):
            res = func(0, xx[k,l], yy[k,l], zz[k,l], xy[k,l], 0, 0)
            p1[k,l] = res[0]
            p2[k,l] = res[1]
            p3[k,l] = res[2]
            tg[k,l] = res[3]
    return {'TG':tg, 'P1':p1, 'P2':p2, 'P3':p3}

##################################### MAIN #####################################

if __name__ == "__main__":
    """
    Bree, J. Strain Anal., 1967, 2, 226-238
    """
    headerprint(' BREE ')

    nr=10; nt=32
    a = 100e-3      # inside tube radius [mm->m]
    b = 101e-3      # outside tube radius [mm->m]

    R = (a + b)/2.
    wt = b-a

    # ## Material properties reflect 316H at 600 degC
    # E = 150e9
    # alpha = 1.88e-5
    # nu = 0.31
    # sig_y = 114e6

    ## Material properties reflect 316H at 650 degC
    E = 146e9
    alpha = 1.9e-5
    nu = 0.31
    sig_y = 110e6

    valprint('E', E*1e-9, 'GPa')
    valprint('alpha', alpha*1e6, 'x1e6 K^-1')
    valprint('sig_y', sig_y*1e-6, 'MPa')

    ## Prescribe actual pressure and temperature difference across wall:
    # P = 5e5 # Pa
    # dT = 50
    # sig_p = P * R / wt
    # sig_t = E * alpha * dT / (2 * (1 - nu))

    ## Bree coordinates:
    X = 0.5
    valprint('X', X)
    Y = 1
    valprint('Y', Y)
    # Primary membrane stress
    sig_p = sig_y * X
    p = sig_p * wt / R
    valprint('p', p*1e-6, 'MPa')
    valprint('sig_p', sig_p*1e-6, 'MPa')
    ## Secondary peak stress:
    sig_t = sig_y * Y
    valprint('max. sig_t', sig_t*1e-6, 'MPa')
    dT = sig_t * (2 * (1 - nu)) / (alpha * E)
    valprint('dT', dT, 'K')


    ## True: generalised plane strain (default); False: simple plane strain:
    GPS = True

    ## Create nashTubeStress instance of Grid and Solver:
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    s = nts.Solver(g, debug=False, alpha=alpha,
                   E=E, nu=nu, n=1, GPS=GPS,
                   P_i=p)
    ## Only pressure:
    s.postProcessing()
    plotComponentStress(
        g.r, s.sigmaR, s.sigmaTheta, s.sigmaZ,
        s.sigmaEq, 'Bree_comp-pres.pdf', 0, 'best')
    c = iterateTensor(tresca, s.sigmaR, s.sigmaTheta, s.sigmaZ, s.sigmaRTheta)
    plotPrincipalStress(
        g.r, c['P1'], c['P2'], c['P3'],
        c['TG'], 'Bree_princ-pres.pdf', 0, 'best')

    ## Apply analytical temperature (without pressure):
    #s.T = dT * ((np.log(g.r / a) / np.log(b / a)))
    s.P_i=0
    s.T = dT * ((g.r - a) / (b - a))
    s.postProcessing()
    plotComponentStress(
        g.r, s.sigmaR, s.sigmaTheta, s.sigmaZ,
        s.sigmaEq, 'Bree_comp-therm.pdf', 0, 'best')
    c = iterateTensor(tresca, s.sigmaR, s.sigmaTheta, s.sigmaZ, s.sigmaRTheta)
    plotPrincipalStress(
        g.r, c['P1'], c['P2'], c['P3'],
        c['TG'], 'Bree_princ-therm.pdf', 0, 'best')

    ## Apply both pressure and temperature:
    #s.T = dT * ((np.log(g.r / a) / np.log(b / a)))
    s.P_i=p
    s.T = dT * ((g.r - a) / (b - a))
    s.postProcessing()
    plotComponentStress(
        g.r, s.sigmaR, s.sigmaTheta, s.sigmaZ,
        s.sigmaEq, 'Bree_comp-total.pdf', 0, 'best')
    c = iterateTensor(tresca, s.sigmaR, s.sigmaTheta, s.sigmaZ, s.sigmaRTheta)
    plotPrincipalStress(
        g.r, c['P1'], c['P2'], c['P3'],
        c['TG'], 'Bree_princ-total.pdf', 0, 'best')
