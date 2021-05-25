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

################################### PLOTTING ###################################

import matplotlib as mpl
import matplotlib.pyplot as plt
## uncomment following if not in ~/.config/matplotlib/matplotlibrc already
#params = {'text.latex.preamble': [r'\usepackage{newtxtext,newtxmath,siunitx}']}
#plt.rcParams.update(params)
mpl.rc('figure.subplot', bottom=0.13, top=0.95)
mpl.rc('figure.subplot', left=0.15, right=0.95)
from matplotlib import colors, ticker, cm
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.projections import PolarAxes
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
from mpl_toolkits.axisartist.grid_finder import \
    (FixedLocator, MaxNLocator, DictFormatter)

def plotTemperatureAnnotate(theta, r, T, TMin, TMax, filename):
    fig = plt.figure(figsize=(3, 3.25))
    fig.subplots_adjust(left=-1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90))
    #cmap = cmaps.magma # magma, inferno, plasma, viridis...
    cmap = cm.get_cmap('magma')
    levels = ticker.MaxNLocator(nbins=10).tick_values(TMin-273.15, TMax-273.15)
    cf = ax.contourf(theta, r, T-273.15, levels=levels, cmap=cmap)
    ax.set_rmin(0)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r'\textsc{temperature}, $T$ (\si{\celsius})')
    ax.patch.set_visible(False)
    ax.spines['polar'].set_visible(False)
    gridlines = ax.get_xgridlines()
    ticklabels = ax.get_xticklabels()
    for i in range(5, len(gridlines)):
        gridlines[i].set_visible(False)
        ticklabels[i].set_visible(False)
    ax.annotate(r'\SI{'+'{0:.0f}'.format(T.max()-273.15)+'}{\celsius}', \
                 xy=(theta[0,-1], r[0,-1]), \
                 xycoords='data', xytext=(40, 10), \
                 textcoords='offset points', fontsize=12, \
                 arrowprops=dict(arrowstyle='->'))
    ax.grid(axis='y', linewidth=0)
    ax.grid(axis='x', linewidth=0.2)
    plt.setp(ax.get_yticklabels(), visible=False)
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotStress(theta, r, sigma, sigmaMin, sigmaMax, filename):
    fig = plt.figure(figsize=(3.5, 3.5))
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
    fig.savefig(filename, transparent=True)
    plt.close(fig)

def plotStressAnnotate(theta, r, sigma, sigmaMin, sigmaMax, annSide, filename):
    annSide = -70 if annSide=='left' else 40
    fig = plt.figure(figsize=(3, 3.25))
    fig.subplots_adjust(left=-1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(bottom=0.1)
    fig.subplots_adjust(top=0.9)
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.radians(90))
    cmap = cm.get_cmap('magma')
    #sigmaMin = np.min(sigma*1e-6); sigmaMax = np.max(sigma*1e-6)
    levels = ticker.MaxNLocator(nbins=10).tick_values(sigmaMin*1e-6, sigmaMax*1e-6)
    cf = ax.contourf(theta, r, sigma*1e-6, levels=levels, cmap=cmap)
    ax.set_rmin(0)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r'\textsc{equivalent stress}, $\sigma_\mathrm{eq}$ (MPa)')
    ax.patch.set_visible(False)
    ax.spines['polar'].set_visible(False)
    gridlines = ax.get_xgridlines()
    ticklabels = ax.get_xticklabels()
    for i in range(5, len(gridlines)):
        gridlines[i].set_visible(False)
        ticklabels[i].set_visible(False)
    #annInd = np.unravel_index(s.sigmaEq.argmax(), s.sigmaEq.shape)
    annInd = (0, -1)
    ax.annotate('\SI{'+'{0:.0f}'.format(np.max(sigma*1e-6))+'}{\mega\pascal}', \
                 xy=(theta[annInd], r[annInd]), \
                 xycoords='data', xytext=(annSide, 10), \
                 textcoords='offset points', fontsize=12, \
                 arrowprops=dict(arrowstyle='->'))
    ax.grid(axis='y', linewidth=0)
    ax.grid(axis='x', linewidth=0.2)
    plt.setp(ax.get_yticklabels(), visible=False)
    fig.savefig(filename, transparent=True)
    plt.close(fig)

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
    axa.axis["right"].set_label(r'\textsc{stress component}, $\sigma$ (ksi)')
    axa.axis["right"].label.set_visible(True)
    ax = fig.add_subplot(ax)
    ax.plot(r[i,:]*1e3, sigmaR[i,:]*1e-6, '^-',
            label='$\sigma_r$')
    ax.plot(r[i,:]*1e3, sigmaTheta[i,:]*1e-6, 'o-',
            label=r'$\sigma_\theta$')
    ax.plot(r[i,:]*1e3, sigmaZ[i,:]*1e-6, 'v-',
            label='$\sigma_z$')
    ax.plot(r[i,:]*1e3, sigmaEq[i,:]*1e-6, 's-',
            label='$\sigma_\mathrm{eq}$')
    ax.set_xlabel(r'\textsc{radius}, $r$ (mm)')
    ax.set_xlim((a*1e3)-0.1,(b*1e3)+0.1)
    ax.set_ylabel(r'\textsc{stress component}, $\sigma$ (MPa)')
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

def findFlux(flux, s, f, i, point):
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
    elif point=='membrane':
        # Assuming shakedown has occured (membrane stress remains):
        sigmaEqMax = np.interp(np.average(s.T[0,:]), f[:,0], f[:,i])
        return sigmaEqMax - np.average(s.sigmaEq[0,:])
    else: sys.exit('Variable point {} not recognised'.format(point))

##################################### MAIN #####################################

if __name__ == "__main__":
    """
    Reproduction of results from Solar Energy 160 (2018) 368-379
    https://doi.org/10.1016/j.solener.2017.12.003
    """

    headerprint(' NPS Sch. 5S 1" S31609 in range 450-600degC ')
    k = 21; alpha=20e-6; E = 165e9; nu = 0.31
    h_ext=30 # convective loss due to wind W/(m2.K)
    nr=30; nt=91
    a = 30.098/2e3     # inside tube radius [mm->m]
    b = 33.4/2e3       # outside tube radius [mm->m]

    """ Create instance of Grid: """
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b) # nr, nt -> resolution

    """ Create instance of LaplaceSolver: """
    s = nts.Solver(g, debug=True, CG=0.85e6, k=k, T_int=723.15, R_f=0,
                   A=0.968, epsilon=0.87, T_ext=293.15, h_ext=h_ext,
                   P_i=0e5, alpha=alpha, E=E, nu=nu, n=1,
                   bend=False)

    """ Any of the properties defined above can be changed, e.g.: """
    # s.CG = 1.2e5 ...

    """ External BC: """
    #s.extBC = s.extTubeHalfTemp
    #s.extBC = s.extTubeHalfFlux
    #s.extBC = s.extTubeHalfConv
    s.extBC = s.extTubeHalfCosFluxRadConv
    #s.extBC = s.extTubeHalfCosFluxRadConvAdiabaticBack

    """ Internal BC: """
    #s.intBC = s.intTubeTemp
    #s.intBC = s.intTubeFlux
    s.intBC = s.intTubeConv

    salt = coolant.nitrateSalt(True); salt.update(723.15)
    h_int, dP = coolant.HTC(True, salt, a, b, 20, 'Dittus', 'mdot', 5)
    s.h_int = h_int
    t = time.perf_counter(); ret = s.solve(eps=1e-6); s.postProcessing()
    valprint('Time', time.perf_counter() - t, 'sec')

    """ To access the temperature distribution: """
    #     s.T[theta,radius] using indexes set by nr and nt
    """ e.g. s.T[0,-1] is outer tube front """

    """ Same goes for stress fields: """
    #     s.sigmaR[theta,radius]
    #     s.sigmaTheta[theta,radius]
    #     s.sigmaZ[theta,radius]
    #     s.sigmaEq[theta,radius]

    plotTemperatureAnnotate(g.theta, g.r, s.T,
                            s.T.min(), s.T.max(),
                            'S31609_nitrateSalt_T.pdf')
    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(),
               'S31609_nitrateSalt_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(),
               'S31609_nitrateSalt_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta,
               s.sigmaRTheta.min(), s.sigmaRTheta.max(),
               'S31609_nitrateSalt_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ,
               s.sigmaZ.min(), s.sigmaZ.max(),
               'S31609_nitrateSalt_sigmaZ.pdf')
    plotStressAnnotate(g.theta, g.r, s.sigmaEq,
                       s.sigmaEq.min(), s.sigmaEq.max(),
                       'right', 'S31609_nitrateSalt_sigmaEq.pdf')

    sodium = coolant.liquidSodium(True); sodium.update(723.15)
    h_int, dP = coolant.HTC(True, sodium, a, b, 20, 'Skupinski', 'mdot', 4)
    s.h_int = h_int
    t = time.perf_counter(); ret = s.solve(eps=1e-6); s.postProcessing()
    valprint('Time', time.perf_counter() - t, 'sec')

    plotTemperatureAnnotate(g.theta, g.r, s.T,
                            s.T.min(), s.T.max(),
                            'S31609_liquidSodium_T.pdf')
    plotStress(g.theta, g.r, s.sigmaR,
               s.sigmaR.min(), s.sigmaR.max(),
               'S31609_liquidSodium_sigmaR.pdf')
    plotStress(g.theta, g.r, s.sigmaTheta,
               s.sigmaTheta.min(), s.sigmaTheta.max(),
               'S31609_liquidSodium_sigmaTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaRTheta,
               s.sigmaRTheta.min(), s.sigmaRTheta.max(),
               'S31609_liquidSodium_sigmaRTheta.pdf')
    plotStress(g.theta, g.r, s.sigmaZ,
               s.sigmaZ.min(), s.sigmaZ.max(),
               'S31609_liquidSodium_sigmaZ.pdf')
    plotStressAnnotate(g.theta, g.r, s.sigmaEq,
                       s.sigmaEq.min(), s.sigmaEq.max(),
                       'right', 'S31609_liquidSodium_sigmaEq.pdf')

    headerprint(' Tube property parameter variation ')

    iterator='numpy'
    nr=12; nt=61

    trX = Q_(1, 'ksi').to('MPa').magnitude
    trans = mtransforms.Affine2D().scale(trX,1)
    fig1 = plt.figure(figsize=(5, 3))
    ax1 = SubplotHost(fig1, 1, 1, 1)
    ax1a = ax1.twin(trans)
    ax1a.set_viewlim_mode("transform")
    ax1a.axis["top"].set_label(r'\textsc{max. equiv. stress}, '+\
                               '$\max\sigma_\mathrm{Eq}$ (ksi)')
    ax1a.axis["top"].label.set_visible(True)
    ax1a.axis["right"].label.set_visible(False)
    ax1a.axis["right"].major_ticklabels.set_visible(False)
    ax1a.axis["right"].major_ticks.set_visible(False)
    ax1 = fig1.add_subplot(ax1)

    P_i = 0e5          # internal pipe pressure

    # salt
    h_int = 10e3

    align = ['left', 'center', 'right']
    alignRev = ['right', 'center', 'left']
    alignCen = ['center', 'center', 'center']
    edgeSalt = 'C0'
    fillSalt = 'C0'
    edgeSod = 'C1'
    fillSod = 'C1'
    fill = 'white'
    fs=11

    # DN 15 to 50 (1" is DN 25):
    labels = np.array([15, 25, 50])
    d = np.array([21.34, 33.4, 60.33]) * 1e-3
    stressSalt = np.zeros(len(d))
    for i in range(len(d)):
        b = d[i] / 2.
        a = b - 1.651e-3
        g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b)
        s = nts.Solver(grid=g, it=iterator, h_int=h_int, P_i=P_i)
        s.extBC = s.extTubeHalfCosFluxRadConv
        s.intBC = s.intTubeConv
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSalt[i] = np.max(s.sigmaEq)
    stressSalt *= 1e-6
    bar1 = (stressSalt[-1], stressSalt[1]-stressSalt[-1])
    bar2 = (stressSalt[1], stressSalt[0]-stressSalt[1])
    ax1.broken_barh([bar1, bar2], (1, 4),
                    facecolors=(fillSalt, fill),
                    edgecolor=edgeSalt,)
    for i in np.array([0,2]):#range(len(labels)):
        ax1.annotate('${0:g}$'.format(labels[i]), \
                     xy=(stressSalt[i], 6), \
                     xycoords='data', horizontalalignment=align[i],
                     fontsize=fs)

    # tube conductivity from 10 to 30 [W/(m.K)]:
    b = 33.4 / 2e3
    a = b - 1.651e-3
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    s = nts.Solver(grid=g, it=iterator, h_int=h_int, P_i=P_i)
    s.extBC = s.extTubeHalfCosFluxRadConv
    s.intBC = s.intTubeConv
    conductivity = np.array([15, 20, 25])
    stressSalt = np.zeros(len(conductivity))
    for i in range(len(conductivity)):
        s.k = conductivity[i]
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSalt[i] = np.max(s.sigmaEq)
    stressSalt *= 1e-6
    bar1 = (stressSalt[-1], stressSalt[1]-stressSalt[-1])
    bar2 = (stressSalt[1], stressSalt[0]-stressSalt[1])
    ax1.broken_barh([bar1, bar2], (11, 4),
                    facecolors=(fillSalt, fill),
                    edgecolor=edgeSalt,)
    for i in range(len(conductivity)):
        ax1.annotate('${0:g}$'.format(conductivity[i]), \
                     xy=(stressSalt[i], 16), \
                     xycoords='data', horizontalalignment=align[i],
                     fontsize=fs)

    s.k = 20
    # tube thermal expansion:
    alpha = np.array([1.4e-05, 1.8e-05, 2e-05])
    stressSalt = np.zeros(len(alpha))
    for i in range(len(alpha)):
        s.alpha = alpha[i]
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSalt[i] = np.max(s.sigmaEq)
    stressSalt *= 1e-6
    bar1 = (stressSalt[-1], stressSalt[1]-stressSalt[-1])
    bar2 = (stressSalt[1], stressSalt[0]-stressSalt[1])
    ax1.broken_barh([bar1, bar2], (21, 4),
                    facecolors=(fillSalt, fill),
                    edgecolor=edgeSalt,)
    for i in range(len(alpha)):
        ax1.annotate('${0:g}$'.format(alpha[i]*1e6), \
                     xy=(stressSalt[i], 26), \
                     xycoords='data', horizontalalignment=alignCen[i],
                     fontsize=fs)

    s.alpha = 1.85e-05
    # Young's modulus:
    youngs = np.array([150e9, 165e9, 200e9])
    stressSalt = np.zeros(len(youngs))
    for i in range(len(youngs)):
        s.E = youngs[i]
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSalt[i] = np.max(s.sigmaEq)
    stressSalt *= 1e-6
    bar1 = (stressSalt[-1], stressSalt[1]-stressSalt[-1])
    bar2 = (stressSalt[1], stressSalt[0]-stressSalt[1])
    ax1.broken_barh([bar1, bar2], (31, 4),
                    facecolors=(fillSalt, fill),
                    edgecolor=edgeSalt,)
    for i in range(len(youngs)):
        ax1.annotate('${0:g}$'.format(youngs[i]*1e-9), \
                     xy=(stressSalt[i], 36), \
                     xycoords='data', horizontalalignment=alignRev[i],
                     fontsize=fs)

    # sodium
    h_int = 40e3

    # DN 15 to 50 (1" is DN 25):
    labels = np.array([15, 25, 50])
    align = ['left', 'center', 'right']
    alignRev = ['right', 'center', 'left']
    d = np.array([21.34, 33.4, 60.33]) * 1e-3
    stressSod = np.zeros(len(d))
    for i in range(len(d)):
        b = d[i] / 2.
        a = b - 1.651e-3
        g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b)
        s = nts.Solver(grid=g, it=iterator, h_int=h_int, P_i=P_i)
        s.extBC = s.extTubeHalfCosFluxRadConv
        s.intBC = s.intTubeConv
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSod[i] = np.max(s.sigmaEq)
    stressSod *= 1e-6
    bar1 = (stressSod[-1], stressSod[1]-stressSod[-1])
    bar2 = (stressSod[1], stressSod[0]-stressSod[1])
    ax1.broken_barh([bar1, bar2], (1, 4),
                    facecolors=(fillSod, fill),
                    edgecolor=edgeSod,)
    for i in np.array([0,2]):#range(len(labels)):
        ax1.annotate('${0:g}$'.format(labels[i]), \
                     xy=(stressSod[i], 6), \
                     xycoords='data', horizontalalignment=align[i],
                     fontsize=fs)

    # tube conductivity from 10 to 30 [W/(m.K)]:
    b = 33.4 / 2e3
    a = b - 1.651e-3
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    s = nts.Solver(grid=g, it=iterator, h_int=h_int, P_i=P_i)
    s.extBC = s.extTubeHalfCosFluxRadConv
    s.intBC = s.intTubeConv
    conductivity = np.array([15, 20, 25])
    stressSod = np.zeros(len(conductivity))
    for i in range(len(conductivity)):
        s.k = conductivity[i]
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSod[i] = np.max(s.sigmaEq)
    stressSod *= 1e-6
    bar1 = (stressSod[-1], stressSod[1]-stressSod[-1])
    bar2 = (stressSod[1], stressSod[0]-stressSod[1])
    ax1.broken_barh([bar1, bar2], (11, 4),
                    facecolors=(fillSod, fill),
                    edgecolor=edgeSod,)
    for i in range(len(conductivity)):
        ax1.annotate('${0:g}$'.format(conductivity[i]), \
                     xy=(stressSod[i], 16), \
                     xycoords='data', horizontalalignment=align[i],
                     fontsize=fs)

    s.k = 20
    # tube thermal expansion:
    alpha = np.array([1.4e-05, 1.8e-05, 2e-05])
    stressSod = np.zeros(len(alpha))
    for i in range(len(alpha)):
        s.alpha = alpha[i]
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSod[i] = np.max(s.sigmaEq)
    stressSod *= 1e-6
    bar1 = (stressSod[-1], stressSod[1]-stressSod[-1])
    bar2 = (stressSod[1], stressSod[0]-stressSod[1])
    ax1.broken_barh([bar1, bar2], (21, 4),
                    facecolors=(fillSod, fill),
                    edgecolor=edgeSod,)
    for i in range(len(alpha)):
        ax1.annotate('${0:g}$'.format(alpha[i]*1e6), \
                     xy=(stressSod[i], 26), \
                     xycoords='data', horizontalalignment=alignCen[i],
                     fontsize=fs)

    s.alpha = 1.85e-05
    # Young's modulus:
    youngs = np.array([150e9, 165e9, 200e9])
    stressSod = np.zeros(len(youngs))
    for i in range(len(youngs)):
        s.E = youngs[i]
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSod[i] = np.max(s.sigmaEq)
    stressSod *= 1e-6
    bar1 = (stressSod[-1], stressSod[1]-stressSod[-1])
    bar2 = (stressSod[1], stressSod[0]-stressSod[1])
    ax1.broken_barh([bar1, bar2], (31, 4),
                    facecolors=(fillSod, fill),
                    edgecolor=edgeSod,)
    for i in range(len(youngs)):
        ax1.annotate('${0:g}$'.format(youngs[i]*1e-9), \
                     xy=(stressSod[i], 36), \
                     xycoords='data', horizontalalignment=alignRev[i],
                     fontsize=fs)

    #ax1.set_title('\\textbf{(b)} Molten salt')
    ax1.set_ylim(0, 40)
    ax1.set_yticks([5, 15, 25, 35])
    ax1.set_yticklabels(
        ['$\\mathrm{DN}$\n\small{(-)}',
         '$\\lambda$\n\small{(\si{\watt\per\meter\per\kelvin})}',
         '$\\alpha$\n\small{(\SI{e-6}{\per\kelvin})}',
         '$E$\n\small{(GPa)}'], fontsize='large'
    )

    ax1.set_xlim(150, 450)
    ax1.set_xlabel(r'\textsc{max. equiv. stress}, '+\
                   '$\max\sigma_\mathrm{Eq}$ (MPa)')
    fig1.tight_layout()
    #plt.show()
    fig1.savefig('S31609_sensitivityTubeProperties.pdf',
                 transparent=True)
    plt.close(fig1)

    headerprint(' Fluid flow parameter variation ')

    fig2 = plt.figure(figsize=(5, 2.5))
    ax2 = SubplotHost(fig2, 1, 1, 1)
    ax2a = ax2.twin(trans)
    ax2a.set_viewlim_mode("transform")
    ax2a.axis["top"].set_label(r'\textsc{max. equiv. stress}, '+\
                               '$\max\sigma_\mathrm{Eq}$ (ksi)')
    ax2a.axis["top"].label.set_visible(True)
    ax2a.axis["right"].label.set_visible(False)
    ax2a.axis["right"].major_ticklabels.set_visible(False)
    ax2a.axis["right"].major_ticks.set_visible(False)
    ax2 = fig2.add_subplot(ax2)

    # salt
    h_int = 10e3

    b = 33.4 / 2e3
    a = b - 1.651e-3
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    s = nts.Solver(grid=g, it=iterator)
    s.P_i = 0e5          # internal pipe pressure
    s.extBC = s.extTubeHalfCosFluxRadConv
    s.intBC = s.intTubeConv

    # convection coefficient:
    labels = np.array([8, 10, 12])
    stressSalt = np.zeros(len(labels))
    for i in range(len(labels)):
        s.h_int = labels[i]*1e3
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSalt[i] = np.max(s.sigmaEq)
    stressSalt *= 1e-6
    bar1 = (stressSalt[-1], stressSalt[1]-stressSalt[-1])
    bar2 = (stressSalt[1], stressSalt[0]-stressSalt[1])
    ax2.broken_barh([bar1, bar2], (1, 4),
                    facecolors=(fillSalt, fill),
                    edgecolor=edgeSalt,)
    for i in range(len(labels)):
        ax2.annotate('${0:g}$'.format(labels[i]), \
                     xy=(stressSalt[i], 6), \
                     xycoords='data', horizontalalignment=align[i],
                     fontsize=fs)

    s.h_int = 10e3
    # fouling factor:
    labels = np.array([0, 2.5, 5])
    stressSalt = np.zeros(len(labels))
    for i in range(len(labels)):
        s.R_f = labels[i]*1e-5
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSalt[i] = np.max(s.sigmaEq)
    stressSalt *= 1e-6
    bar1 = (stressSalt[-1], stressSalt[1]-stressSalt[-1])
    bar2 = (stressSalt[1], stressSalt[0]-stressSalt[1])
    ax2.broken_barh([bar1, bar2], (11, 4),
                    facecolors=(fillSalt, fill),
                    edgecolor=edgeSalt,)
    for i in range(len(labels)):
        ax2.annotate('${0:g}$'.format(labels[i]), \
                     xy=(stressSalt[i], 16), \
                     xycoords='data', horizontalalignment=alignRev[i],
                     fontsize=fs)

    s.R_f = 0.
    # internal pressure:
    specAlign = ['left', 'left', 'right']
    labels = np.array([0, 5, 10])
    stressSalt = np.zeros(len(labels))
    for i in range(len(labels)):
        s.P_i = labels[i]*1e6
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSalt[i] = np.max(s.sigmaEq)
    stressSalt *= 1e-6
    bar1 = (stressSalt[-1], stressSalt[1]-stressSalt[-1])
    bar2 = (stressSalt[1], stressSalt[0]-stressSalt[1])
    ax2.broken_barh([bar1, bar2], (21, 4),
                    facecolors=(fillSalt, fill),
                    edgecolor=edgeSalt,)
    for i in range(len(labels)):
        ax2.annotate('${0:g}$'.format(labels[i]), \
                     xy=(stressSalt[i], 26), \
                     xycoords='data', horizontalalignment=specAlign[i],
                     fontsize=fs)

    s.P_i = 0e5

    # sodium

    # convection coefficient:
    labels = np.array([20, 40, 48])
    stressSod = np.zeros(len(labels))
    for i in range(len(labels)):
        s.h_int = labels[i]*1e3
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSod[i] = np.max(s.sigmaEq)
    stressSod *= 1e-6
    bar1 = (stressSod[-1], stressSod[1]-stressSod[-1])
    bar2 = (stressSod[1], stressSod[0]-stressSod[1])
    ax2.broken_barh([bar1, bar2], (1, 4),
                    facecolors=(fillSod, fill),
                    edgecolor=edgeSod,)
    for i in range(len(labels)):
        ax2.annotate('${0:g}$'.format(labels[i]), \
                     xy=(stressSod[i], 6), \
                     xycoords='data', horizontalalignment=align[i],
                     fontsize=fs)

    s.h_int = 40e3
    # fouling factor:
    labels = np.array([0, 2.5, 5])
    stressSod = np.zeros(len(labels))
    for i in range(len(labels)):
        s.R_f = labels[i]*1e-5
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSod[i] = np.max(s.sigmaEq)
    stressSod *= 1e-6
    bar1 = (stressSod[-1], stressSod[1]-stressSod[-1])
    bar2 = (stressSod[1], stressSod[0]-stressSod[1])
    ax2.broken_barh([bar1, bar2], (11, 4),
                    facecolors=(fillSod, fill),
                    edgecolor=edgeSod,)
    for i in range(len(labels)):
        ax2.annotate('${0:g}$'.format(labels[i]), \
                     xy=(stressSod[i], 16), \
                     xycoords='data', horizontalalignment=alignRev[i],
                     fontsize=fs)

    s.R_f = 0.
    # internal pressure:
    labels = np.array([0, 5, 10])
    stressSod = np.zeros(len(labels))
    for i in range(len(labels)):
        s.P_i = labels[i]*1e6
        ret = s.solve(n_iter=1000)
        s.postProcessing()
        stressSod[i] = np.max(s.sigmaEq)
    stressSod *= 1e-6
    bar1 = (stressSod[-1], stressSod[1]-stressSod[-1])
    bar2 = (stressSod[1], stressSod[0]-stressSod[1])
    ax2.broken_barh([bar1, bar2], (21, 4),
                    facecolors=(fillSod, fill),
                    edgecolor=edgeSod,)
    for i in range(len(labels)):
        ax2.annotate('${0:g}$'.format(labels[i]), \
                     xy=(stressSod[i], 26), \
                     xycoords='data', horizontalalignment=specAlign[i],
                     fontsize=fs)

    #ax2.set_title('\\textbf{(b)} Molten salt')
    ax2.set_ylim(0, 30)
    ax2.set_yticks([5, 15, 25])
    ax2.set_yticklabels([
        '$h_\\mathrm{i}$\n\small{(\si{\kilo\watt\per\meter\squared\per\kelvin})}',
        '$R_\\mathrm{f}$\n\small{(\SI{e-5}{\meter\squared\kelvin\per\watt})}',
        '$p_\\mathrm{i}$\n\small{(MPa)}'], fontsize='large'
    )

    ax2.set_xlim(190, 460)
    ax2.set_xlabel(r'\textsc{max. equiv. stress}, '+\
                   '$\max\sigma_\mathrm{Eq}$ [MPa]')
    fig2.tight_layout()
    #plt.show()
    fig2.savefig('S31609_sensitivityFlowProperties.pdf', transparent=True)
    plt.close(fig2)

    headerprint(' Exploration of flux profile ')
    iterator='inline'
    nr=6; nt=61

    fig = plt.figure(figsize=(4.5, 3))
    ax = fig.add_subplot(111)

    # NPS Sch. 5S 1" UNS S31609 tube:
    a = 30.098/2e3     # inside tube radius [mm->m]
    b = 33.4/2e3       # outside tube radius [mm->m]
    k = 21; alpha=20e-6; E = 165e9; nu = 0.31

    # Create instance of LaplaceSolver:
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    """ SS316: """
    s = nts.Solver(g, CG=8.5e5, k=k, T_int=723.15, R_f=0,#8.808e-5,
                   A=0.968, epsilon=0.87, T_ext=293.15, h_ext=h_ext,
                   P_i=0e5, alpha=alpha, E=E, nu=nu, n=1)

    s.intBC = s.intTubeConv
    for i, fluid, htc in [*zip([0,1],['Nitrate Salt','Liquid Sodium'],[10e3,40e3])]:
        headerprint(fluid, '_')
        s.h_int = htc
        valprint('h_int', s.h_int, 'W/(m^2.K)')

        headerprint('Double-sided flux case:', ' ')
        s.extBC = s.extTubeFullCosFluxRadConv
        t = time.perf_counter(); ret = s.solve(n_iter=1000)
        s.heatFluxBalance()
        if i == 0:
            ax.plot(s.g.theta[:,0], s.phi_inc[1:-1]*1e-3, 'x-',
                    label=r'$|\cos\theta|$', markevery=3)
        s.postProcessing()
        valprint('Time', time.perf_counter() - t, 'sec')
        valprint('max(sigmaEq)', np.max(s.sigmaEq)*1e-6, 'MPa')

        plotStressAnnotate(
            g.theta, g.r, s.sigmaEq,
            s.sigmaEq.min(), s.sigmaEq.max(), 'right',
            'S31609_htc{0}_doubleFlux_sigmaEq.pdf'.format(int(htc*1e-3)
            )
        )
        j = 0 # tube crown
        plotComponentStress(
            g.r, s.sigmaR, s.sigmaTheta, s.sigmaZ, s.sigmaEq,
            'S31609_htc{0}_doubleFlux_theta{1}.pdf'.format(
                int(htc*1e-3), int(np.degrees(g.theta[j,0]))
            ),
            j, 'best'
        )
        j = 30 # 90 degrees (tube side)
        plotComponentStress(
            g.r, s.sigmaR, s.sigmaTheta, s.sigmaZ, s.sigmaEq,
            'S31609_htc{0}_doubleFlux_theta{1}.pdf'.format(
                int(htc*1e-3), int(np.degrees(g.theta[j,0]))
            ),
            j, 'best'
        )

        headerprint('Base case:', ' ')
        s.extBC = s.extTubeHalfCosFluxRadConv
        t = time.perf_counter(); ret = s.solve(n_iter=1000)
        s.heatFluxBalance()
        if i == 0:
            ax.plot(s.g.theta[:,0], s.phi_inc[1:-1]*1e-3, 'o-',
                    label=r'$f^+(\cos\theta)$', markevery=3)
        #label='$q_\mathrm{inc}\'\'\cos{\\theta}$', markevery=3)
        s.postProcessing()
        valprint('Time', time.perf_counter() - t, 'sec')
        valprint('max(sigmaEq)', np.max(s.sigmaEq)*1e-6, 'MPa')

        headerprint('Adiabatic case:', ' ')
        s.extBC = s.extTubeHalfCosFluxRadConvAdiabaticBack
        t = time.perf_counter(); ret = s.solve(n_iter=1000)
        s.heatFluxBalance()
        s.postProcessing()
        valprint('Time', time.perf_counter() - t, 'sec')
        valprint('max(sigmaEq)', np.max(s.sigmaEq)*1e-6, 'MPa')

        headerprint('Foster Wheeler (Abendgoa) case:', ' ')
        s.extBC = s.extTubeFWFluxRadConv
        t = time.perf_counter(); ret = s.solve(n_iter=1000)
        s.heatFluxBalance()
        if i == 0:
            ax.plot(s.g.theta[:,0], s.phi_inc[1:-1]*1e-3, '>-',
                    label=r'Abengoa', markevery=3)
        s.postProcessing()
        valprint('Time', time.perf_counter() - t, 'sec')
        valprint('max(sigmaEq)', np.max(s.sigmaEq)*1e-6, 'MPa')

        headerprint('Step flux case:', ' ')
        sumQ_inc = -np.sum(s.q_inc)
        s.phi_inc = s.g.halfTube * sumQ_inc / \
                    np.sum(s.g.halfTube[1:-1] * s.g.sfRmax)
        s.extBC = s.extTubeFluxProfileRadConv
        t = time.perf_counter(); ret = s.solve(n_iter=1000)
        s.heatFluxBalance()
        if i == 0:
            ax.step(s.g.theta[:,0], s.phi_inc[1:-1]*1e-3, 'v-',
                    label=r'Step', markevery=3)
        s.postProcessing()
        valprint('Time', time.perf_counter() - t, 'sec')
        valprint('max(sigmaEq)', np.max(s.sigmaEq)*1e-6, 'MPa')

        headerprint('Fading flux case:', ' ')
        fade = 2 * s.g.fullTube * sumQ_inc / np.sum(s.g.sfRmax)
        m = -fade / np.pi
        s.phi_inc = m * s.g.meshTheta[:,0] + fade
        s.extBC = s.extTubeFluxProfileRadConv
        t = time.perf_counter(); ret = s.solve(n_iter=1000)
        s.heatFluxBalance()
        if i == 0:
            ax.plot(s.g.theta[:,0], s.phi_inc[1:-1]*1e-3, 's-',
                    label=r'Fade', markevery=3)
        s.postProcessing()
        valprint('Time', time.perf_counter() - t, 'sec')
        valprint('max(sigmaEq)', np.max(s.sigmaEq)*1e-6, 'MPa')

        headerprint('Peak step flux case:', ' ')
        s.phi_inc = s.g.halfTube * s.CG
        s.extBC = s.extTubeFluxProfileRadConv
        t = time.perf_counter(); ret = s.solve(n_iter=1000)
        s.heatFluxBalance()
        if i == 0:
            ax.step(s.g.theta[:,0], s.phi_inc[1:-1]*1e-3, '^-',
                    label=r'Peak step', markevery=3)
        s.postProcessing()
        valprint('Time', time.perf_counter() - t, 'sec')
        valprint('max(sigmaEq)', np.max(s.sigmaEq)*1e-6, 'MPa')
    unit = np.pi/4
    x_tick = np.arange(0, np.pi+unit, unit)
    # x_label = [r'$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', \
    #            r'$\frac{3\pi}{4}$', r'$\pi$']
    x_label = [r'$0$', r'$45^\circ$', r'$90^\circ$', \
               r'$135^\circ$', r'$180^\circ$']
    ax.set_xticks(x_tick)
    ax.set_xticklabels(x_label)
    ax.set_xlabel(r'\textsc{cylindrical coordinate}, $\theta$')
    ax.set_xlim(0, x_tick[-1])
    ax.set_ylabel(r'\textsc{heat flux density}, $\vec{\phi_\mathrm{q}}$ '+\
                  '(\si{\kilo\watt\per\meter\squared})')
    #ax.set_ylim(410,600)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig('S31609_sensitivityFluxProfiles.pdf', transparent=True)
    plt.close(fig)

    OD = 19.05 # mm
    WT = 1.2446 # mm
    b = OD/2e3         # outside tube radius [mm->m]
    a = (b-WT*1e-3)     # inside tube radius [mm->m]
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b) # nr, nt -> resolution
    for mat in ['316H', 'P91']:
        headerprint('Reproducing Kistler (1987) for {}'.format(mat), ' ')
        if mat == '316H':
            k = 21; alpha=20e-6; E = 165e9; nu = 0.31
        if mat == 'P91':
            k = 27.5; alpha=14e-6; E = 183e9; nu = 0.3
        s = nts.Solver(g, debug=False, CG=0.85e6, k=k, T_int=723.15, R_f=0,
                       A=0.968, epsilon=0.87, T_ext=293.15, h_ext=h_ext,
                       P_i=0e5, alpha=alpha, E=E, nu=nu, n=1,
                       bend=False)
        s.extBC = s.extTubeHalfCosFluxRadConv
        s.intBC = s.intTubeConv
        #s.debug = False
        sodium.debug = False; salt.debug = False
        fv = np.genfromtxt(os.path.join('mats', mat), delimiter=';')
        fv[:,0] += 273.15 # degC to K
        fv[:,2] *= 3e6 # apply 3f criteria to Sm and convert MPa->Pa
        T_int = np.linspace(290, 565, 12)+273.15
        TSod_met = np.zeros(len(T_int))
        fluxSod = np.zeros(len(T_int))
        TSalt_met = np.zeros(len(T_int))
        fluxSalt = np.zeros(len(T_int))
        t = time.perf_counter()
        for i in range(len(T_int)):
            s.T_int = T_int[i]
            sodium.update(T_int[i])
            s.h_int, dP = coolant.HTC(False, sodium, a, b, 20, 'Chen', 'velocity', 4)
            fluxSod[i] = opt.newton(
                findFlux, 1e5,
                args=(s, fv, 2, 'outside'),
                maxiter=1000, tol=1e-2
            )
            TSod_met[i] = np.max(s.T)
            salt.update(T_int[i])
            s.h_int, dP = coolant.HTC(False, salt, a, b, 20, 'Dittus', 'velocity', 4)
            fluxSalt[i] = opt.newton(
                findFlux, 1e5,
                args=(s, fv, 2, 'outside'),
                maxiter=1000, tol=1e-2
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
            r'\textsc{incident flux}, $\vec{\phi_\mathrm{q}}$ '+\
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
