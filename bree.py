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
    ax.fill_between(r[i,:]*1e3, sigmaTheta[i,:]*1e-6, color='C1',
                    alpha=0.3)
    ax.fill_between(r[i,:]*1e3, sigmaZ[i,:]*1e-6, color='C2',
                    alpha=0.3)
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

##################################### MAIN #####################################

if __name__ == "__main__":
    """
    Bree, J. Elastic-plastic behaviour of thin tubes subjected to internal pressure and intermittent high-heat fluxes with application to Fast-Nuclear-Reactor fuel elements J. Strain Anal., 1967, 2, 226-238
    """
    headerprint(' BREE ')

    nr=10; nt=32
    a = 100e-3      # inside tube radius [mm->m]
    b = 101e-3      # outside tube radius [mm->m]

    R = (a + b)/2.
    wt = b-a
    P = 1e6 # Pa
    sig_p = P * R / wt
    valprint('sig_p', sig_p*1e-6, 'MPa')

    E = 100e9
    valprint('E', E*1e-9, 'GPa')
    alpha = 1.5e-5
    valprint('alpha', alpha*1e6, 'x1e6 K^-1')
    nu = 0.31

    dT = 50
    valprint('dT', dT, 'K')
    sig_t = E * alpha * dT / (2 * (1 - nu))
    valprint('max. sig_t', sig_t*1e-6, 'MPa')

    ## True: generalised plane strain (default); False: simple plane strain:
    GPS = True

    ## Create nashTubeStress instance of Grid and Solver:
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    s = nts.Solver(g, debug=False, alpha=alpha,
                   E=E, nu=nu, n=1, GPS=GPS,
                   P_i=P)
    s.postProcessing()
    plotComponentStress(
        g.r, s.sigmaR, s.sigmaTheta, s.sigmaZ,
        s.sigmaEq, 'Bree_comp-pres.pdf', 0, 'best')

    ## Apply analytical temperature (without pressure):
    #s.T = dT * ((np.log(g.r / a) / np.log(b / a)))
    s.P_i=0
    s.T = dT * ((g.r - a) / (b - a))
    s.postProcessing()
    plotComponentStress(
        g.r, s.sigmaR, s.sigmaTheta, s.sigmaZ,
        s.sigmaEq, 'Bree_comp-therm.pdf', 0, 'best')

    # plotStress(g.theta, g.r, s.sigmaR,
    #            s.sigmaR.min(), s.sigmaR.max(),
    #            'Bree_sigmaR.pdf')
    # plotStress(g.theta, g.r, s.sigmaTheta,
    #            s.sigmaTheta.min(), s.sigmaTheta.max(),
    #            'Bree_sigmaTheta.pdf')
    # plotStress(g.theta, g.r, s.sigmaRTheta,
    #            s.sigmaRTheta.min(), s.sigmaRTheta.max(),
    #            'Bree_sigmaRTheta.pdf')
    # plotStress(g.theta, g.r, s.sigmaZ,
    #            s.sigmaZ.min(), s.sigmaZ.max(),
    #            'Bree_sigmaZ.pdf')
    # plotStress(g.theta, g.r, s.sigmaEq,
    #            s.sigmaEq.min(), s.sigmaEq.max(),
    #            'Bree_sigmaEq.pdf')
