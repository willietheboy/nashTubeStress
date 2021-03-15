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

def plotNACA(r, sigma, fea, i, filename, loc, ylabel):
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
    axa.axis["right"].set_label(ylabel+' (ksi)')
    axa.axis["right"].label.set_visible(True)
    ax = fig.add_subplot(ax)
    ax.plot(r[0,:]*1e3, sigma[0,:]*1e-6, '-',
            color='C0',label=r'$\theta=0^\circ$')
    ax.plot((a+fea[0][:,0])*1e3, fea[0][:,i]*1e-6, 'o',
            color='C0', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[20,:]*1e-6, '-',
            color='C1', label=r'$\theta=60^\circ$')
    ax.plot((a+fea[1][:,0])*1e3, fea[1][:,i]*1e-6, '^',
            color='C1', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[40,:]*1e-6, '-',
            color='C2', label=r'$\theta=120^\circ$')
    ax.plot((a+fea[2][:,0])*1e3, fea[2][:,i]*1e-6, 'v',
            color='C2', markevery=1)
    ax.plot(r[0,:]*1e3, sigma[60,:]*1e-6, '-',
            color='C3', label=r'$\theta=180^\circ$')
    ax.plot((a+fea[3][:,0])*1e3, fea[3][:,i]*1e-6, 's',
            color='C3', markevery=1)
    ax.set_xlabel(r'\textsc{radius}, $r$ (mm)')
    ax.set_xlim((a*1e3)-10,(b*1e3)+10)
    ax.set_ylabel(ylabel+' (MPa)')
    #ax.set_ylim(-400, 400)
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
    Holms, A.G., 1952. A biharmonic relaxation method for calculating thermal
    stress in cooled irregular cylinders. Technical Report NACA-TR-1059.
    Lewis Flight Propulsion Lab. Cleveland.
    URL: https://ntrs.nasa.gov/search.jsp?R=19930092105
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
    g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b)
    s = nts.Solver(g, debug=True, alpha=alpha,
               E=E, nu=0.3, n=1)
    ## Apply analytical temperature:
    s.T = (((c1-c0) * b) / (b**2 - a**2)) * \
          ((g.r**2 - a**2) / g.r) * np.cos(g.theta) + \
          (c2-c0) * (1 - (np.log(b / g.r) / np.log(b / a)))
    s.postProcessing()

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

    headerprint('Table II, p89: radius vs. tangential (hoop) stress', ' ')
    radius_inch = np.linspace(4,12,9)
    radius = Q_(radius_inch, 'inch').to('m')
    sigmaTheta = Q_(
        np.interp(
            radius.magnitude, s.g.r[0,:],
            s.sigmaTheta[0,:]
        ), 'Pa'
    ).to('psi')
    for r, sig_t in [*zip(radius_inch, sigmaTheta)]:
        valprint('{} (in.)'.format(r), sig_t.magnitude, 'psi')

    """ Comparison with code_aster 13.6 MECA_STATIQUE [U4.51.01]: """
    fea = [None]*4
    for i, theta in enumerate([0, 60, 120, 180]):
        fn = 'NACA-TR-1059_THETA{}'.format(theta) + \
             '_SIEF-CYL.dat'
        fea[i] = np.genfromtxt(os.path.join('aster', fn), skip_header=5)
    plotNACA(g.r, s.sigmaTheta, fea, 4, 'NACA-TR-1059_sigmaTheta.pdf', \
             'best', r'\textsc{hoop stress}, $\sigma_\theta$')
    plotNACA(g.r, s.sigmaR, fea, 5, 'NACA-TR-1059_sigmaR.pdf', \
             'best', r'\textsc{radial stress}, $\sigma_r$')
    plotNACA(g.r, s.sigmaZ, fea, 6, 'NACA-TR-1059_sigmaZ.pdf', \
             'best', r'\textsc{axial stress}, $\sigma_z$')
    plotNACA(g.r, s.sigmaRTheta, fea, 7, 'NACA-TR-1059_sigmaRTheta.pdf', \
             'best', r'\textsc{in-plane shear stress}, $\tau_{r\theta}$')
    fea = [None]*4
    for i, theta in enumerate([0, 60, 120, 180]):
        fn = 'NACA-TR-1059_THETA{}'.format(theta) + \
             '_SIEQ.dat'
        fea[i] = np.genfromtxt(os.path.join('aster', fn), skip_header=5)
    plotNACA(g.r, s.sigmaEq, fea, 1, 'NACA-TR-1059_sigmaEq.pdf', \
             'best', r'\textsc{equivalent stress}, $\sigma_\mathrm{eq}$')
