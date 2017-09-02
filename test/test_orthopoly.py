#!/usr/bin/env python2

"""test_orthopoly.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-09-01 20:21:28 (jmiller)>

Test suite for parity_orthopoly.py
"""

# ======================================================================
# imports
# ======================================================================
from __future__ import print_function
import numpy as np
from numpy import polynomial
from numpy import linalg
from scipy import integrate
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams['font.size'] = 16

import sys
sys.path.append('..')
import parity_orthopoly as parity

# ======================================================================

def convergence_d_even(resolutions):
    """Tests convergence of the derivative operator on even functions.
    Uses resolutions.
    """
    k = 3
    f = lambda x: np.cos(k*2*np.pi*x)
    df = lambda x: -k*2*np.pi*np.sin(k*2*np.pi*x)
    errors = []
    for res in resolutions:
        s = parity.ParityPseudoSpectralDiscretization1D(res)
        x = s.quads
        y = f(x)
        dy = np.dot(s.dNodal_even,y)
        dyf = parity.get_continuous_object(dy,s.van_odd.c2s,parity.ODD)
        error_func = lambda x: dyf(x) - df(x)
        error = parity.reference_2norm(error_func)
        errors.append(error)
    return errors

def convergence_inv_odd(resolutions):
    """Tests convergence of the 1/r operator on even functions.
    Uses resolutions.
    """
    k = 3
    f = lambda x: np.sin(k*2*np.pi*x)
    fox = lambda x: np.sin(k*2*np.pi*x)/x
    errors = []
    for res in resolutions:
        s = parity.ParityPseudoSpectralDiscretization1D(res)
        x = s.quads
        y = f(x)
        yox = np.dot(s.div_x_nodal,y)
        yoxf = parity.get_continuous_object(yox,s.van_even.c2s,parity.EVEN)
        error_func = lambda x: yoxf(x) - fox(x)
        error = parity.reference_2norm(error_func)
        errors.append(error)
    return errors

def convergence_d_odd(resolutions):
    """Tests convergence of the derivative operator on even functions.
    Uses resolutions.
    """
    k = 3
    f = lambda x: np.sin(k*2*np.pi*x)
    df = lambda x: k*2*np.pi*np.cos(k*2*np.pi*x)
    errors = []
    for res in resolutions:
        s = parity.ParityPseudoSpectralDiscretization1D(res)
        x = s.quads
        y = f(x)
        dy = np.dot(s.dNodal_odd,y)
        dyf = parity.get_continuous_object(dy,s.van_even.c2s,parity.EVEN)
        error_func = lambda x: dyf(x) - df(x)
        error = parity.reference_2norm(error_func)
        errors.append(error)
    return errors

if __name__ == "__main__":
    resolutions = [4,7,10,13,16,19,22,25]
    print("Using resolutions: {}".format(resolutions))

    print("Testing differentiation of an even function")
    errors = convergence_d_even(resolutions)
    plt.semilogy(resolutions,errors,'bo-')
    plt.xlabel('Number of (parity-restricted) modes')
    plt.ylabel('Error in '+r'$\partial_r \cos(6\pi r)$')
    plt.savefig('figs/test_dy_even.png',bbox_inches='tight')
    #plt.savefig('figs/test_dy_even.pdf',bbox_inches='tight')
    
    print("Testing differentiation of an odd function")
    errors = convergence_d_odd(resolutions)
    plt.semilogy(resolutions,errors,'bo-')
    plt.xlabel('Number of (parity-restricted) modes')
    plt.ylabel('Error in '+r'$\partial_r \sin(6\pi r)$')
    plt.savefig('figs/test_dy_odd.png',bbox_inches='tight')
    #plt.savefig('figs/test_dy_odd.pdf',bbox_inches='tight')

    print("Testing dividing an odd function by the radial coordinate")
    errors = convergence_inv_odd(resolutions)
    plt.semilogy(resolutions,errors,'bo-')
    plt.xlabel('Number of (parity-restricted) modes')
    plt.ylabel('Error in '+r'$\sin(6\pi r)/x$')
    plt.savefig('figs/test_yox_odd.png',bbox_inches='tight')
    #plt.savefig('figs/test_yox_odd.pdf',bbox_inches='tight')

