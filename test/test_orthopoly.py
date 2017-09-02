#!/usr/bin/env python2

"""test_orthopoly.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-09-02 00:06:51 (jmiller)>

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

def get_l2_error(f_num,f_true):
    "Get l2-norm of the error between f_num and f_true"
    error_func = lambda x: f_num(x) - f_true(x)
    error = parity.reference_2norm(error_func)
    return error

def get_sxy(res,f):
    s = parity.ParityPseudoSpectralDiscretization1D(res)
    x = s.quads
    y = f(x)
    return s,x,y

def convergence_d_even(resolutions):
    """Tests convergence of the derivative operator on even functions.
    Uses resolutions.
    """
    k = 3
    f = lambda x: np.cos(k*2*np.pi*x)
    df = lambda x: -k*2*np.pi*np.sin(k*2*np.pi*x)
    errors = []
    for res in resolutions:
        s,x,y = get_sxy(res,f)
        dy = np.dot(s.dNodal_even,y)
        dyf = parity.get_continuous_object(dy,s.van_odd.c2s,parity.ODD)
        error = get_l2_error(dyf,df)
        errors.append(error)
    return errors

def convergence_inv_odd(resolutions):
    "Tests convergence of the 1/r operator on even functions."
    k = 3
    f = lambda x: np.sin(k*2*np.pi*x)
    fox = lambda x: np.sin(k*2*np.pi*x)/x
    errors = []
    for res in resolutions:
        s,x,y = get_sxy(res,f)
        yox = np.dot(s.div_x_nodal,y)
        yoxf = parity.get_continuous_object(yox,s.van_even.c2s,parity.EVEN)
        error = get_l2_error(yoxf,fox)
        errors.append(error)
    return errors

def convergence_d_odd(resolutions):
    "Tests convergence of the derivative operator on even functions."
    k = 3
    f = lambda x: np.sin(k*2*np.pi*x)
    df = lambda x: k*2*np.pi*np.cos(k*2*np.pi*x)
    errors = []
    for res in resolutions:
        s,x,y = get_sxy(res,f)
        dy = np.dot(s.dNodal_odd,y)
        dyf = parity.get_continuous_object(dy,s.van_even.c2s,parity.EVEN)
        error = get_l2_error(dyf,df)
        errors.append(error)
    return errors

def convergence_d2_even(resolutions):
    """Tests convergence of the second derivative operator on even
    functions.
    """
    k = 3
    f = lambda x: np.cos(k*2*np.pi*x)
    df = lambda x: -((2*np.pi*k)**2)*np.cos(2*k*np.pi*x)
    errors = []
    for res in resolutions:
        s,x,y = get_sxy(res,f)
        dy = np.dot(s.d2Nodal_even,y)
        dyf = parity.get_continuous_object(dy,s.van_even.c2s,parity.EVEN)
        error = get_l2_error(dyf,df)
        errors.append(error)
    return errors

def convergence_poisson_even(resolutions):
    "Tests convergence of the Poisson operator on even functions."
    k = 3
    f = lambda x: np.cos(k*2*np.pi*x)
    df = lambda x: -(12*k*(np.pi**2)*np.cos(k*2*np.pi*x)
                     +4*k*np.pi*np.sin(2*k*np.pi*x)/x)
    errors = []
    for res in resolutions:
        s,x,y = get_sxy(res,f)
        dy = np.dot(s.poisson_nodal,y)
        dyf = parity.get_continuous_object(dy,s.van_even.c2s,parity.EVEN)
        error = get_l2_error(dyf,df)
        errors.append(error)
    return errors

if __name__ == "__main__":
    resolutions = [4,7,10,13,16,19,22,25]
    print("Using resolutions: {}".format(resolutions))

    print("Testing differentiation of an even function")
    errors = convergence_d_even(resolutions)
    plt.semilogy(resolutions,errors,'o-',
                 label=r'$\partial_r \cos(6\pi r)$')
    
    print("Testing differentiation of an odd function")
    errors = convergence_d_odd(resolutions)
    plt.semilogy(resolutions,errors,'o-',
                 label=r'$\partial_r \sin(6\pi r)$')

    print("Testing dividing an odd function by the radial coordinate")
    errors = convergence_inv_odd(resolutions)
    plt.semilogy(resolutions,errors,'o-',
                 label=r'$\sin(6\pi r)/x$')

    print("Testing second derivative of an even function")
    errors = convergence_d2_even(resolutions)
    plt.semilogy(resolutions,errors,'o-',
                 label=r'$\partial_r^2 \cos(6\pi r)$')

    print("Testing Poisson operator on an even function")
    mlabel=r'$\left(\partial^2_r +\frac{2}{r}\partial_r\right)\cos(6\pi r)$'
    errors = convergence_poisson_even(resolutions)
    plt.semilogy(resolutions,errors,'o-',
                 label=mlabel)

    plt.legend(loc='lower left')
    plt.xlabel('Number of (parity-restricted) modes')
    plt.ylabel('Error')
    plt.savefig('figs/test_operator_convergence.png',
                bbox_inches='tight')

    print("Done")
