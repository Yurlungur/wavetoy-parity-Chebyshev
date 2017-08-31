#!/usr/bin/env python2

"""parity_orthopoly.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-08-31 16:43:38 (jmiller)>

A module for parity-restricted orthogonal polynomials for
pseudospectral methods in Python
"""

# ======================================================================
# imports
# ======================================================================
import numpy as np
from numpy import polynomial
from numpy import linalg
from scipy import integrate
# ======================================================================

# ======================================================================
# Global constants
# ======================================================================
LOCAL_XMIN = -1. # Maximum and min values of reference cell
LOCAL_XMAX = 1.
LOCAL_ORIGIN = 0.
LOCAL_WIDTH = float(LOCAL_XMAX-LOCAL_XMIN)
poly = polynomial.chebyshev.Chebyshev  # A class for orthogonal polynomials
weight_func = polynomial.chebyshev.chebweight
integrator = integrate.quad
EVEN='even'
ODD='odd'
# ======================================================================

# ======================================================================
# Nodal and Modal Details
# ======================================================================
def get_full_quadrature_points(order):
    """
    Returns the quadrature points for Gauss-Lobatto quadrature
    as a function of the order of the polynomial we want to
    represent.
    See: https://en.wikipedia.org/wiki/Gaussian_quadrature

    This version is for quadrature on the interval [-1,1].
    Do not use with parity scheme.
    """
    return np.sort(np.concatenate((np.array([-1,1]),
                                   poly.basis(order).deriv().roots())))

def get_parity_quadrature_points(order):
    """Returns the parity-restricted quadrature points for Gauss-Lobatto
    quadrature as a function of the order of the polynomial we want to
    represent.
    """
    p = 2*order+1
    all_points = get_full_quadrature_points(p)
    mpoints = all_points[len(all_points)/2:]
    return mpoints

def get_integration_weights(order,nodes=None):
    """
    Returns the integration weights for Gauss-Lobatto quadrature
    as a function of the order of the polynomial we want to
    represent.
    See: https://en.wikipedia.org/wiki/Gaussian_quadrature
    See: arXive:gr-qc/0609020v1
    """
    p = 2*order+1
    weights = np.empty((p+1))
    weights[1:-1] = np.pi/p
    weights[0] = np.pi/(2*p)
    weights[-1] = weights[0]
    weights = weights[len(weights)/2:]
    return weights
# ======================================================================


# ======================================================================
# A convenience class that generates everything and can be called
# ======================================================================
class ParityPseudoSpectralDiscretization1D:
    """Given an order, and a domain [xmin,xmax]
    defines internally all structures and methods the user needs
    to calculate spectral derivatives in 1D
    """
    def __init__(self,order,rmax):
        "Constructor. Needs the order of the method and the domain [0,rmax]."
        self.order = order
        self.rmin = 0.
        self.rmax = rmax
        self.quads = get_parity_quadrature_points(self.order)
        self.weights = get_integration_weights(self.order,self.quads)
        self.van_even = self._calculate_vandermonde(EVEN)
        self.van_odd = self._calculate_vandermonde(ODD)

    def _calculate_vandermonde(self,parity):
        assert hasattr(self,"order")
        assert hasattr(self,"quads")
        offset = 0 if parity == EVEN else 1
        s2c = np.zeros((order+1,order+1),dtype=float)
        for i in range(order+1):
            for j in range(order+1):
                s2c[i,j] = poly.basis(2*j+offset)(self.quads[i])
        c2s = linalg.inv(s2c)
        out = Vandermonde(s2c,c2s)
        return out

    def 

class Vandermonde:
    """A container class containing Vandermonde matrices.
    """
    def __init__(self,s2c,c2s):
        self.s2c = s2c
        self.c2s = c2s
# ======================================================================


# Warning not to run this program on the command line
if __name__ == "__main__":
    raise ImportError("Warning. This is a library. It contains no main function.")
