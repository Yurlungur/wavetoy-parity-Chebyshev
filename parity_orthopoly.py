#!/usr/bin/env python2

"""parity_orthopoly.py
Author: Jonah Miller (jonah.maxwell.miller@gmail.com)
Time-stamp: <2017-09-01 20:40:03 (jmiller)>

A module for parity-restricted orthogonal polynomials for
pseudospectral methods in Python
"""

# ======================================================================
# imports
# ======================================================================
from __future__ import print_function
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
# A class for orthogonal polynomials
poly = polynomial.chebyshev.Chebyshev
weight_func = polynomial.chebyshev.chebweight
mono = polynomial.polynomial.Polynomial
integrator = integrate.quad
EVEN='even'
ODD='odd'
# ======================================================================

# ======================================================================
# Nodal and Modal Details
# ======================================================================
class Vandermonde:
    """A container class containing Vandermonde matrices.
    """
    def __init__(self,s2c,c2s):
        self.s2c = s2c
        self.c2s = c2s

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

def get_full_modal_differentiation_matrix(order):
    """
    Returns the differentiation matrix for the first derivative in the
    modal basis.

    Full operator. No parity-restricted.
    """
    out = np.zeros((order+1,order+1))
    for i in range(order+1):
        out[:i,i] = poly.basis(i).deriv().coef
    return out

def calculate_vandermonde(order,quads,parity):
    """Given a set of quadrature points,
    calculates the Vandermonde matrix that
    map data at collocation points to the coefficients
    of an interpolating polynomial of appropriate parity.
    """
    offset = 0 if parity == EVEN else 1
    s2c = np.zeros((order+1,order+1),dtype=float)
    for i in range(order+1):
        for j in range(order+1):
            s2c[i,j] = poly.basis(2*j+offset)(quads[i])
    c2s = linalg.inv(s2c)
    out = Vandermonde(s2c,c2s)
    return out

def filter_coefs(coeffs,parity):
    """Given a set of spectral coefficients,
    filters the coefficients to be only of the
    appropriate parity.
    """
    offset = 0 if parity == EVEN else 1
    out = coeffs[offset::2]
    return out

def calculate_dModal(order,parity):
    """For a given order, maps coefficients of polynomials of parity to
    coefficients of the derivative, which will have the opposite
    parity.
    """
    offset = 0 if parity == EVEN else 1
    next_parity = ODD if parity == EVEN else EVEN
    out = np.zeros((order+1,order+1))
    for i in range(order+1):
        coefs = poly.basis(2*i+offset).deriv().coef
        out[:i+offset,i] = filter_coefs(coefs,next_parity)
    return out

def calculate_div_x_modal(order,parity):
    """Maps a polynomial of order to that polynomial divided by x.
    Allowed only for odd parity.
    """
    assert parity == ODD
    out = np.zeros((order+1,order+1))
    x = mono([0,1]).convert(kind=poly)
    for i in range(order+1):
        coefs = (poly.basis(2*i+1)/x).coef
        out[:i+1,i] = filter_coefs(coefs,EVEN)
    return out

def make_nodal_operator(modal_operator,c2s,s2c):
    """Makes a nodal operator out of a modal operator and the appropriate
    Vandermonde matrices. Note that the appropriate Vandermonde
    matrices may be mixed. i.e., you might map nodal to modal for even
    functions but map modal to nodal for odd functions.
    """
    return np.dot(s2c,np.dot(modal_operator,c2s))
    
# ======================================================================


# ======================================================================
# Reconstruct Global Solution
# ======================================================================
def get_continuous_object(grid_func,c2s,parity,
                          xmin=LOCAL_XMIN,xmax=LOCAL_XMAX):
    """
    Maps the grid function grid_func, which is any field defined
    on the colocation points to a continuous function that can
    be evaluated.

    Parameters
    ----------
    xmin   -- the minimum value of the domain
    xmax   -- the maximum value of the domain
    c2s    -- The Vandermonde matrix that maps the colocation representation
              to the spectral representation
    parity -- The parity of grid_func about the origin

    Returns
    -------
    A numpy polynomial object which can be called to be evaluated
    """
    offset = 0 if parity == EVEN else 1
    num_modes = len(grid_func)
    total_modes=2*num_modes
    coefs = np.dot(c2s,grid_func)
    spec_func = np.zeros(total_modes)
    spec_func[offset::2] = coefs
    my_interp = poly(spec_func,domain=[xmin,xmax])
    return my_interp

def reference_inner_product(f,g):
    """Inner product <f,g> on interval [LOCAL_XMIN,LOCAL_XMAX]    
    """
    integrand = lambda x: f(x)*g(x)*weight_func(x)
    integral,err = integrate.quad(integrand,
                                  LOCAL_XMIN,LOCAL_XMAX,
                                  points = [0.0])
    return np.sqrt(integral/2)

def reference_2norm(f):
    "2norm of f on interval [LOCAL_XMIN,LOCAL_XMAX]"
    return reference_inner_product(f,f)
# ======================================================================


# ======================================================================
# A convenience class that generates everything and can be called
# ======================================================================
class ParityPseudoSpectralDiscretization1D:

    """Given an order, and a domain [xmin,xmax]
    defines internally all structures and methods the user needs
    to calculate spectral derivatives in 1D
    """
    def __init__(self,order,rmax=1.):
        "Constructor. Needs the order of the method and the domain [0,rmax]."
        self.order = order
        self.rmin = 0.
        self.rmax = rmax
        self.quads = get_parity_quadrature_points(self.order)
        self.weights = get_integration_weights(self.order,self.quads)
        # Vandermonde
        self.van_even = calculate_vandermonde(self.order,self.quads,EVEN)
        self.van_odd = calculate_vandermonde(self.order,self.quads,ODD)
        # modal operations
        self.dModal_even = calculate_dModal(self.order,EVEN)
        self.d2Modal_even = np.dot(self.dModal_even,self.dModal_even)
        self.dModal_odd = calculate_dModal(self.order,ODD)
        self.div_x_modal = calculate_div_x_modal(self.order,ODD)
        self.poisson_modal = (self.d2Modal_even
                              + 2*np.dot(self.div_x_modal,self.dModal_even))
        # nodal operations
        self.dNodal_even = make_nodal_operator(self.dModal_even,
                                               self.van_even.c2s,
                                               self.van_odd.s2c)
        self.dNodal_odd = make_nodal_operator(self.dModal_odd,
                                              self.van_odd.c2s,
                                              self.van_even.s2c)
        self.d2Nodal_even = make_nodal_operator(self.d2Modal_even,
                                                self.van_even.c2s,
                                                self.van_even.s2c)
        self.div_x_nodal = make_nodal_operator(self.div_x_modal,
                                               self.van_odd.c2s,
                                               self.van_even.s2c)
        self.poisson_nodal = make_nodal_operator(self.poisson_modal,
                                                 self.van_odd.c2s,
                                                 self.van_even.s2c)
        
        

        
# ======================================================================


# Warning not to run this program on the command line
if __name__ == "__main__":
    raise ImportError("Warning. This is a library. It contains no main function.")
