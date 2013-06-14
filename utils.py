#!/usr/bin/python
"""
This module includes various utility functions used during the
simulation phases. For example:

>>> M1 = numpy.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> M2 = numpy.matrix([[9, 8, 7], [6, 5, 4], [3, 2, 1]])*1j
>>> res = overlap(M1, M2)
>>> print "%.6f+%.6f" % (res.real, res.imag)
0.000000+-0.578947
"""

import numpy

def overlap(Er1, Er2):
    """
    This method performs an overlap of the two matrices ``Er1`` and
    ``Er2`` and returns a single number represents the
    integral. ``Er2`` is conjugated before an element-by-element
    multiplication.
    """
    if Er1.shape != Er2.shape:
        raise ValueError('Matrix shapes do not match for overlap!')
        
    # First find the normalization for the matrices
    E1_mag = numpy.linalg.norm(Er1, ord='fro')
    E2_mag = numpy.linalg.norm(Er1, ord='fro')

    return numpy.sum(numpy.multiply(Er1, Er2.conj())) / E1_mag / E2_mag

if __name__ == "__main__":
    import doctest
    doctest.testmod()
