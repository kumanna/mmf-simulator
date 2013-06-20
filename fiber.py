"""
This module defines and implements the models for different fiber types.
"""
import numpy
import abc
import types
from math import factorial

class ModeFamily(object):
    """
    This is an abstract class that represents the modes of the
    fiber. This class is subclassed to obtain various mode patterns.
    """
    __metaclass__ = abc.ABCMeta

    mode_attributes = ['theta', 'offset_x', 'offset_y']
    def __init__(self, w, XX, YY, **kwargs):
        """
        Defines the mode family for a fiber or waveguide. A summary of
        the parameters follows:

        `XX`, `YY`: meshgrid inputs
        `w`: mode radius
        `offset_x`, `offset_y`: Evaluate the parameters off-center
        `theta`: Rotates the mode pattern by this angle
        """
        for i in self.mode_attributes:
            exec("self.%s = 0.0" % i)
            if i in kwargs.keys():
                exec("self.%s = %f" % (i, float(kwargs[i])))

            # Method functions to return paramters
            exec("f = lambda self : self.%s" % i)
            exec("f.__doc__ = \"Returns %s for this mode family\"" % (i))
            exec("self.get_%s = types.MethodType(f, self)" % i)
        self.XX = XX
        self.YY = YY
        self.w = w

    @abc.abstractmethod
    def get_mode_pattern(self, p, q):
        """
        Return the mode pattern for the mesh grid that the class was
        initialized with.
        """
        raise NotImplemented        

class GHModes(ModeFamily):
    """
    This class represents the Hermite-Gauss modes for a fiber. The
    definition of these parameters can be found in *Principal Modes in
    Graded-Index Multimode Fiber in Presence of Spatial- and
    Polarization-Mode Coupling* by Shemirani et al., IEEE/OSA Journal
    of Lightwave Technology, May 2009.

    An examples:
    >>> NA = 0.19
    >>> wavelength = 1.55e-6;
    >>> a = 31.25e-6/2;
    >>> k0 = 2 * numpy.pi / wavelength
    >>> n0 = 1.444;
    >>> Csk0 = 0.0878 * pow(n0, 3)
    >>> delta = 8000;
    >>> n_core = n0;
    >>> DELTA = 0.5 * pow((NA / n0), 2);
    >>> sqrt = numpy.sqrt
    >>> w = sqrt(sqrt(2) * a / (k0 * n0 * sqrt(DELTA)));
    >>> EXTENTS = 30e-6
    >>> STEP = 0.5e-6
    >>> x = numpy.arange(-EXTENTS, EXTENTS, STEP)
    >>> y = numpy.arange(-EXTENTS, EXTENTS, STEP)
    >>> [XX, YY] = numpy.meshgrid(x, y)
    >>> gh_modes = GHModes(w, XX, YY, theta = numpy.pi / 4)
    >>> # gh_modes.plot_mode_pattern(2, 2)
    >>> mode_pattern_1 = gh_modes.get_mode_pattern(0, 0)
    >>> mode_pattern_2 = gh_modes.get_mode_pattern(1, 1)
    >>> print "%.2f" % utils.overlap(mode_pattern_1, mode_pattern_1)
    1.00
    >>> print "%.2f" % utils.overlap(mode_pattern_2, mode_pattern_2)
    1.00
    >>> print "%.2f" % utils.overlap(mode_pattern_1, mode_pattern_2)
    0.00
    """
    def get_mode_pattern(self, p, q):
        """
        Return the Hermite-Gauss mode pattern for the mesh grid that
        the class was initialized with.
        """
        # Initialize the hermite polynomials
        Hp = numpy.polynomial.hermite.Hermite.basis(p)
        Hq = numpy.polynomial.hermite.Hermite.basis(q)

        # Convenience assignments
        XX = self.XX
        YY = self.YY
        theta = self.theta
        sqrt = numpy.sqrt
        pi = numpy.pi
        exp = numpy.exp
        w = self.w
        cos = numpy.cos
        sin = numpy.sin

        # Radial distances, with offsets
        R2 = numpy.square(XX - self.offset_x) + numpy.square(YY - self.offset_y)

        # Split the multiplication into steps
        P1 = sqrt(2 / pi) / w / sqrt(pow(2.0, p + q) * float(factorial(p)) * float(factorial(q)))
        P1 = 1.0
        P2 = P1 * Hp(sqrt(2.0) * (XX * cos(theta) + YY * sin(theta)) / w)
        P3 = numpy.multiply(P2, Hq(sqrt(2.0) * (XX * -sin(theta) + YY * cos(theta)) / w))
        P4 = numpy.multiply(P3, exp(-R2 / w / w))
        P5 = P4 / sqrt(numpy.sum(numpy.square(P4)))
        return P5

    def plot_mode_pattern(self, p, q):
        """
        Plot the `p`, `q` mode pattern.
        """
        try:
            import matplotlib
        except ImportError:
            print "Can't plot, because matplotlib seems missing"
            return
        import matplotlib.patches as patches
        import matplotlib.pyplot as pyplot

        pattern = self.get_mode_pattern(p, q)
        pattern = pattern / numpy.max(pattern)
        pyplot.imshow(numpy.abs(pattern), interpolation = 'bilinear', cmap = matplotlib.cm.spectral)
        pyplot.grid(True)
        pyplot.show()

class Fiber(object):
    """
    This object represents an abstract notion of a fiber. It is intended
    to be subclassed to implement a fiber model, based on various
    characteristics.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, length = 1000.0, step_length = 0.1):
        """
        Initialize the fiber with length and step size.
        
        """
        self.length = length
        self.step_length = step_length
        self.current_mode_vector = numpy.matrix([])

        self.transmitter_connected = False
        self.receiver_connected = False

    @abc.abstractmethod
    def connect_transmitter(self):
        """
        Connect a transmit laser array
        """
        raise NotImplemented

    @abc.abstractmethod
    def connect_receiver(self):
        """
        Connect a receiver detector array
        """
        raise NotImplemented

if __name__ == "__main__":
    import doctest
    import utils
    doctest.testmod()
