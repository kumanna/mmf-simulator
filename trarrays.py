#!/usr/bin/python
"""
This class defines and implements the transmit and receive arrays.

>>> EXTENTS = 30e-6 # Larger than the diameter
>>> STEP = 0.5e-6 # fixed for now
>>> t_array = Transmitter_Array(EXTENTS, STEP)
>>> t_array.add_element(0.0, 0.0, 5.0e-6)
>>> t_array.add_element(10.0e-6, 10.0e-6, 5.0e-6)
>>> for i in t_array.get_elements():
...     print "Array element at (" + str(i.x) + ", " + str(i.y) + ") with diameter " + str(i.diameter) + "."
Array element at (0.0, 0.0) with diameter 5e-06.
Array element at (1e-05, 1e-05) with diameter 5e-06.
"""
import abc
import fiber
import numpy
import utils

class LDArrayElement(object):
    """
    Class that represents each element of the laser/detector array
    """
    def __init__(self, x = 0.0, y = 0.0, diameter = 5.0e-6, modes = None):
        """
        Initialize a laser/detector element with a diameter of ``diameter`` and center (``x``, ``y``).
        """
        self.x = x
        self.y = y
        self.diameter = diameter
        self.modes = modes

    def get_mode_pattern(self, p = 0, q = 0):
        return self.modes.get_mode_pattern(p, q)

class ModeFilteredElement(LDArrayElement):
    """
    Class to hold special elements that contain mode filtered
    transmt/receive lasers/detectors.
    """
    def __init__(self, w, XX, YY, p = 0, q = 0, offset_x = 0.0, offset_y = 0.0, theta = 0.0):
        """
        Adds a laser or detector with a specific mode pattern.
        """
        super(ModeFilteredElement, self).__init__(offset_x, offset_y, w)
        modes = fiber.GHModes(w, XX, YY, offset_x = offset_x, offset_y = offset_y, theta = theta)
        self.mode_pattern = modes.get_mode_pattern(p, q)

    def get_mode_pattern(self, p = 0, q = 0):
        """
        Returns the specific mode pattern
        """
        return self.mode_pattern

class TR_Array(object):
    """
    Base class that defines the characteristic of laser and detector arrays
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._device_list = []
        self.modes = None

    @abc.abstractmethod
    def overlap_matrix(self, fiber_instance):
        """
        This method returns the overlap matrix for calculating the
        MIMO channel matrix of a complete system.
        """
        element_arrays = []
        for device in self._device_list:
            Er1 = device.get_mode_pattern(0, 0)
            fiber_modes = fiber_instance.get_admissible_modes()
            M = len(fiber_modes)
            mode_vector = numpy.zeros(2 * M)
            for i in range(M):
                current_mode = fiber_modes[i]
                Er2 = fiber_instance.modes.get_mode_pattern(current_mode[0], current_mode[1])
                mode_vector[i] = mode_vector[M + i] = utils.overlap(Er1, Er2)
            element_arrays.append(mode_vector)
        element_arrays = numpy.array(element_arrays)
        return element_arrays

    def add_element(self, x, y, diameter = 5.0e-6, modes = None):
        """
        Add a circular laser/detector element, with ``diameter''
        specified in microns.
        """
        self._device_list.append(LDArrayElement(x, y, diameter, modes))

    def add_filtered_element(self, element):
        """
        Adds a mode-filtered element
        """
        self._device_list.append(element)

    def get_elements(self):
        return self._device_list

    def plot_system(self, fiber_diameter = 62.5e-6, fec = 'black', ffc = 'white', dec = 'black', dfc = 'blue'):
        """
        Plot the array using a fiber with ``fiber_diameter`` as backdrop.

        Parameters:
        ``fec``: Fiber edge colour
        ``ffc``: Fiber face colour
        ``dec``: Device edge colour
        ``dfc``: Device face colour
        """
        try:
            import matplotlib
        except ImportError:
            print "Can't plot, because matplotlib seems missing"
            return
        import matplotlib.patches as patches
        import matplotlib.pyplot as pyplot

        # First, create a patch for the fiber circle
        fiber_circle = patches.Circle(xy = (0.0, 0.0), radius = fiber_diameter / 2.0, ec = fec, fc = ffc)

        # Next, the device element patches
        device_patches = []
        for i in self._device_list:
            device_patches.append(patches.Circle(xy = (i.x, i.y), radius = i.diameter / 2.0, ec = dec, fc = dfc))

        fig = pyplot.figure(figsize=(8,8))
        ax = fig.add_subplot(111)

        # Add the fiber circle first
        ax.add_patch(fiber_circle)

        # Now, add the device elements
        for i in device_patches: ax.add_patch(i)
        excess_margin = 1.05
        ax.set_xlim(-fiber_diameter / 2.0 * excess_margin, fiber_diameter / 2.0 * excess_margin)
        ax.set_ylim(-fiber_diameter / 2.0 * excess_margin, fiber_diameter / 2.0 * excess_margin)
        pyplot.grid(True)
        pyplot.show()

class Transmitter_Array(TR_Array):
    """
    A transmitter array element. Most of the functionality is derived
    from the ``TR_Array'' class.
    """

    def __init__(self, extents, step):
        super(Transmitter_Array, self).__init__()
        x = numpy.arange(-extents, extents, step)
        y = numpy.arange(-extents, extents, step)
        [self.XX, self.YY] = numpy.meshgrid(x, y)

    def add_element(self, x, y, diameter = 5.0e-6):
        modes = fiber.GHModes(diameter, self.XX, self.YY, offset_x = x, offset_y = y)
        super(Transmitter_Array, self).add_element(x, y, diameter, modes)
        #self.plot_system()

    def overlap_matrix(self, fiber_instance):
        return super(Transmitter_Array, self).overlap_matrix(fiber_instance)

class Receiver_Array(TR_Array):
    """
    A receiver array element. Most of the functionality is derived
    from the ``TR_Array'' class.
    """

    def __init__(self, extents, step):
        super(Receiver_Array, self).__init__()
        x = numpy.arange(-extents, extents, step)
        y = numpy.arange(-extents, extents, step)
        [self.XX, self.YY] = numpy.meshgrid(x, y)

    def add_element(self, x, y, diameter = 5.0e-6):
        modes = fiber.GHModes(diameter, self.XX, self.YY, offset_x = x, offset_y = y)
        super(Receiver_Array, self).add_element(x, y, diameter, modes)
        #self.plot_system()

    def overlap_matrix(self, fiber_instance):
        return super(Receiver_Array, self).overlap_matrix(fiber_instance).conj().T

if __name__ == "__main__":
    import doctest
    doctest.testmod()
