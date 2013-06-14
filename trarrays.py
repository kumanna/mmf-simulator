#!/usr/bin/python
"""
This class defines and implements the transmit and receive arrays.

>>> t_array = Transmitter_Array()
>>> t_array.add_element(0.0, 0.0, 5.0)
>>> t_array.add_element(10.0, 10.0, 5.0)
>>> for i in t_array.get_elements():
...     print "Array element at (" + str(i.x) + ", " + str(i.y) + ") with diameter " + str(i.diameter) + "."
Array element at (0.0, 0.0) with diameter 5.0.
Array element at (10.0, 10.0) with diameter 5.0.
"""
import abc

class LDArrayElement:
    """
    Class that represents each element of the laser/detector array
    """
    def __init__(self, x = 0.0, y = 0.0, diameter = 5.0):
        """
        Initialize a laser/detector element with a diameter of ``diameter`` and center (``x``, ``y``).
        """
        self.x = x
        self.y = y
        self.diameter = diameter

class TR_Array(object):
    """
    Base class that defines the characteristic of laser and detector arrays
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.__device_list = []

    @abc.abstractmethod
    def overlap_matrix(self):
        """
        This method returns the overlap matrix for calculating the
        MIMO channel matrix of a complete system.
        """
        raise NotImplemented

    def add_element(self, x, y, diameter = 5.0):
        """
        Add a circular laser/detector element, with ``diameter''
        specified in microns.
        """
        self.__device_list.append(LDArrayElement(x, y, diameter))

    def get_elements(self):
        return self.__device_list

    def plot_system(self, fiber_diameter = 62.5, fec = 'black', ffc = 'white', dec = 'black', dfc = 'blue'):
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
        for i in self.__device_list:
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
    def overlap_matrix(self):
        pass

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    t_array = Transmitter_Array()
    t_array.add_element(0.0, 0.0, 5.0)
    t_array.add_element(10.0, 10.0, 5.0)
    t_array.plot_system()
