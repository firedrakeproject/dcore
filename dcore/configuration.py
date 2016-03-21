"""
Some simple tools for making model configuration nicer.
"""


class Configuration(object):

    def __init__(self, **kwargs):

        for name, value in kwargs.iteritems():
            self.__setattr__(name, value)

    def __setattr__(self, name, value):
        """Cause setting an unknown attribute to be an error"""
        if not hasattr(self, name):
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))
        object.__setattr__(self, name, value)


class TimesteppingParameters(Configuration):

    """
    Timestepping parameters for dcore
    """
    dt = None
    alpha = 0.5
    maxk = 2
    maxi = 2


class OutputParameters(Configuration):

    """
    Output parameters for dcore
    """

    Verbose = False
    dumpfreq = 10
    dumplist = None
    dirname = None


class CompressibleParameters(Configuration):

    """
    Physical parameters for 3d Compressible Euler
    """
    g = 9.81
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cp = 1004.5  # SHC of dry air at const. pressure (J/kg/K)
    R_d = 287.  # Gas constant for dry air (J/kg/K)
    kappa = 2.0/7.0  # R_d/c_p
    p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)
    k = None  # vertical direction
    Omega = None  # rotation vector


class ShallowWaterParameters(Configuration):

    """
    Physical parameters for 3d Compressible Euler
    """
    g = 9.806
    Omega = 7.292e-5  # rotation rate