"""
Some simple tools for making model configuration nicer.
"""
from abc import ABCMeta, abstractproperty
import logging
from logging import DEBUG, INFO, WARNING
from firedrake import sqrt


__all__ = ["WARNING", "INFO", "DEBUG", "TimesteppingParameters", "OutputParameters", "CompressibleParameters", "ShallowWaterParameters", "EadyParameters", "CompressibleEadyParameters", "logger", "EmbeddedDGOptions", "RecoveredOptions"]

logger = logging.getLogger("gusto")


def set_log_handler(comm):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt="%(name)s:%(levelname)s %(message)s"))
    if logger.hasHandlers():
        logger.handlers.clear()
    if comm.rank == 0:
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())


class Configuration(object):

    def __init__(self, **kwargs):

        for name, value in kwargs.items():
            self.__setattr__(name, value)

    def __setattr__(self, name, value):
        """Cause setting an unknown attribute to be an error"""
        if not hasattr(self, name):
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))
        object.__setattr__(self, name, value)


class TimesteppingParameters(Configuration):

    """
    Timestepping parameters for Gusto
    """
    dt = None
    alpha = 0.5
    maxk = 4
    maxi = 1


class OutputParameters(Configuration):

    """
    Output parameters for Gusto
    """

    #: log_level for logger, can be DEBUG, INFO or WARNING. Takes
    #: default value "warning"
    log_level = WARNING
    dump_vtus = True
    dumpfreq = 1
    dumplist = None
    dumplist_latlon = []
    dump_diagnostics = True
    checkpoint = True
    chkptfreq = 1
    dirname = None
    #: Should the output fields be interpolated or projected to
    #: a linear space?  Default is interpolation.
    project_fields = False
    #: List of fields to dump error fields for steady state simulation
    steady_state_error_fields = []
    #: List of fields for computing perturbations
    perturbation_fields = []
    #: List of ordered pairs (name, points) where name is the field
    # name and points is the points at which to dump them
    point_data = []


class CompressibleParameters(Configuration):

    """
    Physical parameters for Compressible Euler
    """
    g = 9.810616
    N = 0.01  # Brunt-Vaisala frequency (1/s)
    cp = 1004.5  # SHC of dry air at const. pressure (J/kg/K)
    R_d = 287.  # Gas constant for dry air (J/kg/K)
    kappa = 2.0/7.0  # R_d/c_p
    p_0 = 1000.0*100.0  # reference pressure (Pa, not hPa)
    cv = 717.  # SHC of dry air at const. volume (J/kg/K)
    c_pl = 4186.  # SHC of liq. wat. at const. pressure (J/kg/K)
    c_pv = 1885.  # SHC of wat. vap. at const. pressure (J/kg/K)
    c_vv = 1424.  # SHC of wat. vap. at const. pressure (J/kg/K)
    R_v = 461.  # gas constant of water vapour
    L_v0 = 2.5e6  # ref. value for latent heat of vap. (J/kg)
    T_0 = 273.15  # ref. temperature
    w_sat1 = 380.3  # first const. in Teten's formula (Pa)
    w_sat2 = -17.27  # second const. in Teten's formula (no units)
    w_sat3 = 35.86  # third const. in Teten's formula (K)
    w_sat4 = 610.9  # fourth const. in Teten's formula (Pa)


class ShallowWaterParameters(Configuration):

    """
    Physical parameters for 3d Compressible Euler
    """
    g = 9.80616
    Omega = 7.292e-5  # rotation rate
    H = None  # mean depth


class EadyParameters(Configuration):

    """
    Physical parameters for Incompressible Eady
    """
    Nsq = 2.5e-05  # squared Brunt-Vaisala frequency (1/s)
    dbdy = -1.0e-07
    H = None
    L = None
    f = None
    deltax = None
    deltaz = None
    fourthorder = False


class CompressibleEadyParameters(CompressibleParameters, EadyParameters):

    """
    Physical parameters for Compressible Eady
    """
    g = 10.
    N = sqrt(EadyParameters.Nsq)
    theta_surf = 300.
    dthetady = theta_surf/g*EadyParameters.dbdy
    Pi0 = 0.0


class AdvectionOptions(Configuration, metaclass=ABCMeta):

    @abstractproperty
    def name(self):
        pass


class EmbeddedDGOptions(AdvectionOptions):

    name = "embedded_dg"
    embedding_space = None


class RecoveredOptions(AdvectionOptions):

    name = "recovered"
    embedding_space = None
    recovered_space = None
    broken_space = None
    boundary_method = None
    spherical_transformation = False


class SUPGOptions(AdvectionOptions):

    name = "supg"
    tau = None
    default = 1/sqrt(15)
