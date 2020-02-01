from abc import ABCMeta, abstractmethod
from gusto.transport_equation import EmbeddedDGAdvection
from gusto.recovery import Recoverer, Boundary_Method
from gusto.advection import SSPRK3
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter
from gusto.limiters import ThetaLimiter, NoLimiter
from gusto.configuration import logger, EmbeddedDGOptions, RecoveredOptions
from firedrake import (Interpolator, conditional, Function,
                       min_value, max_value, as_vector, BrokenElement, FunctionSpace,
                       Constant, pi, Projector)
from gusto import thermodynamics
from math import gamma
from enum import Enum


__all__ = ["Condensation", "Fallout", "Coalescence", "Evaporation", "AdvectedMoments"]


class Physics(object, metaclass=ABCMeta):
    """
    Base class for physics processes for Gusto.

    :arg state: :class:`.State` object.
    """

    def __init__(self, state):
        self.state = state

    @abstractmethod
    def apply(self):
        """
        Function computes the value of specific
        fields at the next time step.
        """
        pass


class Condensation(Physics):
    """
    The process of condensation of water vapour
    into liquid water and evaporation of liquid
    water into water vapour, with the associated
    latent heat changes. The parametrization follows
    that used in Bryan and Fritsch (2002).

    :arg state: :class:`.State.` object.
    :arg iterations: number of iterations to do
         of condensation scheme per time step.
    """

    def __init__(self, state, iterations=1):
        super().__init__(state)

        self.iterations = iterations
        # obtain our fields
        self.theta = state.fields('theta')
        self.water_v = state.fields('water_v')
        self.water_c = state.fields('water_c')
        rho = state.fields('rho')
        try:
            rain = state.fields('rain')
            water_l = self.water_c + rain
        except NotImplementedError:
            water_l = self.water_c

        # declare function space
        Vt = self.theta.function_space()

        # make rho variables
        # we recover rho into theta space
        if state.vertical_degree == 0 and state.horizontal_degree == 0:
            boundary_method = Boundary_Method.physics
        else:
            boundary_method = None
        Vt_broken = FunctionSpace(state.mesh, BrokenElement(Vt.ufl_element()))
        rho_averaged = Function(Vt)
        self.rho_recoverer = Recoverer(rho, rho_averaged, VDG=Vt_broken, boundary_method=boundary_method)

        # define some parameters as attributes
        dt = state.timestepping.dt
        R_d = state.parameters.R_d
        cp = state.parameters.cp
        cv = state.parameters.cv
        c_pv = state.parameters.c_pv
        c_pl = state.parameters.c_pl
        c_vv = state.parameters.c_vv
        R_v = state.parameters.R_v

        # make useful fields
        Pi = thermodynamics.pi(state.parameters, rho_averaged, self.theta)
        T = thermodynamics.T(state.parameters, self.theta, Pi, r_v=self.water_v)
        p = thermodynamics.p(state.parameters, Pi)
        L_v = thermodynamics.Lv(state.parameters, T)
        R_m = R_d + R_v * self.water_v
        c_pml = cp + c_pv * self.water_v + c_pl * water_l
        c_vml = cv + c_vv * self.water_v + c_pl * water_l

        # use Teten's formula to calculate w_sat
        w_sat = thermodynamics.r_sat(state.parameters, T, p)

        # make appropriate condensation rate
        dot_r_cond = ((self.water_v - w_sat)
                      / (dt * (1.0 + ((L_v ** 2.0 * w_sat)
                                      / (cp * R_v * T ** 2.0)))))

        # make cond_rate function, that needs to be the same for all updates in one time step
        cond_rate = Function(Vt)

        # adjust cond rate so negative concentrations don't occur
        self.lim_cond_rate = Interpolator(conditional(dot_r_cond < 0,
                                                      max_value(dot_r_cond, - self.water_c / dt),
                                                      min_value(dot_r_cond, self.water_v / dt)), cond_rate)

        # tell the prognostic fields what to update to
        self.water_v_new = Interpolator(self.water_v - dt * cond_rate, Vt)
        self.water_c_new = Interpolator(self.water_c + dt * cond_rate, Vt)
        self.theta_new = Interpolator(self.theta
                                      * (1.0 + dt * cond_rate
                                         * (cv * L_v / (c_vml * cp * T)
                                            - R_v * cv * c_pml / (R_m * cp * c_vml))), Vt)

    def apply(self):
        self.rho_recoverer.project()
        for i in range(self.iterations):
            self.lim_cond_rate.interpolate()
            self.theta.assign(self.theta_new.interpolate())
            self.water_v.assign(self.water_v_new.interpolate())
            self.water_c.assign(self.water_c_new.interpolate())


class AdvectedMoments(Enum):
    """
    An Enum object storing moments of the rain drop
    size distribution. This is then used in deciding
    which to be advected in the Fallout step of the model.
    """

    M0 = 0  # don't advect the distribution -- advect all rain at the same speed
    M3 = 1  # advect the mass of the distribution


class Fallout(Physics):
    """
    The fallout process of hydrometeors.

    :arg state :class: `.State.` object.
    :arg moments: an AdvectedMoments Enum object, indicating which
                rainfall scheme to use. Current valid values are:
                AdvectedMoments.M0 -- advects all rain at constant speed;
                AdvectedMoments.M3 -- advects mean mass of droplet distribution.
                The default value is AdvectedMoments.M3.
    :arg limit: if True (the default value), applies a limiter to the
                rainfall advection.
    """

    def __init__(self, state, moments=AdvectedMoments.M3, limit=True):
        super().__init__(state)

        # function spaces
        Vt = state.fields('rain').function_space()
        Vu = state.fields('u').function_space()

        # declare properties of class
        self.state = state
        self.moments = moments
        self.rain = state.fields('rain')
        self.v = state.fields('rainfall_velocity', Vu)
        self.limit = limit

        if moments == AdvectedMoments.M0:
            # all rain falls at terminal velocity
            terminal_velocity = Constant(5)  # in m/s
            if state.mesh.geometric_dimension() == 2:
                self.v.project(as_vector([0, -terminal_velocity]))
            elif state.mesh.geometric_dimension() == 3:
                self.v.project(as_vector([0, 0, -terminal_velocity]))
        elif moments == AdvectedMoments.M3:
            # this advects the third moment M3 of the raindrop
            # distribution, which corresponds to the mean mass
            rho = state.fields('rho')
            rho_w = Constant(1000.0)  # density of liquid water
            # assume n(D) = n_0 * D^mu * exp(-Lambda*D)
            # n_0 = N_r * Lambda^(1+mu) / gamma(1 + mu)
            N_r = Constant(10**5)  # number of rain droplets per m^3
            mu = 0.0  # shape constant of droplet gamma distribution
            # assume V(D) = a * D^b * exp(-f*D) * (rho_0 / rho)^g
            # take f = 0
            a = Constant(362.)  # intercept for velocity distr. in log space
            b = 0.65  # inverse scale parameter for velocity distr.
            rho0 = Constant(1.22)  # reference density in kg/m^3
            g = Constant(0.5)  # scaling of density correction
            # we keep mu in the expressions even though mu = 0
            threshold = Constant(10**-10)  # only do rainfall for r > threshold
            Lambda = (N_r * pi * rho_w * gamma(4 + mu)
                      / (6 * gamma(1 + mu) * rho * self.rain)) ** (1. / 3)
            Lambda0 = (N_r * pi * rho_w * gamma(4 + mu)
                       / (6 * gamma(1 + mu) * rho * threshold)) ** (1. / 3)
            v_expression = conditional(self.rain > threshold,
                                       (a * gamma(4 + b + mu)
                                        / (gamma(4 + mu) * Lambda ** b)
                                        * (rho0 / rho) ** g),
                                       (a * gamma(4 + b + mu)
                                        / (gamma(4 + mu) * Lambda0 ** b)
                                        * (rho0 / rho) ** g))
        else:
            raise NotImplementedError('Currently we only have implementations for zero and one moment schemes for rainfall. Valid options are AdvectedMoments.M0 and AdvectedMoments.M3')

        if moments != AdvectedMoments.M0:
            if state.mesh.geometric_dimension() == 2:
                self.determine_v = Projector(as_vector([0, -v_expression]), self.v)
            elif state.mesh.geometric_dimension() == 3:
                self.determine_v = Projector(as_vector([0, 0, -v_expression]), self.v)

        # determine whether to do recovered space advection scheme
        # if horizontal and vertical degrees are 0 do recovered space
        if state.horizontal_degree == 0 and state.vertical_degree == 0:
            VDG1 = state.spaces("DG1")
            VCG1 = FunctionSpace(Vt.mesh(), "CG", 1)
            Vbrok = FunctionSpace(Vt.mesh(), BrokenElement(Vt.ufl_element()))
            boundary_method = Boundary_Method.dynamics
            advect_options = RecoveredOptions(embedding_space=VDG1,
                                              recovered_space=VCG1,
                                              broken_space=Vbrok,
                                              boundary_method=boundary_method)
        else:
            advect_options = EmbeddedDGOptions()

        # need to define advection equation before limiter (as it is needed for the ThetaLimiter)
        advection_equation = EmbeddedDGAdvection(state, Vt, equation_form="advective", outflow=True, options=advect_options)

        # decide which limiter to use
        if self.limit:
            if state.horizontal_degree == 0 and state.vertical_degree == 0:
                limiter = VertexBasedLimiter(VDG1)
            elif state.horizontal_degree == 1 and state.vertical_degree == 1:
                limiter = ThetaLimiter(Vt)
            else:
                logger.warning("There is no limiter yet implemented for the spaces used. NoLimiter() is being used for the rainfall in this case.")
                limiter = NoLimiter()
        else:
            limiter = None

        # sedimentation will happen using a full advection method
        self.advection_method = SSPRK3(state, self.rain, advection_equation, limiter=limiter)

    def apply(self):
        if self.moments != AdvectedMoments.M0:
            self.determine_v.project()
        self.advection_method.update_ubar(self.v, self.v, 0)
        self.advection_method.apply(self.rain, self.rain)


class Coalescence(Physics):
    """
    The process of the coalescence of cloud
    droplets to form rain droplets. These
    parametrizations come from Klemp and
    Wilhelmson (1978).

    :arg state: :class:`.State.` object.
    :arg accretion: Boolean which determines
                    whether the accretion
                    process is used.
    :arg accumulation: Boolean which determines
                    whether the accumulation
                    process is used.
    """

    def __init__(self, state, accretion=True, accumulation=True):
        super().__init__(state)

        # obtain our fields
        self.water_c = state.fields('water_c')
        self.rain = state.fields('rain')

        # declare function space
        Vt = self.water_c.function_space()

        # define some parameters as attributes
        dt = state.timestepping.dt
        k_1 = Constant(0.001)  # accretion rate in 1/s
        k_2 = Constant(2.2)  # accumulation rate in 1/s
        a = Constant(0.001)  # min cloud conc in kg/kg
        b = Constant(0.875)  # power for rain in accumulation

        # make default rates to be zero
        accr_rate = Constant(0.0)
        accu_rate = Constant(0.0)

        if accretion:
            accr_rate = k_1 * (self.water_c - a)
        if accumulation:
            accu_rate = k_2 * self.water_c * self.rain ** b

        # make coalescence rate function, that needs to be the same for all updates in one time step
        coalesce_rate = Function(Vt)

        # adjust coalesce rate using min_value so negative cloud concentration doesn't occur
        self.lim_coalesce_rate = Interpolator(conditional(self.rain < 0.0,  # if rain is negative do only accretion
                                                          conditional(accr_rate < 0.0,
                                                                      0.0,
                                                                      min_value(accr_rate, self.water_c / dt)),
                                                          # don't turn rain back into cloud
                                                          conditional(accr_rate + accu_rate < 0.0,
                                                                      0.0,
                                                                      # if accretion rate is negative do only accumulation
                                                                      conditional(accr_rate < 0.0,
                                                                                  min_value(accu_rate, self.water_c / dt),
                                                                                  min_value(accr_rate + accu_rate, self.water_c / dt)))),
                                              coalesce_rate)

        # tell the prognostic fields what to update to
        self.water_c_new = Interpolator(self.water_c - dt * coalesce_rate, Vt)
        self.rain_new = Interpolator(self.rain + dt * coalesce_rate, Vt)

    def apply(self):
        self.lim_coalesce_rate.interpolate()
        self.rain.assign(self.rain_new.interpolate())
        self.water_c.assign(self.water_c_new.interpolate())


class Evaporation(Physics):
    """
    The process of evaporation of rain into water vapour
    with the associated latent heat change. This
    parametrization comes from Klemp and Wilhelmson (1978).

    :arg state: :class:`.State.` object.
    """

    def __init__(self, state):
        super().__init__(state)

        # obtain our fields
        self.theta = state.fields('theta')
        self.water_v = state.fields('water_v')
        self.rain = state.fields('rain')
        rho = state.fields('rho')
        try:
            water_c = state.fields('water_c')
            water_l = self.rain + water_c
        except NotImplementedError:
            water_l = self.rain

        # declare function space
        Vt = self.theta.function_space()

        # make rho variables
        # we recover rho into theta space
        if state.vertical_degree == 0 and state.horizontal_degree == 0:
            boundary_method = Boundary_Method.physics
        else:
            boundary_method = None
        Vt_broken = FunctionSpace(state.mesh, BrokenElement(Vt.ufl_element()))
        rho_averaged = Function(Vt)
        self.rho_recoverer = Recoverer(rho, rho_averaged, VDG=Vt_broken, boundary_method=boundary_method)

        # define some parameters as attributes
        dt = state.timestepping.dt
        R_d = state.parameters.R_d
        cp = state.parameters.cp
        cv = state.parameters.cv
        c_pv = state.parameters.c_pv
        c_pl = state.parameters.c_pl
        c_vv = state.parameters.c_vv
        R_v = state.parameters.R_v

        # make useful fields
        Pi = thermodynamics.pi(state.parameters, rho_averaged, self.theta)
        T = thermodynamics.T(state.parameters, self.theta, Pi, r_v=self.water_v)
        p = thermodynamics.p(state.parameters, Pi)
        L_v = thermodynamics.Lv(state.parameters, T)
        R_m = R_d + R_v * self.water_v
        c_pml = cp + c_pv * self.water_v + c_pl * water_l
        c_vml = cv + c_vv * self.water_v + c_pl * water_l

        # use Teten's formula to calculate w_sat
        w_sat = thermodynamics.r_sat(state.parameters, T, p)

        # expression for ventilation factor
        a = Constant(1.6)
        b = Constant(124.9)
        c = Constant(0.2046)
        C = a + b * (rho_averaged * self.rain) ** c

        # make appropriate condensation rate
        f = Constant(5.4e5)
        g = Constant(2.55e6)
        h = Constant(0.525)
        dot_r_evap = (((1 - self.water_v / w_sat) * C * (rho_averaged * self.rain) ** h)
                      / (rho_averaged * (f + g / (p * w_sat))))

        # make evap_rate function, needs to be the same for all updates in one time step
        evap_rate = Function(Vt)

        # adjust evap rate so negative rain doesn't occur
        self.lim_evap_rate = Interpolator(conditional(dot_r_evap < 0,
                                                      0.0,
                                                      conditional(self.rain < 0.0,
                                                                  0.0,
                                                                  min_value(dot_r_evap, self.rain / dt))),
                                          evap_rate)

        # tell the prognostic fields what to update to
        self.water_v_new = Interpolator(self.water_v + dt * evap_rate, Vt)
        self.rain_new = Interpolator(self.rain - dt * evap_rate, Vt)
        self.theta_new = Interpolator(self.theta
                                      * (1.0 - dt * evap_rate
                                         * (cv * L_v / (c_vml * cp * T)
                                            - R_v * cv * c_pml / (R_m * cp * c_vml))), Vt)

    def apply(self):
        self.rho_recoverer.project()
        self.lim_evap_rate.interpolate()
        self.theta.assign(self.theta_new.interpolate())
        self.water_v.assign(self.water_v_new.interpolate())
        self.rain.assign(self.rain_new.interpolate())
