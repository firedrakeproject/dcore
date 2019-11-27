from firedrake import (split, LinearVariationalProblem, Constant,
                       LinearVariationalSolver, TestFunctions, TrialFunctions,
                       TestFunction, TrialFunction, lhs, rhs, FacetNormal,
                       div, dx, jump, avg, dS_v, dS_h, ds_v, ds_t, ds_b, ds_tb, inner,
                       dot, grad, Function, VectorSpaceBasis, BrokenElement,
                       FunctionSpace, MixedFunctionSpace)
from firedrake.petsc import flatten_parameters
from firedrake.parloops import par_loop, READ, INC
from pyop2.profiling import timed_function, timed_region

from gusto.configuration import logger, DEBUG
from gusto import thermodynamics
from abc import ABCMeta, abstractmethod, abstractproperty


__all__ = ["IncompressibleSolver", "ShallowWaterSolver", "CompressibleSolver"]


class TimesteppingSolver(object, metaclass=ABCMeta):
    """
    Base class for timestepping linear solvers for Gusto.

    This is a dummy base class.

    :arg state: :class:`.State` object.
    :arg solver_parameters (optional): solver parameters
    :arg overwrite_solver_parameters: boolean, if True use only the
         solver_parameters that have been passed in, if False then update
         the default solver parameters with the solver_parameters passed in.
    """

    def __init__(self, state, solver_parameters=None,
                 overwrite_solver_parameters=False):

        self.state = state

        if solver_parameters is not None:
            if not overwrite_solver_parameters:
                p = flatten_parameters(self.solver_parameters)
                p.update(flatten_parameters(solver_parameters))
                solver_parameters = p
            self.solver_parameters = solver_parameters

        if logger.isEnabledFor(DEBUG):
            self.solver_parameters["ksp_monitor_true_residual"] = None

        # setup the solver
        self._setup_solver()

    @abstractproperty
    def solver_parameters(self):
        """Solver parameters for this solver"""
        pass

    @abstractmethod
    def solve(self):
        pass


class CompressibleSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the compressible equations
    in theta-pi formulation with prognostic variables u, rho, and theta.

    This solver follows the following strategy:

    (1) Analytically eliminate theta (introduces error near topography)

    (2a) Formulate the resulting mixed system for u and rho using a
         hybridized mixed method. This breaks continuity in the
         linear perturbations of u, and introduces a new unknown on the
         mesh interfaces approximating the average of the Exner pressure
         perturbations. These trace unknowns also act as Lagrange
         multipliers enforcing normal continuity of the "broken" u variable.

    (2b) Statically condense the block-sparse system into a single system
         for the Lagrange multipliers. This is the only globally coupled
         system requiring a linear solver.

    (2c) Using the computed trace variables, we locally recover the
         broken velocity and density perturbations. This is accomplished
         in two stages:
         (i): Recover rho locally using the multipliers.
         (ii): Recover "broken" u locally using rho and the multipliers.

    (2d) Project the "broken" velocity field into the HDiv-conforming
         space using local averaging.

    (3) Reconstruct theta

    :arg state: a :class:`.State` object containing everything else.
    :arg quadrature degree: tuple (q_h, q_v) where q_h is the required
         quadrature degree in the horizontal direction and q_v is that in
         the vertical direction.
    :arg solver_parameters (optional): solver parameters for the
         trace system.
    :arg overwrite_solver_parameters: boolean, if True use only the
         solver_parameters that have been passed in, if False then update.
         the default solver parameters with the solver_parameters passed in.
    :arg moisture (optional): list of names of moisture fields.
    """

    solver_parameters = {'mat_type': 'matfree',
                         'ksp_type': 'preonly',
                         'pc_type': 'python',
                         'pc_python_type': 'firedrake.SCPC',
                         'pc_sc_eliminate_fields': '0, 1',
                         # The reduced operator is not symmetric
                         'condensed_field': {'ksp_type': 'fgmres',
                                             'ksp_rtol': 1.0e-8,
                                             'ksp_atol': 1.0e-8,
                                             'ksp_max_it': 100,
                                             'pc_type': 'gamg',
                                             'pc_gamg_sym_graph': None,
                                             'mg_levels': {'ksp_type': 'gmres',
                                                           'ksp_max_it': 5,
                                                           'pc_type': 'bjacobi',
                                                           'sub_pc_type': 'ilu'}}}

    def __init__(self, state, quadrature_degree=None, solver_parameters=None,
                 overwrite_solver_parameters=False, moisture=None):

        self.moisture = moisture

        self.state = state

        if quadrature_degree is not None:
            self.quadrature_degree = quadrature_degree
        else:
            dgspace = state.spaces("DG")
            if any(deg > 2 for deg in dgspace.ufl_element().degree()):
                logger.warning("default quadrature degree most likely not sufficient for this degree element")
            self.quadrature_degree = (5, 5)

        if logger.isEnabledFor(DEBUG):
            # Set outer solver to FGMRES and turn on KSP monitor for the outer system
            self.solver_parameters["ksp_type"] = "fgmres"
            self.solver_parameters["mat_type"] = "aij"
            self.solver_parameters["pmat_type"] = "matfree"
            self.solver_parameters["ksp_monitor_true_residual"] = None

            # Turn monitor on for the trace system
            self.solver_parameters["condensed_field"]["ksp_monitor_true_residual"] = None

        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    @timed_function("Gusto:SolverSetup")
    def _setup_solver(self):
        import numpy as np

        state = self.state
        Dt = state.timestepping.dt
        beta_ = Dt*state.timestepping.alpha
        cp = state.parameters.cp
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vu_broken = FunctionSpace(state.mesh, BrokenElement(Vu.ufl_element()))
        Vtheta = state.spaces("HDiv_v")
        Vrho = state.spaces("DG")

        # Store time-stepping coefficients as UFL Constants
        dt = Constant(Dt)
        beta = Constant(beta_)
        beta_cp = Constant(beta_ * cp)

        h_deg = state.horizontal_degree
        v_deg = state.vertical_degree
        Vtrace = FunctionSpace(state.mesh, "HDiv Trace", degree=(h_deg, v_deg))

        # Split up the rhs vector (symbolically)
        u_in, rho_in, theta_in = split(state.xrhs)

        # Build the function space for "broken" u, rho, and pressure trace
        M = MixedFunctionSpace((Vu_broken, Vrho, Vtrace))
        w, phi, dl = TestFunctions(M)
        u, rho, l0 = TrialFunctions(M)

        n = FacetNormal(state.mesh)

        # Get background fields
        thetabar = state.fields("thetabar")
        rhobar = state.fields("rhobar")
        pibar = thermodynamics.pi(state.parameters, rhobar, thetabar)
        pibar_rho = thermodynamics.pi_rho(state.parameters, rhobar, thetabar)
        pibar_theta = thermodynamics.pi_theta(state.parameters, rhobar, thetabar)

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        theta = -dot(k, u)*dot(k, grad(thetabar))*beta + theta_in

        # Only include theta' (rather than pi') in the vertical
        # component of the gradient

        # The pi prime term (here, bars are for mean and no bars are
        # for linear perturbations)
        pi = pibar_theta*theta + pibar_rho*rho

        # Vertical projection
        def V(u):
            return k*inner(u, k)

        # Specify degree for some terms as estimated degree is too large
        dxp = dx(degree=(self.quadrature_degree))
        dS_vp = dS_v(degree=(self.quadrature_degree))
        dS_hp = dS_h(degree=(self.quadrature_degree))
        ds_vp = ds_v(degree=(self.quadrature_degree))
        ds_tbp = (ds_t(degree=(self.quadrature_degree))
                  + ds_b(degree=(self.quadrature_degree)))

        # Add effect of density of water upon theta
        if self.moisture is not None:
            water_t = Function(Vtheta).assign(0.0)
            for water in self.moisture:
                water_t += self.state.fields(water)
            theta_w = theta / (1 + water_t)
            thetabar_w = thetabar / (1 + water_t)
        else:
            theta_w = theta
            thetabar_w = thetabar

        _l0 = TrialFunction(Vtrace)
        _dl = TestFunction(Vtrace)
        a_tr = _dl('+')*_l0('+')*(dS_vp + dS_hp) + _dl*_l0*ds_vp + _dl*_l0*ds_tbp

        def L_tr(f):
            return _dl('+')*avg(f)*(dS_vp + dS_hp) + _dl*f*ds_vp + _dl*f*ds_tbp

        cg_ilu_parameters = {'ksp_type': 'cg',
                             'pc_type': 'bjacobi',
                             'sub_pc_type': 'ilu'}

        # Project field averages into functions on the trace space
        rhobar_avg = Function(Vtrace)
        pibar_avg = Function(Vtrace)

        rho_avg_prb = LinearVariationalProblem(a_tr, L_tr(rhobar), rhobar_avg)
        pi_avg_prb = LinearVariationalProblem(a_tr, L_tr(pibar), pibar_avg)

        rho_avg_solver = LinearVariationalSolver(rho_avg_prb,
                                                 solver_parameters=cg_ilu_parameters,
                                                 options_prefix='rhobar_avg_solver')
        pi_avg_solver = LinearVariationalSolver(pi_avg_prb,
                                                solver_parameters=cg_ilu_parameters,
                                                options_prefix='pibar_avg_solver')

        with timed_region("Gusto:HybridProjectRhobar"):
            rho_avg_solver.solve()

        with timed_region("Gusto:HybridProjectPibar"):
            pi_avg_solver.solve()

        # "broken" u, rho, and trace system
        # NOTE: no ds_v integrals since equations are defined on
        # a periodic (or sphere) base mesh.
        eqn = (
            # momentum equation
            inner(w, (state.h_project(u) - u_in))*dx
            - beta_cp*div(theta_w*V(w))*pibar*dxp
            # following does nothing but is preserved in the comments
            # to remind us why (because V(w) is purely vertical).
            # + beta_cp*jump(theta_w*V(w), n=n)*pibar_avg('+')*dS_vp
            + beta_cp*jump(theta_w*V(w), n=n)*pibar_avg('+')*dS_hp
            + beta_cp*dot(theta_w*V(w), n)*pibar_avg*ds_tbp
            - beta_cp*div(thetabar_w*w)*pi*dxp
            # trace terms appearing after integrating momentum equation
            + beta_cp*jump(thetabar_w*w, n=n)*l0('+')*(dS_vp + dS_hp)
            + beta_cp*dot(thetabar_w*w, n)*l0*(ds_tbp + ds_vp)
            # mass continuity equation
            + (phi*(rho - rho_in) - beta*inner(grad(phi), u)*rhobar)*dx
            + beta*jump(phi*u, n=n)*rhobar_avg('+')*(dS_v + dS_h)
            # term added because u.n=0 is enforced weakly via the traces
            + beta*phi*dot(u, n)*rhobar_avg*(ds_tb + ds_v)
            # constraint equation to enforce continuity of the velocity
            # through the interior facets and weakly impose the no-slip
            # condition
            + dl('+')*jump(u, n=n)*(dS_vp + dS_hp)
            + dl*dot(u, n)*(ds_tbp + ds_vp)
        )

        # contribution of the sponge term
        if mu is not None:
            eqn += dt*mu*inner(w, k)*inner(u, k)*dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Function for the hybridized solutions
        self.urhol0 = Function(M)

        hybridized_prb = LinearVariationalProblem(aeqn, Leqn, self.urhol0)
        hybridized_solver = LinearVariationalSolver(hybridized_prb,
                                                    solver_parameters=self.solver_parameters,
                                                    options_prefix='ImplicitSolver')
        self.hybridized_solver = hybridized_solver

        # Project broken u into the HDiv space using facet averaging.
        # Weight function counting the dofs of the HDiv element:
        shapes = {"i": Vu.finat_element.space_dimension(),
                  "j": np.prod(Vu.shape, dtype=int)}
        weight_kernel = """
        for (int i=0; i<{i}; ++i)
            for (int j=0; j<{j}; ++j)
                w[i*{j} + j] += 1.0;
        """.format(**shapes)

        self._weight = Function(Vu)
        par_loop(weight_kernel, dx, {"w": (self._weight, INC)})

        # Averaging kernel
        self._average_kernel = """
        for (int i=0; i<{i}; ++i)
            for (int j=0; j<{j}; ++j)
                vec_out[i*{j} + j] += vec_in[i*{j} + j]/w[i*{j} + j];
        """.format(**shapes)

        # HDiv-conforming velocity
        self.u_hdiv = Function(Vu)

        # Reconstruction of theta
        theta = TrialFunction(Vtheta)
        gamma = TestFunction(Vtheta)

        self.theta = Function(Vtheta)
        theta_eqn = gamma*(theta - theta_in
                           + dot(k, self.u_hdiv)*dot(k, grad(thetabar))*beta)*dx

        theta_problem = LinearVariationalProblem(lhs(theta_eqn), rhs(theta_eqn), self.theta)
        self.theta_solver = LinearVariationalSolver(theta_problem,
                                                    solver_parameters=cg_ilu_parameters,
                                                    options_prefix='thetabacksubstitution')

        # Store boundary conditions for the div-conforming velocity to apply
        # post-solve
        self.bcs = self.state.bcs

    @timed_function("Gusto:LinearSolve")
    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        # Solve the hybridized system
        self.hybridized_solver.solve()

        broken_u, rho1, _ = self.urhol0.split()
        u1 = self.u_hdiv

        # Project broken_u into the HDiv space
        u1.assign(0.0)

        with timed_region("Gusto:HybridProjectHDiv"):
            par_loop(self._average_kernel, dx,
                     {"w": (self._weight, READ),
                      "vec_in": (broken_u, READ),
                      "vec_out": (u1, INC)})

        # Reapply bcs to ensure they are satisfied
        for bc in self.bcs:
            bc.apply(u1)

        # Copy back into u and rho cpts of dy
        u, rho, theta = self.state.dy.split()
        u.assign(u1)
        rho.assign(rho1)

        # Reconstruct theta
        with timed_region("Gusto:ThetaRecon"):
            self.theta_solver.solve()

        # Copy into theta cpt of dy
        theta.assign(self.theta)


class IncompressibleSolver(TimesteppingSolver):
    """Timestepping linear solver object for the incompressible
    Boussinesq equations with prognostic variables u, p, b.

    This solver follows the following strategy:
    (1) Analytically eliminate b (introduces error near topography)
    (2) Solve resulting system for (u,p) using a hybrid-mixed method
    (3) Reconstruct b

    :arg state: a :class:`.State` object containing everything else.
    :arg solver_parameters: (optional) Solver parameters.
    :arg overwrite_solver_parameters: boolean, if True use only the
         solver_parameters that have been passed in, if False then
         update the default solver parameters with the solver_parameters
         passed in.
    """

    solver_parameters = {
        'ksp_type': 'preonly',
        'mat_type': 'matfree',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {'ksp_type': 'cg',
                          'pc_type': 'gamg',
                          'ksp_rtol': 1e-8,
                          'mg_levels': {'ksp_type': 'chebyshev',
                                        'ksp_max_it': 2,
                                        'pc_type': 'bjacobi',
                                        'sub_pc_type': 'ilu'}}
    }

    def __init__(self, state, solver_parameters=None,
                 overwrite_solver_parameters=False):
        super().__init__(state, solver_parameters, overwrite_solver_parameters)

    @timed_function("Gusto:SolverSetup")
    def _setup_solver(self):
        state = self.state      # just cutting down line length a bit
        Dt = state.timestepping.dt
        beta_ = Dt*state.timestepping.alpha
        mu = state.mu
        Vu = state.spaces("HDiv")
        Vb = state.spaces("HDiv_v")
        Vp = state.spaces("DG")

        # Store time-stepping coefficients as UFL Constants
        dt = Constant(Dt)
        beta = Constant(beta_)

        # Split up the rhs vector (symbolically)
        u_in, p_in, b_in = split(state.xrhs)

        # Build the reduced function space for u,p
        M = MixedFunctionSpace((Vu, Vp))
        w, phi = TestFunctions(M)
        u, p = TrialFunctions(M)

        # Get background fields
        bbar = state.fields("bbar")

        # Analytical (approximate) elimination of theta
        k = state.k             # Upward pointing unit vector
        b = -dot(k, u)*dot(k, grad(bbar))*beta + b_in

        # vertical projection
        def V(u):
            return k*inner(u, k)

        eqn = (
            inner(w, (u - u_in))*dx
            - beta*div(w)*p*dx
            - beta*inner(w, k)*b*dx
            + phi*div(u)*dx
        )

        if mu is not None:
            eqn += dt*mu*inner(w, k)*inner(u, k)*dx
        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u p solver
        self.up = Function(M)

        # Boundary conditions (assumes extruded mesh)
        bcs = None if len(self.state.bcs) == 0 else self.state.bcs

        # Solver for u, p
        up_problem = LinearVariationalProblem(aeqn, Leqn, self.up, bcs=bcs)

        # Provide callback for the nullspace of the trace system
        def trace_nullsp(T):
            return VectorSpaceBasis(constant=True)

        appctx = {"trace_nullspace": trace_nullsp}
        self.up_solver = LinearVariationalSolver(up_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 appctx=appctx)

        # Reconstruction of b
        b = TrialFunction(Vb)
        gamma = TestFunction(Vb)

        u, p = self.up.split()
        self.b = Function(Vb)

        b_eqn = gamma*(b - b_in
                       + dot(k, u)*dot(k, grad(bbar))*beta)*dx

        b_problem = LinearVariationalProblem(lhs(b_eqn),
                                             rhs(b_eqn),
                                             self.b)
        self.b_solver = LinearVariationalSolver(b_problem)

    @timed_function("Gusto:LinearSolve")
    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        with timed_region("Gusto:VelocityPressureSolve"):
            self.up_solver.solve()

        u1, p1 = self.up.split()
        u, p, b = self.state.dy.split()
        u.assign(u1)
        p.assign(p1)

        with timed_region("Gusto:BuoyancyRecon"):
            self.b_solver.solve()

        b.assign(self.b)


class ShallowWaterSolver(TimesteppingSolver):
    """
    Timestepping linear solver object for the nonlinear shallow water
    equations with prognostic variables u and D. The linearized system
    is solved using a hybridized-mixed method.
    """

    solver_parameters = {
        'ksp_type': 'preonly',
        'mat_type': 'matfree',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {'ksp_type': 'cg',
                          'pc_type': 'gamg',
                          'ksp_rtol': 1e-8,
                          'mg_levels': {'ksp_type': 'chebyshev',
                                        'ksp_max_it': 2,
                                        'pc_type': 'bjacobi',
                                        'sub_pc_type': 'ilu'}}
    }

    @timed_function("Gusto:SolverSetup")
    def _setup_solver(self):
        state = self.state
        H_ = state.parameters.H
        g_ = state.parameters.g
        beta_ = state.timestepping.dt*state.timestepping.alpha

        # Store time-stepping coefficients as UFL Constants
        beta = Constant(beta_)
        H = Constant(H_)
        g = Constant(g_)

        # Split up the rhs vector (symbolically)
        u_in, D_in = split(state.xrhs)

        W = state.W
        w, phi = TestFunctions(W)
        u, D = TrialFunctions(W)

        eqn = (
            inner(w, u) - beta*g*div(w)*D
            - inner(w, u_in)
            + phi*D + beta*H*phi*div(u)
            - phi*D_in
        )*dx

        aeqn = lhs(eqn)
        Leqn = rhs(eqn)

        # Place to put result of u rho solver
        self.uD = Function(W)

        # Solver for u, D
        bcs = None if len(self.state.bcs) == 0 else self.state.bcs
        uD_problem = LinearVariationalProblem(
            aeqn, Leqn, self.state.dy, bcs=bcs)

        self.uD_solver = LinearVariationalSolver(uD_problem,
                                                 solver_parameters=self.solver_parameters,
                                                 options_prefix='SWimplicit')

    @timed_function("Gusto:LinearSolve")
    def solve(self):
        """
        Apply the solver with rhs state.xrhs and result state.dy.
        """

        self.uD_solver.solve()
