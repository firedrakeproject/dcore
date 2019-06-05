from firedrake import (TrialFunctions, Function, TestFunctions, dx,
                       DirichletBC, LinearVariationalProblem,
                       LinearVariationalSolver, Constant)
from gusto.configuration import logger, DEBUG
from gusto.form_manipulation_labelling import (drop, time_derivative,
                                               subject, name, index,
                                               advection, has_labels,
                                               replace_labelled)

__all__ = ["Forcing"]


class Forcing(object):
    """
    Class to compute forcing terms for split forcing and advection schemes.

    :arg equation: a :class:`PrognosticEquation` object containing the
    form for the equation to be solved
    :arg dt: timestep
    :arg alpha: off-centering parameter
    """

    def __init__(self, equation, dt, alpha):

        # this is the function that the forcing term is applied to
        W = equation.function_space
        self.x0 = Function(W)
        self.xF = Function(W)

        eqn = equation.residual.label_map(has_labels(advection), drop)
        assert len(eqn) > 1
        self._build_forcing_solver(W, eqn, dt, alpha)

    def _build_forcing_solver(self, W, equation, dt, alpha):
        """
        Builds the forcing solvers for the explicit and the implicit steps
        """
        trials = TrialFunctions(W)

        a_explicit = equation.label_map(has_labels(time_derivative),
                                        replace_labelled(trials, subject),
                                        drop)

        L_explicit = equation.label_map(
            lambda t: t.get(name) == "incompressible",
            drop)
        L_explicit = Constant(-(1-alpha)*dt)*L_explicit.label_map(
            has_labels(time_derivative),
            drop,
            replace_labelled(self.x0.split(), subject))

        a_implicit = a_explicit
        for t in equation:
            if t.get(name) == "incompressible":
                idx = t.get(index)
                test = TestFunctions(W)[idx]
                a_implicit += trials[idx]*test*dx

        L_implicit = Constant(-alpha*dt)*equation.label_map(
            has_labels(time_derivative),
            drop,
            replace_labelled(self.x0.split(), subject))

        Vu = W.split()[0]
        bcs = None if len(self.state.bcs) == 0 else self.state.bcs

        explicit_forcing_problem = LinearVariationalProblem(
            a_explicit.form, L_explicit.form, self.xF, bcs=bcs
        )

        implicit_forcing_problem = LinearVariationalProblem(
            a_implicit.form, L_implicit.form, self.xF, bcs=bcs
        )

        solver_parameters = {}
        if logger.isEnabledFor(DEBUG):
            solver_parameters["ksp_monitor_true_residual"] = True

        self.solvers = {}
        self.solvers["explicit"] = LinearVariationalSolver(
            explicit_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix="ExplicitForcingSolver"
        )
        self.solvers["implicit"] = LinearVariationalSolver(
            implicit_forcing_problem,
            solver_parameters=solver_parameters,
            options_prefix="ImplicitForcingSolver"
        )

    def apply(self, x_in, x_nl, x_out, label):
        """
        Function takes x_in as input, computes F(x_nl) and returns
        x_out = x_in + F(x_nl)
        as output.

        :arg x_in: :class:`.Function` object
        :arg x_nl: :class:`.Function` object
        :arg x_out: :class:`.Function` object
        """

        self.x0.assign(x_nl)
        x_out.assign(x_in)

        self.solvers[label].solve()
        x = x_out
        x += self.xF
