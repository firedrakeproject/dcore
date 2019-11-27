from firedrake import dx, BrokenElement, Function, FunctionSpace
from firedrake.parloops import par_loop, READ, WRITE, INC
from firedrake.slope_limiter.vertex_based_limiter import VertexBasedLimiter

__all__ = ["ThetaLimiter", "NoLimiter"]


class ThetaLimiter(object):
    """
    A vertex based limiter for fields in the DG1xCG2 space,
    i.e. temperature variables. This acts like the vertex-based
    limiter implemented in Firedrake, but in addition corrects
    the central nodes to prevent new maxima or minima forming.
    """

    def __init__(self, space):
        """
        Initialise limiter
        :param space: the space in which theta lies.
        It should be the DG1xCG2 space.
        """

        self.Vt = FunctionSpace(space.mesh(), BrokenElement(space.ufl_element()))
        # check this is the right space, currently working for 2D and 3D extruded meshes
        # check that horizontal degree is 1 and vertical degree is 2
        if self.Vt.ufl_element().degree()[0] != 1 or \
           self.Vt.ufl_element().degree()[1] != 2:
            raise ValueError('This is not the right limiter for this space.')

        if not self.Vt.extruded:
            raise ValueError('This is not the right limiter for this space.')
        if self.Vt.mesh().topological_dimension() == 2:
            # check that continuity of the spaces is correct
            # this will fail if the space does not use broken elements
            if self.Vt.ufl_element()._element.sobolev_space()[0].name != 'L2' or \
               self.Vt.ufl_element()._element.sobolev_space()[1].name != 'H1':
                raise ValueError('This is not the right limiter for this space.')
        elif self.Vt.mesh().topological_dimension() == 3:
            # check that continuity of the spaces is correct
            # this will fail if the space does not use broken elements
            if self.Vt.ufl_element()._element.sobolev_space()[0].name != 'L2' or \
               self.Vt.ufl_element()._element.sobolev_space()[2].name != 'H1':
                raise ValueError('This is not the right limiter for this space.')
        else:
            raise ValueError('This is not the right limiter for this space.')

        self.DG1 = FunctionSpace(self.Vt.mesh(), 'DG', 1)  # space with only vertex DOFs
        self.vertex_limiter = VertexBasedLimiter(self.DG1)
        self.theta_hat = Function(self.DG1)  # theta function with only vertex DOFs
        self.theta_old = Function(self.Vt)
        self.w = Function(self.Vt)
        self.result = Function(self.Vt)

        shapes = {'nDOFs': self.Vt.finat_element.space_dimension(),
                  'nDOFs_base': int(self.Vt.finat_element.space_dimension() / 3)}
        averager_domain = "{{[i]: 0 <= i < {nDOFs}}}".format(**shapes)
        theta_domain = "{{[i,j]: 0 <= i < {nDOFs_base} and 0 <= j < 2}}".format(**shapes)

        average_instructions = ("""
                                for i
                                    vo[i] = vo[i] + v[i] / w[i]
                                end
                                """)

        weight_instructions = ("""
                               for i
                                  w[i] = w[i] + 1.0
                               end
                               """)

        copy_into_DG1_instrs = ("""
                                for i
                                    for j
                                        theta_hat[i*2+j] = theta[i*3+j]
                                    end
                                end
                                """)

        copy_from_DG1_instrs = ("""
                                <float64> max_value = 0.0
                                <float64> min_value = 0.0
                                for i
                                    for j
                                        theta[i*3+j] = theta_hat[i*2+j]
                                    end
                                    max_value = fmax(theta_hat[i*2], theta_hat[i*2+1])
                                    min_value = fmin(theta_hat[i*2], theta_hat[i*2+1])
                                    if theta_old[i*3+2] > max_value
                                        theta[i*3+2] = 0.5 * (theta_hat[i*2] + theta_hat[i*2+1])
                                    elif theta_old[i*3+2] < min_value
                                        theta[i*3+2] = 0.5 * (theta_hat[i*2] + theta_hat[i*2+1])
                                    else
                                        theta[i*3+2] = theta_old[i*3+2]
                                    end
                                end
                                """)

        self._average_kernel = (averager_domain, average_instructions)
        _weight_kernel = (averager_domain, weight_instructions)
        self._copy_into_DG1_kernel = (theta_domain, copy_into_DG1_instrs)
        self._copy_from_DG1_kernel = (theta_domain, copy_from_DG1_instrs)

        par_loop(_weight_kernel, dx, {"w": (self.w, INC)}, is_loopy_kernel=True)

    def copy_vertex_values(self, field):
        """
        Copies the vertex values from temperature space to
        DG1 space which only has vertices.
        """
        par_loop(self._copy_into_DG1_kernel, dx,
                 {"theta": (field, READ),
                  "theta_hat": (self.theta_hat, WRITE)},
                 is_loopy_kernel=True)

    def copy_vertex_values_back(self, field):
        """
        Copies the vertex values back from the DG1 space to
        the original temperature space, and checks that the
        midpoint values are within the minimum and maximum
        at the adjacent vertices.
        If outside of the minimum and maximum, correct the values
        to be the average.
        """
        par_loop(self._copy_from_DG1_kernel, dx,
                 {"theta": (field, WRITE),
                  "theta_hat": (self.theta_hat, READ),
                  "theta_old": (self.theta_old, READ)},
                 is_loopy_kernel=True)

    def apply(self, field):
        """
        The application of the limiter to the theta-space field.
        """
        assert field.function_space() == self.Vt, \
            'Given field does not belong to this objects function space'

        self.theta_old.assign(field)
        self.copy_vertex_values(field)
        self.vertex_limiter.apply(self.theta_hat)
        self.copy_vertex_values_back(field)


class NoLimiter(object):
    """
    A blank limiter that does nothing.
    """

    def __init__(self):
        """
        Initialise the blank limiter.
        """
        pass

    def apply(self, field):
        """
        The application of the blank limiter.
        """
        pass
