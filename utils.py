""" Utility functions for the ns_lock_exchange code. """

from fenics import (
    HDF5File,
    MPI,
    UserExpression,
    SubDomain,
    Constant,
    DirichletBC,
    near, 
)
import random


def export_concentration_snapshot(dir_output, mesh_domain, c_export, cont):
    """
    Exports concentration snapshots in HDF5 regardless of selected
    OUTPUT_TYPE in main
    """
    filename = dir_output + "Snapshots/c" + str(cont) + ".h5"
    snapshot = HDF5File(mesh_domain.mpi_comm(), filename, "w")
    snapshot.write(c_export, "u")


class InitialConditionsNS(UserExpression):
    """
    Initial conditions for the velocity field (NS equation)
    """

    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        """ Initialize random values for velocity """
        values[0] = 0.0 + 0.002 * (0.5 - random.random())
        values[1] = 0.0

    def value_shape(self):
        """ Dimensions for output """
        return (4,)


class InitialConditionsCD(UserExpression):
    """
    Initial conditions for the concentration field (transport equation)
    """

    def eval_cell(self, value, x, ufc_cell):
        """ Values for IC on concentration field """
        if x[0] < 1.05 and x[1] < 2.00:
            value[0] = 1.0
        elif x[1] >= 2.0 and x[0] <= 1.05 and x[1] < 2.03 and x[0] > 0.02:
            value[0] = 0.5
        elif x[0] >= 1.05 and x[0] < 1.08 and x[1] < 2.03:
            value[0] = 0.5
        else:
            value[0] = 0.0


def Lock_Exchange_boundary_condition(W, width, height):
    """
    Boundary conditions for the 2D tank
    """

    class Walls(SubDomain):
        """ Bottom and Top BC """
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[1], 0.0) or near(x[1], height))

    class SlipWallsLeft(SubDomain):
        """ Left BC """
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[0], 0.0))

    class SlipWallsRight(SubDomain):
        """ Right BC """
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[0], width))

    # Dirichlet value
    g_1 = Constant(0.0)
    g_2 = Constant((0.0, 0.0))
    # conditions
    bc_1 = DirichletBC(W.sub(0).sub(0), g_1, SlipWallsLeft()) #fixed u on the left
    bc_2 = DirichletBC(W.sub(0).sub(0), g_1, SlipWallsRight()) # fixed u on the right
    bc_3 = DirichletBC(W.sub(0), g_2, Walls()) # no-slip top and bottom
    bc_4 = DirichletBC(W.sub(2), g_1, SlipWallsRight()) # fixed p on the right
    bcs = [bc_1, bc_2, bc_3, bc_4]

    return bcs