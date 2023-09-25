""" Code used to process the lock-exchange simulation using FEniCS"""

import numpy as np
from fenics import (MPI, CellDiameter, Constant, File, FiniteElement, Function,
                    FunctionSpace, HDF5File, MixedElement,
                    NonlinearVariationalProblem, NonlinearVariationalSolver,
                    Point, RectangleMesh, TestFunctions, TrialFunction,
                    VectorElement, XDMFFile, action, derivative, div, dot, dx,
                    grad, info, inner, list_krylov_solver_preconditioners,
                    nabla_grad, parameters, set_log_level, split, sqrt)

from utils import (InitialConditionsCD, InitialConditionsNS,
                   Lock_Exchange_boundary_condition,
                   export_concentration_snapshot)

comm = MPI.comm_world
rank = comm.Get_rank()
psize = comm.Get_size()
print("psize = ", psize)

set_log_level(40)
if rank == 0:
    set_log_level(20)

# =====================================================================|
#                                                                      |
#                                                                      |
#                               HEADER                                 |
#                                                                      |
#                                                                      |
# =====================================================================|

# General flags
OUTPUT_TYPE = "XDMF"  # VTK, XDMF, HDF5, OUT
EXPORT_INTERVAL = 1
SUPG = True
DIR = "Results/"
ITER_SOLVER_NS = True  # True = GMRES / False = LU
SUPS = True  # True = SUPS / False = Taylor-Hood
LSIC = True


# Parameters
GR = sqrt(5e6)  # Grashof
SC = 1.00  # Schmidt
F_FORCE = Constant((0.0, -1))  # body forces
SED = Constant((0.0, -1 * 0.00))  # sedimentation vel (SHOULD NOT BE CHANGED)
NU = Constant(1 / GR)
NU2 = Constant(1 / (GR * SC))


# Time parameters
T = 0.0
DT = 0.01
TOTAL_T = 5.00


# Domain + Mesh
HEIGHT = 2.05
WIDTH = 18.0
NX = 700
NY = 100


# =================      FENICS PREPROCESSING     ======================
comm = MPI.comm_world
rank = comm.Get_rank()

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# =====================================================================|
#                                                                      |
#                                                                      |
#                           PROCESSING                                 |
#                                                                      |
#                                                                      |
# =====================================================================|


mesh = RectangleMesh(Point(0.0, 0.0), Point(WIDTH, HEIGHT), NX, NY)


# Problem setup
if SUPS is False:
    P1v = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    parameters["form_compiler"]["quadrature_degree"] = 3
else:
    P1v = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    parameters["form_compiler"]["quadrature_degree"] = 1
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([P1v, P1, P1])
W = FunctionSpace(mesh, TH)
vp = TrialFunction(W)
(v, p, c) = split(vp)
(w, q, qc) = TestFunctions(W)
vp_ = Function(W)
v_function, p_function, c_function = split(vp_)
u_init = InitialConditionsNS()
vp_.interpolate(u_init)
c_function = vp_.split(deepcopy=True)[2]
c_function.interpolate(InitialConditionsCD())
out_c = File(DIR + "initialconcentration.pvd", "compressed")
out_c << (c_function, T)
vn = Function(W.sub(0).collapse())
cn = Function(W.sub(2).collapse())
cn = c_function
bcs = Lock_Exchange_boundary_condition(W, WIDTH, HEIGHT)
dtt = Constant(DT)
h = CellDiameter(mesh)
vnorm = sqrt(dot(v_function, v_function))
F = (
    inner(v - vn, w) / dtt
    + inner(dot(v_function, nabla_grad(v)), w)
    + NU * inner(grad(w), grad(v))
    - inner(p, div(w))
    + inner(q, div(v))
    - c_function * inner(F_FORCE, w)
) * dx + (
    inner(c - cn, qc) / dtt
    + inner(dot((v_function + SED), nabla_grad(c)), qc)
    + NU2 * inner(grad(qc), grad(c))
) * dx  # - (c - cn)/(DT*0.02)*qc*ds(1)
R = (
    (1.0 / dtt) * (v_function - vn)
    + dot(v_function, nabla_grad(v))
    - NU * div(grad(v))
    + grad(p)
    - c_function * F_FORCE
)
R2 = (
    (1.0 / dtt) * (c_function - cn)
    + dot((v_function + SED), nabla_grad(c))
    - NU2 * div(grad(c))
)
tau = ((2.0 / dtt) ** 2 + (2.0 * vnorm / h) ** 2 + 9.0 * (4.0 * NU / (h * 2)) ** 2) ** (
    -0.5
)

# Stabilization
vnorm = sqrt(dot(v_function, v_function))
tau_lsic = (vnorm * h) / 2.0
F_lsic = tau_lsic * inner(div(v), div(w)) * dx
F_supg = tau * inner(R, dot(v_function, nabla_grad(w))) * dx
F_supg2 = tau * inner(R2, dot((v_function + SED), grad(qc))) * dx
F_pspg = tau * inner(R, grad(q)) * dx

if SUPG is True:
    F += F_supg + F_supg2

if SUPS is True:
    F += F_pspg

if LSIC is True:
    F += F_lsic


F1 = action(F, vp_)
J = derivative(F1, vp_, vp)

problem = NonlinearVariationalProblem(F1, vp_, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["absolute_tolerance"] = 1e-6
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["convergence_criterion"] = "incremental"
prm["newton_solver"]["maximum_iterations"] = 20
prm["newton_solver"]["relaxation_parameter"] = 0.9
prm["newton_solver"]["linear_solver"] = "direct"
prm["newton_solver"]["error_on_nonconvergence"] = False

if ITER_SOLVER_NS is True:
    prm["newton_solver"]["linear_solver"] = "gmres"
    prm["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1e-5
    prm["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1e-5
    prm["newton_solver"]["krylov_solver"]["maximum_iterations"] = 75000
    prm["newton_solver"]["preconditioner"] = "default"
    info(prm, True)  # uncomment if any setup checking is needed
    list_krylov_solver_preconditioners()  # uncomment to verify krylov preconditioners available

xyz = W.sub(2).collapse().tabulate_dof_coordinates()
np.savetxt("xyz.out", np.around(xyz, 2), delimiter=",")

if OUTPUT_TYPE == "VTK":
    FILE_U = DIR + "velocity.pvd"
    FILE_P = DIR + "pressure.pvd"
    FILE_C = DIR + "concentration.pvd"
    out_u = File(FILE_U, "compressed")
    out_p = File(FILE_P, "compressed")
    out_c = File(FILE_C, "compressed")

    def export_output(v_export, p_export, c_export, t_export, cont):
        """Exports desired output file"""
        out_u << (v_export, t_export)
        out_p << (p_export, t_export)
        out_c << (c_export, t_export)

elif OUTPUT_TYPE == "XDMF":
    def export_output(v_export, p_export, c_export, t_export, cont):
        """Exports desired output file"""
        FILE_U = DIR + f"velocity_{cont}.xdmf"
        FILE_P = DIR + f"pressure_{cont}.xdmf"
        FILE_C = DIR + f"concentration_{cont}.xdmf"
        out_u = XDMFFile(FILE_U)
        out_p = XDMFFile(FILE_P)
        out_c = XDMFFile(FILE_C)

        out_u.write_checkpoint(v_export, "velocity", t_export)
        out_p.write_checkpoint(p_export, "velocity", t_export)
        out_c.write_checkpoint(c_export, "velocity", t_export)


elif OUTPUT_TYPE == "HDF5":
    if psize == 1:
        FILENAME = DIR + "/Snapshots/mesh.h5"
        snapshot = HDF5File(mesh.mpi_comm(), FILENAME, "w")
        snapshot.write(mesh, "u")

    def export_output(v_export, p_export, c_export, t_export, cont):
        """Exports desired output file"""
        filename = DIR + "/Snapshots/u" + str(cont) + ".h5"
        snapshot = HDF5File(mesh.mpi_comm(), filename, "w")
        snapshot.write(v_export, "u")
        filename = DIR + "/Snapshots/p" + str(cont) + ".h5"
        snapshot = HDF5File(mesh.mpi_comm(), filename, "w")
        snapshot.write(p_export, "u")
        filename = DIR + "/Snapshots/c" + str(cont) + ".h5"
        snapshot = HDF5File(mesh.mpi_comm(), filename, "w")
        snapshot.write(c_export, "u")

elif OUTPUT_TYPE == "OUT":
    FILE_U = DIR + "velocity.out"
    FILE_P = DIR + "pressure.out"
    FILE_C = DIR + "concentration.out"

    def export_output(v_export, p_export, c_export, t_export, cont):
        """Exports desired output file"""
        np.savetxt(
            DIR + "Snapshots/velocity" + str(cont) + ".out",
            v_export.vector().get_local(),
            delimiter=",",
        )
        np.savetxt(
            DIR + "Snapshots/pressure" + str(cont) + ".out",
            p_export.vector().get_local(),
            delimiter=",",
        )
        np.savetxt(
            DIR + "Snapshots/concentration" + str(cont) + ".out",
            c_export.vector().get_local(),
            delimiter=",",
        )


if psize == 1:  # HDF5 mesh export only works for serial runs on FEniCS
    FILENAME = DIR + "/Snapshots/mesh.h5"
    snapshot = HDF5File(mesh.mpi_comm(), FILENAME, "w")
    snapshot.write(mesh, "u")

STEP = 0

while T < TOTAL_T:
    if rank == 0:
        print(f"\nt = {T:10.3e}\n")

    converged_flag, nIter = solver.solve()
    v_function, p_function, c_function = vp_.split(True)
    T = T + DT
    STEP += 1
    vn.assign(v_function)
    cn.assign(c_function)
    if STEP % EXPORT_INTERVAL == 0:
        export_output(v_function, p_function, c_function, T, STEP)
        #export_concentration_snapshot(DIR, mesh, c_function, STEP)