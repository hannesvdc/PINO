import json
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==== Load forcing data from file ====
# Expected format: three columns (y, f1, f2)
# Example row: 0.35   1.2   -0.5
index = 1001
config_file = 'DataConfig.json'
config = json.load(open(config_file))
data_directory = config["Data Directory"]
forcing_data = np.load(data_directory + 'branch_data.npy').transpose()
y_vals = np.linspace(0.0, 1.0, 101)
f1_vals = forcing_data[index, 0:101]
f2_vals = forcing_data[index, 101:]
plt.plot(y_vals, f1_vals)
plt.plot(y_vals, f2_vals)
plt.show()

# Interpolants for the force components
f1_interp = interp1d(y_vals, f1_vals, kind='linear', bounds_error=False, fill_value=0.0)
f2_interp = interp1d(y_vals, f2_vals, kind='linear', bounds_error=False, fill_value=0.0)

# ==== Define the external force from file ====
class FileForce(UserExpression):
    def __init__(self, f1_func, f2_func, **kwargs):
        super().__init__(**kwargs)
        self.f1_func = f1_func
        self.f2_func = f2_func

    def eval(self, values, x):
        if near(x[0], 1.0):  # Right boundary
            y = x[1]
            values[0] = -float(self.f1_func(y))
            values[1] = -float(self.f2_func(y))
        else:
            values[0] = 0.0
            values[1] = 0.0

    def value_shape(self):
        return (2,)

# Create force expression
T = FileForce(f1_interp, f2_interp, degree=2)

# ==== FEM Setup ====
N = 100
mesh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(mesh, "CG", 1) 
u = TrialFunction(V)
v = TestFunction(V)

# Boundary conditions: Clamped at x=0
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)
bc = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)

# Define right boundary markers
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)
right_boundary = RightBoundary()
right_boundary.mark(boundaries, 1)
ds_right = Measure("ds", domain=mesh, subdomain_data=boundaries)

# ==== Material Parameters ====
E = Constant(300.0 * 1.e6)
nu = Constant(0.3)
mu  = E / (2.0*(1.0 + nu))
lambda_ = E*nu / (1.0 - nu**2)
# --- kinematics --------------------------------------------------------------
def eps(u):
    """Small-strain tensor ε(u) = sym(∇u)."""
    return sym(grad(u))

# --- constitutive law --------------------------------------------------------
def sigma(u):
    """Plane-stress C : ε(u) written explicitly."""
    exx = u[0].dx(0)
    eyy = u[1].dx(1)
    gxy = u[0].dx(1) + u[1].dx(0)          # γxy  (engineering shear strain)

    C11 =  E/(1.0 - nu**2)                 #  = C22
    C12 =  nu*C11
    C66 =  E*(1.0 - nu)/(2.0*(1.0 + nu))   #  = G  for plane stress

    return as_tensor([[C11*exx + C12*eyy, C66*gxy],
                      [C66*gxy,           C12*exx + C11*eyy]])

# ==== Weak Form ====
a = inner(sigma(u), eps(v)) * dx          # ∫ σ(u) : ε(v)  dx
L = dot(T, v) * ds_right(1)

# Assemble and apply BCs
A = assemble(a)
b = assemble(L)
bc.apply(A, b)

# ==== Solve ====
u_sol = Function(V)
solve(A, u_sol.vector(), b, "gmres", "ilu")

# ==== Interpolate to Grid for Plotting ====
print('Plotting displacement field')
Nplot = 101
xx = np.linspace(0.0, 1.0, Nplot)
yy = np.linspace(0.0, 1.0, Nplot)
u_x_grid = np.empty((Nplot, Nplot))
v_x_grid = np.empty((Nplot, Nplot))

for j, y in enumerate(yy):
    for i, x in enumerate(xx):
        u_x_grid[j, i] = u_sol(Point(x, y))[0]   # x-component
        v_x_grid[j, i] = u_sol(Point(x, y))[1]

X_plot, Y_plot = np.meshgrid(np.linspace(0, 1, N+1), np.linspace(0, 1, N+1))
#u_interpolated = Function(V)
#u_interpolated.interpolate(u_sol)
#u_x = u_interpolated.split()[0]
#u_x_values = u_x.compute_vertex_values()
#u_x_grid = np.reshape(u_x_values, (N+1, N+1))

# ==== Plot ====
plt.pcolor(X_plot, Y_plot, u_x_grid, shading="auto", cmap='jet')
plt.colorbar()
plt.title("X-Displacement Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.figure()
plt.pcolor(X_plot, Y_plot, v_x_grid, shading="auto", cmap='jet')
plt.colorbar()
plt.title("Y-Displacement Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()