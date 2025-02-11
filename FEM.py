from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from dolfin import plot

# Setup the Geometry, Trial, and Test functions
N = 100
mesh = UnitSquareMesh(N, N)
V = VectorFunctionSpace(mesh, "CG", 1) 
u = TrialFunction(V) # A 2-D trial function representing (u, v) from the paper
v = TestFunction(V)

# Define the boundary conditions
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)  # Clamped boundary on the left wall
bc = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)
right_boundary = RightBoundary()
right_boundary.mark(boundaries, 1)
ds_right = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Setup the Model parameters
E = Constant(410.0 * 1.e3)
nu = Constant(0.3)
def stress(u, E, nu):
    return (E / (1 - nu**2)) * as_tensor([
        [u[0].dx(0) + nu * u[1].dx(1), (1 - nu)/2 * (u[0].dx(1) + u[1].dx(0))],
        [(1 - nu)/2 * (u[0].dx(1) + u[1].dx(0)), nu * u[0].dx(0) + u[1].dx(1)]
    ])

# Define the random forcing on the right boundary
class RandomForce(UserExpression):
    def __init__(self, mesh, **kwargs):
        super().__init__(**kwargs)
        np.random.seed()
        self.forces = {}

        for f in facets(mesh):
            if near(f.midpoint()[0], 1.0):  # Only on the right boundary
                self.forces[f.index()] = (np.random.normal(0, 0.1), np.random.normal(0, 0.1))

    def eval(self, values, x):
        # Find the closest facet and apply the random force
        for f in facets(mesh):
            if near(f.midpoint()[0], 1.0):
                values[0], values[1] = self.forces.get(f.index(), (0.0, 0.0))
                return
        values[0], values[1] = (0.0, 0.0)  # Default zero force

    def value_shape(self):
        return (2,)
T = RandomForce(mesh)  # External traction force

# Define weak form
a = inner(stress(u, E, nu), grad(v)) * dx
L = dot(T, v) * ds_right(1)
A = assemble(a)
b = assemble(L)
bc.apply(A, b)
print("b vector:", b.get_local())

# Solve the Linear problem
u_sol = Function(V)
solve(A, u_sol.vector(), b, "gmres", "ilu")
print('u_sol', u_sol.vector().get_local())

# Interpolate the solution on a rectangular grid
print('Plotting displacement field')
X_plot, Y_plot = np.meshgrid(np.linspace(0, 1, N+1), np.linspace(0, 1, N+1))
u_interpolated = Function(V)
u_interpolated.interpolate(u_sol)
u_x = u_interpolated.split()[0]
u_x_values = u_x.compute_vertex_values()
u_x_grid = np.reshape(u_x_values, (N+1, N+1))

# Make a color plot
plt.pcolor(X_plot, Y_plot, u_x_grid, shading="auto")
plt.colorbar()
plt.title("X-Displacement Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
