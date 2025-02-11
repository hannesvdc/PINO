from dolfin import *
import numpy as np
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

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
    def __init__(self, mesh, correlation_length=0.12, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.correlation_length = correlation_length
        self.kernel = lambda y, yp: np.exp(-0.5 * (y - yp) ** 2 / correlation_length ** 2)
        self.boundary_points, self.forces = self._generate_correlated_forces()

    def _generate_correlated_forces(self):
        # Extract boundary points on x = 1
        boundary_points = sorted([f.midpoint()[1] for f in facets(self.mesh) if near(f.midpoint()[0], 1.0)])
        n = len(boundary_points)

        # Construct covariance matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(boundary_points[i], boundary_points[j])

        # Cholesky decomposition to sample correlated values
        L = cholesky(K + 1e-6 * np.eye(n), lower=True)  # Add small noise for numerical stability
        uncorrelated_samples = np.random.normal(0, 0.1, (n, 2))  # Two components (x and y)
        correlated_samples = L @ uncorrelated_samples

        # Store the forces mapped to boundary points
        forces = {boundary_points[i]: correlated_samples[i, :] for i in range(n)}
        return boundary_points, forces

    def eval(self, values, x):
        if near(x[0], 1.0):
            # Find the closest boundary point in the precomputed list
            closest_y = min(self.boundary_points, key=lambda yp: abs(yp - x[1]))
            values[:] = self.forces[closest_y]
        else:
            values[:] = (0.0, 0.0)  # No force elsewhere

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
