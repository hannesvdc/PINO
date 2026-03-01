from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

def fem( nu_val : float,
         y_points : np.ndarray,
         f_1 : np.ndarray, 
         f_2 : np.ndarray ):

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
    E = 1.0
    nu = Constant( nu_val )
    def stress(u, E, nu):
        return (E / (1 - nu**2)) * as_tensor([
            [u[0].dx(0) + nu * u[1].dx(1), (1 - nu)/2 * (u[0].dx(1) + u[1].dx(0))],
            [(1 - nu)/2 * (u[0].dx(1) + u[1].dx(0)), nu * u[0].dx(0) + u[1].dx(1)]
        ])

    # Define the random forcing on the right boundary
    class TractionY(UserExpression):
        def __init__(self, y_points, f1, f2, **kwargs):
            super().__init__(**kwargs)
            self.y_points = np.asarray(y_points)
            self.f1 = np.asarray(f1)
            self.f2 = np.asarray(f2)

        def eval(self, values, x):
            y = x[1]
            values[0] = np.interp(y, self.y_points, self.f1)
            values[1] = np.interp(y, self.y_points, self.f2)

        def value_shape(self):
            return (2,)
    T = TractionY( y_points, f_1, f_2 )

    # Define weak form
    a = inner(stress(u, E, nu), grad(v)) * dx
    L = dot(T, v) * ds_right(1)
    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)

    # Solve the Linear problem
    u_sol = Function(V)
    solve(A, u_sol.vector(), b, "gmres", "ilu")

    # Interpolate the solution on a rectangular grid
    print('Plotting displacement field')
    X_plot, Y_plot = np.meshgrid(np.linspace(0, 1, N+1), np.linspace(0, 1, N+1))
    points = np.c_[X_plot.ravel(), Y_plot.ravel()] # Flatten points for evaluation
    Ux = np.zeros(len(points))
    Uy = np.zeros(len(points))
    for i, (x, y) in enumerate(points):
        val = u_sol(Point(x, y))   # IMPORTANT: use Point
        Ux[i] = val[0]
        Uy[i] = val[1]

    # Reshape back to grid and return
    Ux_grid = Ux.reshape(X_plot.shape)
    Uy_grid = Uy.reshape(Y_plot.shape)
    return X_plot, Y_plot, Ux_grid, Uy_grid

    # Make a color plot
    plt.pcolor(X_plot, Y_plot, u_x_grid, shading="auto", cmap='jet')
    plt.colorbar()
    plt.title("X-Displacement Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
