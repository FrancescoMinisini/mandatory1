import numpy as np
import sympy as sp
import scipy.sparse as sparse

x, y = sp.symbols('x,y')

class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.

    """

    def __init__(self, L, ue,  N=None):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)
        self.N = None
        self.xij, self.yij = None, None
        self.h = None
        if isinstance(N, int):
            self.create_mesh(N)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = self.L/self.N
        x = np.linspace(0, self.L, self.N+1)
        y = np.linspace(0, self.L, self.N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')
        return self.xij, self.yij

    def D2(self):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.h**2
        return D

    def laplace(self):
        """Return vectorized Laplace operator"""
        D = self.D2()
        return (sparse.kron(D, sparse.eye(self.N+1)) + sparse.kron(sparse.eye(self.N+1), D))

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary"""
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        return np.where(B.ravel() == 1)[0]

    def assemble(self, g=None):
        """Return assembled matrix A and right hand side vector b"""
        F = sp.lambdify((x, y), self.f)(self.xij, self.yij)
        
        if g is None:
            G = sp.lambdify((x, y), self.ue)(self.xij, self.yij)
        else:
            G = sp.lambdify((x, y), g)(self.xij, self.yij)

        A = self.laplace().tolil()
        bnds = self.get_boundary_indices()

        for i in bnds:
            A[i, : ] = 0
            A[i, i] = 1
        
        A = A.tocsr()
        
        b = F.ravel()
        b[bnds] = G.ravel()[bnds]

        return A , b 

    def l2_error(self, u):
        """Return l2-error norm"""
        return np.sqrt(self.h**2 *np.sum((u - sp.lambdify((x, y), self.ue)(self.xij, self.yij))**2))


    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        The solution as a Numpy array

        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, xq, yq):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        if not hasattr(self, "U"):
            raise RuntimeError("Call the solver first to compute self.U (e.g., U = self(N)).")

        # Clamp query point to the domain
        xq = float(np.clip(xq, 0.0, self.L))
        yq = float(np.clip(yq, 0.0, self.L))

        h = self.h
        N = self.N

        # Cell indices (use the last cell if we hit the boundary)
        i = min(int(np.floor(xq / h)), N - 1)
        j = min(int(np.floor(yq / h)), N - 1)

        x0, x1 = i * h, (i + 1) * h
        y0, y1 = j * h, (j + 1) * h

        # Local (normalized) coordinates in the cell
        tx = 0.0 if x1 == x0 else (xq - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (yq - y0) / (y1 - y0)

        # Corner values
        u00 = self.U[i,     j    ]
        u10 = self.U[i + 1, j    ]
        u01 = self.U[i,     j + 1]
        u11 = self.U[i + 1, j + 1]

        # Bilinear blend
        return ((1 - tx) * (1 - ty) * u00 +
                tx       * (1 - ty) * u10 +
                (1 - tx) * ty       * u01 +
                tx       * ty       * u11)


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

