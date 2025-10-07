import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import os
from mpl_toolkits.mplot3d import Axes3D  

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.N = N
        self.h = 1./self.N
        x = np.linspace(0, 1., self.N+1)
        y = np.linspace(0, 1., self.N+1)
        self.xij, self.yij = np.meshgrid(x, y, indexing='ij')
        return self.xij, self.yij

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * sp.pi * sp.sqrt(self.mx**2 + self.my**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.N = N
        self.mx = mx
        self.my = my
        self.create_mesh(N)
        self.uem = self.ue(self.mx, self.my)
        self.f_expr = sp.diff(self.uem, t, 2) - self.c**2 * (sp.diff(self.uem, x, 2) + sp.diff(self.uem, y, 2))
        self.h = 1./N
        D = self.D2(N)
        self.D2XY = D / self.h**2
        I = sparse.eye(self.N+1, format='csr')
        self.laplace = (sparse.kron(self.D2XY, I, format='csr') + sparse.kron(I, self.D2XY, format='csr'))
        self.f = sp.lambdify((x, y, t), self.f_expr, "numpy")

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue_fun = sp.lambdify((x, y, t), self.uem, "numpy")
        ue_grid = ue_fun(self.xij, self.yij, t0)
        return np.sqrt(self.h**2 * np.sum((u - ue_grid)**2))

    def apply_bcs(self, u):
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.c = c
        self.cfl = cfl
        self.initialize(N, mx, my)
        self.U = []
        dt = self.dt
        dt2 = dt**2

        # Initial condition u^0
        ue_fun = sp.lambdify((x, y, t), self.uem, "numpy")
        ue0 = ue_fun(self.xij, self.yij, 0)
        u_now = ue0.copy()
        self.U.append(u_now.copy())

        # Compute u^1 using second order starter
        lap_u = self.laplace @ u_now.ravel()
        lap_u = lap_u.reshape((N+1, N+1))
        f_now = self.f(self.xij, self.yij, 0)
        rhs = self.c**2 * lap_u + f_now
        u_next = u_now + (dt2 / 2) * rhs
        self.apply_bcs(u_next)
        self.U.append(u_next.copy())

        u_prev = u_now.copy()
        u_now = u_next.copy()

        for n in range(1, Nt):
            self.t_cur = n * dt
            lap_u = self.laplace @ u_now.ravel()
            lap_u = lap_u.reshape((N+1, N+1))
            f_now = self.f(self.xij, self.yij, self.t_cur)
            rhs = self.c**2 * lap_u + f_now
            u_next = 2 * u_now - u_prev + dt2 * rhs
            self.apply_bcs(u_next)
            self.U.append(u_next.copy())
            u_prev = u_now.copy()
            u_now = u_next.copy()
        
        if store_data == -1:
            err = [self.l2_error(self.U[n], n*self.dt) for n in range(Nt+1)]
            return self.h, err
        elif store_data > 0:
            out = {}
            for n in range(Nt+1):
                out[n*self.dt] = self.U[n]
            return out
        else: raise RuntimeError("Invalid stored_data parameter")

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

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
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0,1] = 2
        D[-1, -2] = 2
        return D

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self, u):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    # Test for Dirichlet problem
    sol_dir = Wave2D()
    _, err_dir = sol_dir(64, 100, cfl=1/np.sqrt(2), c=1.0, mx=2, my=2, store_data=-1)
    assert err_dir[-1] < 1e-12, f"Dirichlet l2-error: {err_dir[-1]}"

    # Test for Neumann problem
    sol_neum = Wave2D_Neumann()
    _, err_neum = sol_neum(64, 100, cfl=1/np.sqrt(2), c=1.0, mx=2, my=2, store_data=-1)
    assert err_neum[-1] < 1e-12, f"Neumann l2-error: {err_neum[-1]}"