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
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N+1, self.N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c * np.sqrt(self.mx**2 + self.my**2)

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
        self.f = self.c**2 * (sp.diff(self.uem, x, 2) + sp.diff(self.uem, y, 2)) - sp.diff(self.uem, t, 2)

        self.h = 1./N
        D = self.D2(N)
        D2XY = D / self.h**2
        I = sparse.eye(self.N+1, format='csr')
        laplace = (sparse.kron(D2XY, I, format='csr') + sparse.kron(I, D2XY, format='csr'))
        D2T = D / self.dt**2

        # D2T_big = sparse.eye((self.N+1)*(self.N+1), format='csr')
        D2T_big = sparse.kron(I, D2T,format = 'csr')
        self.A = self.c**2 * laplace - D2T_big 
        # self.A = self.c**2 * laplace
 

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

    def apply_bcs(self):
        F = sp.lambdify((x, y, t), self.f, "numpy")(self.xij, self.yij, self.t_cur)
        B = np.ones((self.N+1, self.N+1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        u_bc = sp.lambdify((x, y, t), self.uem, "numpy")(self.xij, self.yij, self.t_cur)
        A = self.A.tolil()
        for i in bnds:
            A[i, : ] = 0
            A[i, i] = 1
        b = F.ravel()
        # b[bnds] = u_bc.ravel()[bnds]
        b[bnds] = 0.0
        A = A.tocsr()
        self.A = A
        self.b = b
        return A, b

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
        self.initialize(N, mx,my)
        self.U=[]
        for n in range(Nt):
            self.t_cur = n * self.dt
            A , b = self.apply_bcs()
            um = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
            self.U.append(um)
        
        if store_data == -1:
            err = [self.l2_error(self.U[n], n*self.dt) for n in range(Nt)]
            return self.h, err
        elif store_data > 0:
            out = {}
            for n in range(Nt):
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
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        raise NotImplementedError

    def ue(self, mx, my):
        raise NotImplementedError

    def apply_bcs(self):
        raise NotImplementedError

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

# def test_convergence_wave2d_neumann():
#     solN = Wave2D_Neumann()
#     r, E, h = solN.convergence_rates(mx=2, my=3)
#     assert abs(r[-1]-2) < 0.05

# def test_exact_wave2d():
#     raise NotImplementedError

def plot_solution_3d(xij, yij, u, *, ax=None, kind="wireframe", stride=2):
    """
    Quick 3D snapshot of a single solution frame u(x,y).
    kind: 'wireframe' (lighter) or 'surface'
    """
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if kind == "wireframe":
        ax.plot_wireframe(xij, yij, u, rstride=stride, cstride=stride)
    else:
        ax.plot_surface(xij, yij, u, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
    return ax

def make_gif_from_data(xij, yij, data_dict, *, filename="report/neumannwave.gif",
                       kind="wireframe", stride=2, fps=8, dpi=90,
                       subsample=1, zclip=None):
    """
    Create a lightweight GIF from the dict returned by __call__ when store_data>0.
    - subsample: keep every k-th frame to shrink size
    - kind='wireframe' keeps file small; 'surface' looks nicer but heavier
    - zclip=(vmin, vmax) to fix color/axis range and improve compression
    """

    # order frames by time and subsample
    times = sorted(data_dict.keys())
    if subsample > 1:
        times = times[::subsample]

    # optional z-limits to stabilize color/scale
    if zclip is None:
        vmax = max(np.nanmax(np.abs(data_dict[t])) for t in times)
        vmin = -vmax
    else:
        vmin, vmax = zclip

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u')
    ax.set_zlim(vmin, vmax)

    artists = []
    for t in times:
        u = data_dict[t].astype(np.float32)  # smaller memory, better compression
        if kind == "wireframe":
            art = ax.plot_wireframe(xij, yij, u, rstride=stride, cstride=stride)
        else:
            art = ax.plot_surface(xij, yij, u, linewidth=0, antialiased=False, cmap=cm.coolwarm)
        artists.append([art])

    ani = animation.ArtistAnimation(fig, artists, interval=1000//fps, blit=True, repeat_delay=800)

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Save as GIF with Pillow (good compression). Tweak fps/dpi/subsample if >1MB.
    ani.save(filename, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    return filename

if __name__ == "__main__":
    N   = 100
    Nt  = 200
    # cfl = .7/np.sqrt(2)
    cfl = 10
    c   = 1.0
    mx, my = 4, 1

    solN = Wave2D()
    data = solN(N=N, Nt=Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=1)

    # Quick visual sanity check for one frame
    some_t = sorted(data.keys())[0]
    plot_solution_3d(solN.xij, solN.yij, data[some_t], kind="wireframe", stride=2)
    plt.title(f"Wave2D – t={some_t:.3f}")
    plt.show()

    # Make the GIF (wireframe + subsampling keeps the file tiny)
    out_path = make_gif_from_data(
        solN.xij, solN.yij, data,
        filename="report/wave.gif",
        kind="wireframe",
        stride=3,      # fewer mesh lines -> smaller file
        fps=10,         # modest fps
        dpi=90,        # modest resolution
        subsample=2,   # keep every 2nd frame
        zclip=None     # auto symmetric z-limits
    )
    print("Saved:", out_path)