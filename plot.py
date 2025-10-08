
import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import os
from mpl_toolkits.mplot3d import Axes3D  
from Wave2D import Wave2D, Wave2D_Neumann

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
    N = 60
    Nt = 70
    cfl = 1/np.sqrt(2)
    c = 1.0
    mx, my = 2, 2

    solN = Wave2D_Neumann()
    data = solN(N=N, Nt=Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=1)

    # Quick visual sanity check for one frame
    some_t = sorted(data.keys())[0]
    plot_solution_3d(solN.xij, solN.yij, data[some_t], kind="wireframe", stride=2)
    plt.title(f"Wave2D_Neumann – t={some_t:.3f}")

    # Make the GIF (wireframe + subsampling keeps the file tiny)
    out_path = make_gif_from_data(
        solN.xij, solN.yij, data,
        filename="report/neumannwave.gif",
        kind="wireframe",
        stride=3,
        fps=8,
        dpi=90,
        subsample=2,
        zclip=None
    )
    # plt.show()
    print("Saved:", out_path)