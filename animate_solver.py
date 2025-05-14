import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import scipy.sparse.linalg as spla

# import your solver routines
from solver import assemble_Crank_Nicolson_matrices, u_exact

# ─── PARAMETERS ────────────────────────────────────────────────────────────────
L     = 1.0      # domain length
c     = 1.0      # wave speed
alpha = 0.2      # damping

nx    = 100      # number of subintervals → interior points = nx-1
dt    = 0.01     # time step
T_final = 10.0   # total simulation time
frame_skip = 5   # save every 5th step

# ─── GRID SETUP ────────────────────────────────────────────────────────────────
dx = L / nx
x  = np.linspace(0, L, nx+1)     # solver.py uses nx+1 total points
Nt = int(np.round(T_final / dt))

# ─── BUILD MATRICES ───────────────────────────────────────────────────────────
M, B = assemble_Crank_Nicolson_matrices(nx, dx, dt, c, alpha)
lu = spla.splu(M)  # factorization for fast solves

# ─── INITIAL CONDITIONS ────────────────────────────────────────────────────────
# interior U and V
U = u_exact(x, 0.0, L, c, alpha)[1:-1].copy()
V = np.zeros_like(U)

# ─── TIME‑STEPPING & FRAME CAPTURE ─────────────────────────────────────────────
frame_files = []
for n in range(Nt+1):
    t = n * dt

    # grab a frame every `frame_skip` steps
    if n % frame_skip == 0:
        U_full = np.zeros_like(x)
        U_full[1:-1] = U
        fig, ax = plt.subplots()
        ax.plot(x, U_full, 'b-', label='numerical')
        ax.plot(x, u_exact(x, t, L, c, alpha), 'r--', label='exact')
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f'Damped wave @ t = {t:.1f}s')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        fname = f'frame_{n:05d}.png'
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        frame_files.append(fname)

    # one Crank–Nicolson step (interior)
    RHS   = B.dot(U) + dt*V
    U_new = lu.solve(RHS)
    V_new = 2*(U_new - U)/dt - V

    U, V = U_new, V_new

# ─── WRITE OUT GIF ─────────────────────────────────────────────────────────────
gif_name = 'damped_wave_long.gif'
with imageio.get_writer(gif_name, mode='I', fps=10) as writer:
    for fname in sorted(frame_files):
        writer.append_data(imageio.imread(fname))

print(f"✔ GIF saved as {gif_name}")

# ─── CLEAN UP ─────────────────────────────────────────────────────────────────
for fname in frame_files:
    os.remove(fname)