import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import imageio
import os

#--- PARAMETERS (modified for longer sim + faster runtime) ---
L = 1.0            # domain length
c = 1.0            # wave speed
alpha = 0.2        # damping coefficient

Nx = 50            # number of interior spatial points
dx = L / (Nx+1)
x = np.linspace(0, L, Nx+2)

dt = 0.01           # increased time‐step for speed
T_final = 10     # simulate out to 20 s to see near‐zero amplitude
Nt = int(T_final / dt)

#--- BUILD SPATIAL OPERATOR D2 ---
e = np.ones(Nx)
D2 = sp.diags([e, -2*e, e], offsets=[-1,0,1], shape=(Nx,Nx)) / dx**2

#--- BUILD BLOCK MATRIX FOR FIRST‐ORDER SYSTEM ---
Z = sp.csr_matrix((Nx, Nx))
I = sp.eye(Nx)

# du/dt = v
A11 = Z
A12 = I

# dv/dt = c^2 D2 u - 2α v
A21 = (c**2) * D2
A22 = -2*alpha * I

A_block = sp.bmat([[A11, A12],
                   [A21, A22]]).tocsr()

#--- CRANK–NICHOLSON MATRICES ---
Id = sp.eye(2*Nx)
M1 = Id - 0.5*dt*A_block
M2 = Id + 0.5*dt*A_block
solve_step = spla.factorized(M1)

#--- INITIAL CONDITIONS (mode‐1 sine wave) ---
u0 = np.sin(np.pi * x[1:-1])
v0 = np.zeros_like(u0)
w = np.hstack([u0, v0])

#--- GENERATE FRAMES ---
frame_files = []
for n in range(Nt+1):
    t = n * dt
    # save every 5th frame
    if n % 5 == 0:
        ua = np.concatenate(([0], w[:Nx], [0]))
        fig, ax = plt.subplots()
        ax.plot(x, ua, 'b-', label='u(x,t)')
        ax.set_title(f'Damped wave at t = {t:.1f} s')
        ax.set_ylim(-1.0, 1.0)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        fname = f'frame_{n:04d}.png'
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        frame_files.append(fname)
    # advance one Crank–Nicolson step
    w = solve_step(M2.dot(w))

#--- MAKE GIF ---
gif_name = 'damped_wave_long.gif'
with imageio.get_writer(gif_name, mode='I', fps=10) as writer:
    for fname in frame_files:
        image = imageio.imread(fname)
        writer.append_data(image)

#--- CLEAN UP FRAMES ---
for fname in frame_files:
    os.remove(fname)

print(f"✔ GIF saved as {gif_name}")