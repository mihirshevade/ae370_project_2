import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib import animation

def damped_wave_cn(c, alpha, L, nx, dt, T):
    x = np.linspace(0, L, nx+1)
    dx = L / nx
    r = (c * dt / dx)**2

    # Construct A matrix for second derivative
    diagonals = [np.ones(nx-1), -2*np.ones(nx-1), np.ones(nx-1)]
    A = sp.diags(diagonals, [-1, 0, 1], shape=(nx-1, nx-1), format='csr')

    # Identity
    I = sp.eye(nx-1, format='csr')

    # Initial conditions: u(x,0) = sin(pi x / L), ut(x,0)=0
    u0 = np.sin(np.pi * x[1:-1] / L)
    v0 = np.zeros_like(u0)

    # Time stepping: recast to first-order system [u; v]
    # We form Crank-Nicolson for v dynamics: v_{n+1} = f(u_n, v_n, u_{n+1})
    # But easier: update u via
    # u_{n+1} = u_n + dt * v_n + dt^2/2 * (c^2 u_{xx} - 2 alpha v) at midpoints
    # Here we implement standard CN on the second-order form directly:
    # (I + alpha*dt) u_{n+1} - r/2 * A u_{n+1} = (2 I - alpha*dt + r/2 * A) u_n - (I - alpha*dt) u_{n-1}
    # But since ut(0)=0, we use a ghost u_{-1} = u0 - dt*v0 = u0
    
    # Precompute matrices for CN:
    M1 = (1 + alpha*dt) * I - 0.5 * r * A
    M2 = (2 - alpha*dt) * I + 0.5 * r * A
    M3 = (1 - alpha*dt) * I - 0.5 * r * A  # multiplies u_{n-1}

    # initial "ghost" step u_{-1}
    u_prev = u0.copy()  
    u_curr = u0.copy()

    times = np.arange(0, T+dt, dt)
    solutions = []

    for n, t in enumerate(times):
        solutions.append(u_curr.copy())
        # Solve for u_next
        b = M2.dot(u_curr) - M3.dot(u_prev)
        u_next = spla.spsolve(M1, b)
        u_prev, u_curr = u_curr, u_next

    return x, times, solutions

# Parameters
c = 1.0
alpha = 0.2
L = 1.0
nx = 100
dt = 0.005
T = 2.0

x, times, sols = damped_wave_cn(c, alpha, L, nx, dt, T)

# Choose snapshot times
snapshot_times = [0.0, 0.67*T, 1.33*T, T]
indices = [np.argmin(np.abs(times - t)) for t in snapshot_times]

# Plot subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
axs = axs.flatten()
for ax, idx, t in zip(axs, indices, snapshot_times):
    ax.plot(x, np.concatenate(([0], sols[idx], [0])), 'o-')
    ax.set_title(f"t = {t:.2f}")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
fig.tight_layout()
plt.show()

# Create animation
fig2, ax2 = plt.subplots()
line, = ax2.plot([], [], lw=2)
ax2.set_xlim(0, L)
ax2.set_ylim(-1.1, 1.1)
ax2.set_xlabel("x")
ax2.set_ylabel("u(x,t)")
title = ax2.text(0.5, 1.05, "", transform=ax2.transAxes, ha="center")

def init():
    line.set_data([], [])
    title.set_text("")
    return line, title

def animate(i):
    u = np.concatenate(([0], sols[i], [0]))
    line.set_data(x, u)
    title.set_text(f"t = {times[i]:.2f}")
    return line, title

anim = animation.FuncAnimation(fig2, animate, init_func=init,
                               frames=len(times), interval=30, blit=True)

anim.save('damped_wave.gif', writer='imagemagick')

plt.close(fig2)

# The subplot figure is displayed above,
# and the animation is saved as a GIF below for download.