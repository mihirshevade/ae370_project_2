import numpy as np
import matplotlib.pyplot as plt
from solver import time_step_damped_wave

def spatial_convergence(nx_list, dt, T, L, c, alpha):
    """
    Spatial convergence: fix dt small, vary nx, compute error at t=T.
    Returns dx_list, err_list.
    """
    dx_list = []
    err_list = []
    for nx in nx_list:
        x, U_num, U_ex = time_step_damped_wave(L=L, c=c, alpha=alpha,
                                                nx=nx, dt=dt, T=T)
        dx = L / nx
        dx_list.append(dx)
        err_list.append(np.max(np.abs(U_num - U_ex)))
    return np.array(dx_list), np.array(err_list)

def temporal_convergence(dt_list, nx, T, L, c, alpha):
    """
    Temporal convergence: fix nx fine, vary dt, compute error at t=T.
    Returns dt_list, err_list.
    """
    err_list = []
    for dt in dt_list:
        x, U_num, U_ex = time_step_damped_wave(L=L, c=c, alpha=alpha,
                                                nx=nx, dt=dt, T=T)
        err_list.append(np.max(np.abs(U_num - U_ex)))
    return np.array(dt_list), np.array(err_list)

def fit_slope(x, y):
    """Fit log(y)=m log(x)+b to compute slope m."""
    m, b = np.polyfit(np.log(x), np.log(y), 1)
    return m

if __name__ == "__main__":
    # parameters
    L     = 1.0
    c     = 1.0
    alpha = 0.1
    T     = 1.0

    # Spatial study: dt small enough so temporal error ≪ spatial error
    dt_spatial = 1e-4
    nx_list    = np.array([50, 100, 200, 400, 800])
    dx, e_spat = spatial_convergence(nx_list, dt_spatial, T, L, c, alpha)
    m_spat     = fit_slope(dx, e_spat)

    # Temporal study: nx large so spatial error ≪ temporal error
    nx_temp    = 800
    dt_list    = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3])
    dt, e_temp = temporal_convergence(dt_list, nx_temp, T, L, c, alpha)
    m_temp     = fit_slope(dt, e_temp)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Spatial convergence plot
    ax = axes[0]
    ax.loglog(dx, e_spat, 'o-', label=f"error (slope ≈ {m_spat:.2f})")
    # reference slope‑2 line
    ref = e_spat[0] * (dx / dx[0])**2
    ax.loglog(dx, ref, '--', label="Δx² reference")
    ax.set_xlabel("Δx")
    ax.set_ylabel("max |u_num − u_exact|")
    ax.set_title("Spatial convergence (Δt = {:.0e})".format(dt_spatial))
    ax.legend()
    ax.grid(True, which="both", ls=":")

    # Temporal convergence plot
    ax = axes[1]
    ax.loglog(dt, e_temp, 'o-', label=f"error (slope ≈ {m_temp:.2f})")
    # reference slope‑2 line
    ref = e_temp[0] * (dt / dt[0])**2
    ax.loglog(dt, ref, '--', label="Δt² reference")
    ax.set_xlabel("Δt")
    ax.set_ylabel("max |u_num − u_exact|")
    ax.set_title(f"Temporal convergence (nx = {nx_temp})")
    ax.legend()
    ax.grid(True, which="both", ls=":")

    plt.tight_layout()
    plt.show()