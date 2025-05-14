import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def u_exact(x, t, L, c, alpha):
    """
    Analytical solution for the single sine‐mode damped wave:
      u(x,t) = sin(pi x/L) * exp(-alpha t) * [cos(omega_d t) + (alpha/omega_d) sin(omega_d t)]
    where omega_d = sqrt((pi c / L)^2 - alpha^2).
    """
    omega0 = np.pi * c / L
    omega_d = np.sqrt(max(omega0**2 - alpha**2, 0))
    return np.sin(np.pi * x / L) * np.exp(-alpha * t) * \
           (np.cos(omega_d * t) + (alpha/omega_d) * np.sin(omega_d * t))

def assemble_Crank_Nicolson_matrices(nx, dx, dt, c, alpha):
    """
    Build the Crank–Nicolson left‐ and right‐hand‐side matrices M and B
    for the damped wave finite‐difference discretization in space.
    """
    N = nx - 1  # interior points
    main = -2.0 * np.ones(N) / dx**2
    off  =  1.0 * np.ones(N-1) / dx**2
    A = sp.diags([off, main, off], [-1, 0, 1], format='csc')
    I = sp.eye(N, format='csc')
    M = (1 + alpha*dt)*I - (c**2 * dt**2 / 4) * A
    B = (1 + alpha*dt)*I + (c**2 * dt**2 / 4) * A
    return M, B

def time_step_damped_wave(L=1.0, c=1.0, alpha=0.1,
                           nx=50, dt=0.01, T=1.0):
    """
    Solve u_tt + 2 alpha u_t = c^2 u_xx on [0,L] with u=0 at x=0,L,
    initial u(x,0)=sin(pi x/L), u_t(x,0)=0, via Crank–Nicolson in time.
    Returns the final numerical u, the grid x, and the analytical u_exact at time T.
    """
    dx = L / nx
    x = np.linspace(0, L, nx+1)
    nt = int(np.round(T/dt))
    
    # initial conds (interior)
    U = u_exact(x, 0, L, c, alpha)[1:-1]
    V = np.zeros_like(U)
    
    # assemble matrices & factorize
    M, B = assemble_Crank_Nicolson_matrices(nx, dx, dt, c, alpha)
    lu = spla.splu(M)
    
    # time stepping
    for n in range(nt):
        # build RHS
        RHS = B.dot(U) + dt * V
        # solve for new U
        U_new = lu.solve(RHS)
        # update velocity
        V_new = 2*(U_new - U)/dt - V
        U, V = U_new, V_new
    
    # embed boundary conditions
    U_full = np.zeros(nx+1)
    U_full[1:-1] = U
    
    # analytical at T
    U_ex = u_exact(x, T, L, c, alpha)
    return x, U_full, U_ex

if __name__ == "__main__":
    # parameters
    L     = 1.0     # domain length
    c     = 1.0     # wave speed
    alpha = 0.1     # damping coefficient
    nx    = 100     # number of subintervals
    dt    = 0.005   # time step
    T     = 1.0     # final time
    
    # run solver
    x, U_num, U_ana = time_step_damped_wave(L, c, alpha, nx, dt, T)
    
    # compute max‐norm error
    err = np.max(np.abs(U_num - U_ana))
    print(f"Max error at t={T:.3f}: {err:.2e}")
    
    # plot
    plt.plot(x, U_num, 'o-', label="Numerical")
    plt.plot(x, U_ana, '-', label="Analytical")
    plt.xlabel("x")
    plt.ylabel("u(x,T)")
    plt.title(f"Damped wave @ t={T:.2f}")
    plt.legend()
    plt.show()
