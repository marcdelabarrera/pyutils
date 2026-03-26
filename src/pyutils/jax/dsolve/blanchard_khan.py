
from jax import Array
from jax.scipy.linalg import inv
import jax.numpy as jnp
from dataclasses import dataclass, field



@dataclass
class BKSolution:
    Theta_x: Array
    Theta_p: Array
    B_11: Array
    B_21: Array
    B_12: Array
    B_22: Array
    C_11: Array
    C_21: Array
    C_12: Array
    C_22: Array
    J_1: Array
    J_2: Array
    gamma_1: Array
    gamma_2: Array


@dataclass
class BKModel:
    """
    x_t+1 p_t+1 = A x_t p_t+ gamma z_t
    """
    A: Array
    gamma: Array
    n: int                         # number of backward looking variables
    m: int = field(init=False)   # number of forward looking variables
    solution: BKSolution = field(init=False)

    def __post_init__(self):
        self.solution = solve_blanchard_khan_matrices(self.A, self.gamma, self.n)
        self.m = self.A.shape[0] - self.n

    def solve_path(self, x0:Array, z:Array, T: int|None = None):
        if T is None:
            T = z.shape[1]
        else:
            z = jnp.concatenate([z, jnp.zeros((z.shape[0], T-z.shape[1]))], axis=1) 
        return solve_blanchard_khan_path(self.solution, x0, z)


def solve_blanchard_khan_path(matrices:BKSolution, x0:Array, z:Array):
    """
    Solves the system given a deterministic path of shocks z and initial state x0
    """

    
    
    B_11, B_12 = matrices.B_11, matrices.B_12
    C_12, C_22, C_21 = matrices.C_12, matrices.C_22, matrices.C_21
    J_1, J_2 = matrices.J_1, matrices.J_2
    gamma_1, gamma_2 = matrices.gamma_1, matrices.gamma_2

    if x0.shape[0] != B_11.shape[0]:
        raise ValueError(f"Dimension of x0 {x0.shape[0]} does not match expected dimension {B_11.shape[0]}.")
    if z.shape[0] != gamma_1.shape[1]:
        raise ValueError(f"Dimension of z {z.shape[0]} does not match expected dimension {gamma_1.shape[1]}.")
    B_11, B_12 = B_11, B_12
    C_12, C_22, C_21 = C_12, C_22, C_21
    J_1, J_2 = J_1, J_2
    gamma_1, gamma_2 = gamma_1, gamma_2
    B11_inv = jnp.linalg.inv(B_11)
    C22_inv = jnp.linalg.inv(C_22)
    J2_inv = jnp.linalg.inv(J_2)
    T = z.shape[1]
    n_x = x0.shape[0]
    n_p = C_22.shape[0]
    n_u = J_2.shape[0]

    B11_inv = jnp.linalg.inv(B_11)
    C22_inv = jnp.linalg.inv(C_22)
    J2_inv = jnp.linalg.inv(J_2)

    A_x = B_11 @ J_1 @ B11_inv
    G = C_21 @ gamma_1 + C_22 @ gamma_2
    F_x = (B_11 @ J_1 @ C_12 + B_12 @ J_2 @ C_22) @ C22_inv

    s = jnp.zeros((n_u, T + 1), dtype=complex)

    for t in range(T - 1, -1, -1):
        s_t = J2_inv @ (G @ z[:, t] + s[:, t + 1])
        s = s.at[:, t].set(s_t)

    x = jnp.zeros((n_x, T + 1), dtype=complex)
    x = x.at[:, 0].set(x0)

    for t in range(1, T + 1):
        x_t = A_x @ x[:, t - 1] + gamma_1 @ z[:, t - 1] - F_x @ s[:, t - 1]
        x = x.at[:, t].set(x_t)

    p = jnp.zeros((n_p, T + 1), dtype=complex)

    for t in range(T + 1):
        p_t = -C22_inv @ (C_21 @ x[:, t] + s[:, t])
        p = p.at[:, t].set(p_t)

    if jnp.abs(x.imag).max() > 1e-6:
        raise ValueError("Warning: solution has significant imaginary part.")
    elif jnp.abs(p.imag).max() > 1e-6:
        raise ValueError("Warning: solution has significant imaginary part.")
    
    return {
        "t": jnp.arange(T + 1),
        "x": jnp.real(x),
        "p": jnp.real(p),
    }
    




def solve_blanchard_khan_matrices(A:Array, gamma:Array, n:int)-> BKSolution:
    """
    Solves the system [X_t+1, P_t+1] = A [X_t, P_t] + gamma Z_t
    where X is an (nx1) vector of variables predetermined at t; P is an (mx1) vector
    
    Returns Theta_x so X_t = Theta_x@X_t-1+gamma_1 Z_t-1... and Theta_p so P_t = Theta_p@X_t+...
    #TODO: return full solutions
    Parameters:
    A: (n+m)x(n+m) matrix
    gamma: (n+m)x1 vector
    n: number of predetermined variables
    """
    eigenvalues, V = jnp.linalg.eig(A)
    idx = jnp.argsort(jnp.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]       
    C = inv(V) 
    C_inv = inv(C)         
    J = jnp.diag(eigenvalues)
    n_bar = int(jnp.sum(jnp.abs(eigenvalues)<=1))
    m = A.shape[0] - n
    m_bar = int(jnp.sum(jnp.abs(eigenvalues)>1))
    if m_bar > m:
        raise ValueError(f"No Solution")
    elif m_bar < m:
        raise ValueError(f"Infinite Solution")

    J_1 = J[:n_bar,:n_bar]
    J_2 = J[n_bar:,n_bar:]
    C_11 = C[:n_bar,:n]
    C_12 = C[:n_bar,n_bar:]
    C_21 = C[n_bar:,:n]
    C_22 = C[n_bar:,n_bar:]
    B_11 = C_inv[:n,:n_bar]
    B_12 = C_inv[:n,n_bar:]
    B_21 = C_inv[n_bar:,:n_bar]
    B_22 = C_inv[n_bar:,n_bar:]
    gamma_1 = gamma[:n_bar,:]
    gamma_2 = gamma[n_bar:,:]

    Theta_x = B_11@J_1@inv(B_11)
    Theta_p = -inv(C_22)@C_21
    return BKSolution(Theta_x=Theta_x, Theta_p=Theta_p,
            B_11=B_11, B_21=B_21, B_12=B_12, B_22=B_22,
            C_11=C_11, C_21=C_21, C_12=C_12, C_22=C_22,
            J_1=J_1, J_2=J_2,
            gamma_1=gamma_1, gamma_2=gamma_2,
            )









