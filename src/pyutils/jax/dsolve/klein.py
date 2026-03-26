
from jax import Array
from scipy.linalg import ordqz
from jax.scipy.linalg import inv
import jax.numpy as jnp
from dataclasses import dataclass, field
import jax
jax.config.update("jax_enable_x64", True)


@dataclass
class KleinSolution:
    A: Array
    B: Array
    C: Array
    Z_11: Array
    Z_12: Array
    Z_21: Array
    Z_22: Array
    S_11: Array
    S_12: Array
    S_22: Array
    T_11: Array
    T_12: Array
    T_22: Array
    Q_1: Array
    Q_2: Array
    Theta_x: Array
    Theta_p: Array
    L: Array
    N: Array

@dataclass
class KleinModel:
    A: Array
    B: Array
    C: Array
    n: int
    solution: KleinSolution = field(init=False)

    def __post_init__(self):
        self.solution = solve_klein_matrices(self.A, self.B, self.C, self.n)

    def solve_path(self, k0:Array, z:Array, periods=20):
        return solve_klein_path(self.solution, k0, z, periods)


def solve_klein_matrices(A:Array, B:Array, C:Array, n:int):
    """
    Solves the system 
    AE[x_{t+1}] = Bx_t + C z_t
    x_t is decomposed in k_t backward-looking variables and d_t forward looking variables 
    Uses Klein 2000 notation.
    """
    S, T, _, _, Q, Z = ordqz(A, B, output='complex',sort=lambda alpha,beta: jnp.round(jnp.abs(beta/jnp.maximum(alpha,1e-15)),6)<=1) # type: ignore
    Q = Q.conjugate().T

    n_s = len([_ for i in range(S.shape[0]) if jnp.abs(S[i,i])>1e-6 and jnp.round(jnp.abs(T[i,i]/S[i,i]),6)<=1])
    if n_s>n:
        raise ValueError(f"Number of stable eigenvalues {n_s} is larger than the number of backward looking variables {n}.")
    elif n_s<n:
        raise ValueError(f"Number of stable eigenvalues {n_s} is smaller than the number of backward looking variables {n}.")
    Z_11 = Z[:n_s,:n_s]
    Z_12 = Z[:n_s,n_s:]
    Z_21 = Z[n_s:,:n_s]
    Z_22 = Z[n_s:,n_s:]
    S_11 = S[:n_s,:n_s]
    S_12 = S[:n_s,n_s:]
    S_22 = S[n_s:,n_s:]
    T_11 = T[:n_s,:n_s]
    T_12 = T[:n_s,n_s:]
    T_22 = T[n_s:,n_s:]
    Q_1 = Q[:n_s,:]
    Q_2 = Q[n_s:,:]

    Theta_p = jnp.real(Z[n_s:,:n_s]@inv(Z[:n_s,:n_s]))
    Theta_x = jnp.real(Z[:n_s,:n_s]@inv(S[:n_s,:n_s])@T[:n_s,:n_s]@inv(Z[:n_s,:n_s]))
    M = -inv(T[n_s:,n_s:])@Q[n_s:,:]@C
    N = jnp.real((Z[n_s:,n_s:]-Z[n_s:,:n_s]@inv(Z[:n_s,:n_s])@Z[:n_s,n_s:])@M)
    L = jnp.real(Z[:n_s,:n_s]@inv(S[:n_s,:n_s])@((-T[:n_s,:n_s]@inv(Z[:n_s,:n_s])@Z[:n_s,n_s:]+T[:n_s,n_s:])@M+Q[:n_s,:]@C))

    return KleinSolution(A, B, C, Z_11, Z_12, Z_21, Z_22, S_11, S_12, S_22, T_11, T_12, T_22, Q_1, Q_2, Theta_x, Theta_p, L, N)



def compute_u(z:Array, klein_matrices:KleinSolution)->Array:
    """
    Given a path for z, computes the path for the forward-looking variables u using the Klein solution matrices.
    """
    T_22, S_22, Q_2, C = klein_matrices.T_22, klein_matrices.S_22, klein_matrices.Q_2, klein_matrices.C
    T_22_inv = inv(T_22)
    u = -T_22_inv@Q_2@C@z[:,[0]]
    for k in range(1, z.shape[1]):
        u = u - jnp.linalg.matrix_power(T_22_inv@S_22,k)@T_22_inv@Q_2@C@z[:,[k]]
    return u.flatten()




def solve_klein_path(solution: KleinSolution, k0:Array, z:Array, periods=20):
    """
    Solves the system for a perfect foresight path of z and k0
    """
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.shape[1]<=periods:
        z = jnp.hstack([z, jnp.zeros((z.shape[0], periods+1-z.shape[1]))])
    Z_11, Z_12 = solution.Z_11, solution.Z_12
    Z_21, Z_22 = solution.Z_21, solution.Z_22
    S_11, S_12 = solution.S_11, solution.S_12
    T_11, T_12 = solution.T_11, solution.T_12
    Q_1, C = solution.Q_1, solution.C

    u = jnp.zeros((Z_22.shape[0], periods+1), dtype = jnp.complex128)
    for t in range(periods + 1):
        u_t = compute_u(z[:,t:], solution)
        u = u.at[:, t].set(u_t)

    k = jnp.zeros((k0.shape[0], periods+1))
    k = k.at[:, 0].set(k0.reshape(-1))
    s = jnp.zeros((Z_11.shape[0], periods+1), dtype=jnp.complex128)
    d = jnp.zeros((Z_21.shape[0], periods+1))

    s0 = jnp.linalg.solve(
        Z_11,
        k0.reshape(-1, 1) - Z_12 @ u[:, 0].reshape(-1, 1)
    )
    s = s.at[:, 0].set(s0.flatten())

    d0 = Z_21 @ s[:, 0].reshape(-1, 1) + Z_22 @ u[:, 0].reshape(-1, 1)
    d = d.at[:, 0].set(jnp.real(d0).flatten())

    for t in range(periods):
        rhs = (
            T_11 @ s[:, t].reshape(-1, 1)
            + T_12 @ u[:, t].reshape(-1, 1)
            + Q_1 @ C @ z[:, t].reshape(-1, 1)
            - S_12 @ u[:, t+1].reshape(-1, 1)
        )

        s_next = jnp.linalg.solve(S_11, rhs)
        s = s.at[:, t+1].set(s_next.flatten())

        k_next = Z_11 @ s_next + Z_12 @ u[:, t+1].reshape(-1, 1)
        d_next = Z_21 @ s_next + Z_22 @ u[:, t+1].reshape(-1, 1)

        k = k.at[:, t+1].set(jnp.real(k_next).flatten())
        d = d.at[:, t+1].set(jnp.real(d_next).flatten())

    return {
        "t": jnp.arange(periods+1),
        "s": s,
        "u": u,
        "k": k,
        "d": d,
        "z": z,
        "x": jnp.vstack([k, d]),
    }




# def inspect_eigenvalues(A:Array):
#     eigenvalues, _ = jnp.linalg.eig(A)
#     return jnp.sort(jnp.abs(eigenvalues))



