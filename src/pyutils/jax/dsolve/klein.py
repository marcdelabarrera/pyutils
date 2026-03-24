
from jax import Array
from scipy.linalg import ordqz
from jax.scipy.linalg import inv
import jax.numpy as jnp
from dataclasses import dataclass, field



def solve_klein(A:Array, B:Array, C:Array, k0:Array, z0:Array, periods=20):
    """
    Solves the system 
    AE[x_{t+1}] = Bx_t + C z_t
    x_t is decomposed in k_t backward-looking variables and d_t forward looking variables 
    Uses Klein 2000 notation.
    #TODO: right now only for mixed systems.
    """
    S, T, _, _, Q, Z = ordqz(A, B, output='complex',sort=lambda alpha,beta: jnp.round(jnp.abs(beta/jnp.maximum(alpha,1e-15)),6)<=1) # type: ignore
    Q = Q.conjugate().T

    n_s = len([_ for i in range(S.shape[0]) if jnp.abs(S[i,i])>1e-6 and jnp.round(jnp.abs(T[i,i]/S[i,i]),6)<=1])

    if n_s>k0.shape[0]:
        raise ValueError(f"Number of stable eigenvalues {n_s} is larger than the number of backward looking variables {k0.shape[0]}.")
    elif n_s<k0.shape[0]:
        raise ValueError(f"Number of stable eigenvalues {n_s} is smaller than the number of backward looking variables {k0.shape[0]}.")
    
    Theta_p = jnp.real(Z[n_s:,:n_s]@inv(Z[:n_s,:n_s]))
    Theta_x = jnp.real(Z[:n_s,:n_s]@inv(S[:n_s,:n_s])@T[:n_s,:n_s]@inv(Z[:n_s,:n_s]))
    M = -inv(T[n_s:,n_s:])@Q[n_s:,:]@C
    N = jnp.real((Z[n_s:,n_s:]-Z[n_s:,:n_s]@inv(Z[:n_s,:n_s])@Z[:n_s,n_s:])@M)
    L = jnp.real(Z[:n_s,:n_s]@inv(S[:n_s,:n_s])@((-T[:n_s,:n_s]@inv(Z[:n_s,:n_s])@Z[:n_s,n_s:]+T[:n_s,n_s:])@M+Q[:n_s,:]@C))

    k = jnp.zeros((Theta_x.shape[0], periods+1))
    d = jnp.zeros((Theta_p.shape[0], periods+1))
    k = k.at[:,0].set(k0)
    z = jnp.zeros((C.shape[1], periods+1))
    z = z.at[:,0].set(z0)
    # Initialize d at t=0 using the saddle-path condition d_t = Theta_p @ k_t + N @ z_t
    d = d.at[:,0].set((Theta_p@k0.reshape(-1,1)+N@z0.reshape(-1,1)).flatten())
    for t in range(periods):
        iz = z[:,t].reshape(-1,1)
        k_next = (Theta_x@k[:,t].reshape(-1,1)+L@iz).flatten()
        k = k.at[:,t+1].set(k_next)
        iz_next = z[:,t+1].reshape(-1,1)
        d = d.at[:,t+1].set((Theta_p@k_next.reshape(-1,1)+N@iz_next).flatten())
    return k, d, z



def inspect_eigenvalues(A:Array):
    eigenvalues, _ = jnp.linalg.eig(A)
    return jnp.sort(jnp.abs(eigenvalues))



