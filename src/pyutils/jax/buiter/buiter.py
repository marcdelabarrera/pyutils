import jax.numpy as jnp
from jax import Array

def diagonalize(A:Array)->tuple[Array,Array]:
    vals, V_inv = jnp.linalg.eig(A)
    idx = jnp.argsort(jnp.real(vals))
    vals = vals[idx]
    V_inv = V_inv[:, idx]
    V = jnp.linalg.inv(V_inv)
    Lambda = jnp.diag(vals)
    return V_inv, Lambda, V


def solve(A:Array, B:Array, x:Array):
    """
    n_1: number of pre-determined variables
    """
    x = x.reshape(-1,1)
    n_1 = x.shape[1]
    V_inv, Lambda, V = diagonalize(A)
    if jnp.sum(jnp.diag(Lambda.real)<=0)>n_1:
        raise ValueError("More stable roots than pre-determined variables, infinite solutions")
    if jnp.sum(jnp.diag(Lambda.real)<=0)<n_1:
        raise ValueError("Less stable roots than pre-determined variables, no solution")
    

    Lambda_1 = Lambda[:n_1,:n_1]
    Lambda_2 = Lambda[n_1:,n_1:]
    A_11 = A[:n_1,:n_1]
    A_12 = A[:n_1,n_1:]
    A_21 = A[n_1:,:n_1]
    A_22 = A[n_1:,n_1:]
    V_11 = V[:n_1,:n_1]
    V_12 = V[:n_1,n_1:]
    V_21 = V[n_1:,:n_1]
    V_22 = V[n_1:,n_1:]
    W_11 = V_inv[:n_1,:n_1]
    W_12 = V_inv[:n_1,n_1:]
    W_21 = V_inv[n_1:,:n_1]
    W_22 = V_inv[n_1:,n_1:]
    B_1 = B[:n_1]
    B_2 = B[n_1:]

    D = V_21@B_1 + V_22 @ B_2
    y = -jnp.linalg.inv(V_22)@V_21@x-jnp.linalg.inv(V_22)@jnp.linalg.inv(Lambda_2)@D
    if jnp.any(jnp.abs(y.imag)>1e-6):
        raise ValueError(f"y = {y} has imaginary numbers")
    y = y.real
    return y