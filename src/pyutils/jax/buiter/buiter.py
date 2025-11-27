import jax.numpy as jnp
from jax import Array
import jax
from jax.experimental.ode import odeint
from pyutils.jax.solvers import newton_solver

def diagonalize(A:Array)->tuple[Array,Array]:
    vals, V_inv = jnp.linalg.eig(A)
    idx = jnp.argsort(jnp.real(vals))
    vals = vals[idx]
    V_inv = V_inv[:, idx]
    V = jnp.linalg.inv(V_inv)
    Lambda = jnp.diag(vals)
    return V_inv, Lambda, V


def find_y0(A:Array, B:Array, x0:Array):
    """
    n_1: number of pre-determined variables
    """
    x = x0.reshape(-1,1)
    n_1 = x0.shape[0]
    V_inv, Lambda, V = diagonalize(A)
    if jnp.sum(jnp.diag(Lambda.real)<=0)>n_1:
        raise ValueError(f"More stable roots ({jnp.sum(jnp.diag(Lambda.real)<=0)}) than pre-determined variables ({n_1}), infinite solutions")
    if jnp.sum(jnp.diag(Lambda.real)<=0)<n_1:
        raise ValueError(f"Less stable roots ({jnp.sum(jnp.diag(Lambda.real)<=0)}) than pre-determined variables ({n_1}), no solution")
    
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
    y0 = -jnp.linalg.inv(V_22)@V_21@x-jnp.linalg.inv(V_22)@jnp.linalg.inv(Lambda_2)@D
    if jnp.any(jnp.abs(y0.imag)>1e-6):
        raise ValueError(f"y0 = {y0} has imaginary numbers")
    y0 = y0.real
    return y0

def simulate(A:Array, B:Array, x0:Array, T:int=1, dt:float = 0.01)->Array:
    """
    Simulate the system given initial state x0 and time horizon T
    [dot x; dot y] = A [x; y] + B
    """
    n = x0.shape[0]
    sol = jnp.zeros((T,n))
    y0 = find_y0(A, B, x0)
    def f(x:Array, t:float):
        x = x.reshape(-1,1)
        return (A @ x + B).flatten()
    t = jnp.arange(0, T, dt)
    sol = odeint(f, jnp.concatenate([x0,y0]).flatten(), t)
    return t, sol


def linearize(f:callable, x0:Array)->tuple[Array,Array,Array]:

    if jnp.max(jnp.abs(f(x0)))<1e-8:
        pass
    else:
        steady_state = newton_solver(f, x0, verbose = False)
        if not steady_state.success:
            raise ValueError("Steady state not found")
        steady_state = steady_state.x.reshape(-1,1)
    A = jax.jacfwd(f)(steady_state).squeeze()
    B = -A @ steady_state
    return A, B, steady_state