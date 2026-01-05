from jax import Array
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from .sparse import spdiagm

def compute_forward_derivative(x:Array, ghost_node:bool=False) -> BCOO:
    """
    Computes the matrix D such that D@f = df/dx using forward differences. Given a vector
    x[i] for i = 0,...,n-1 and a function f defined on those points, then

    df_i/dx = (f_{i+1} - f_{i}) / (x_{i+1} - x_{i})

    if ghost_node is False, then df[n-1]/dx = 0. If ghost_node is True
    then df[n-1]/dx = -f[n-1] / (x[n-1] - x[n-2]). That is, assumes a ghost node
    at position x[n] = x[n-1] + (x[n-1]-x[n-2]) with f[n] = 0.
    """
    if ghost_node:
        return spdiagm(1/jnp.diff(x), k = 1) - spdiagm(jnp.concatenate((1/jnp.diff(x), jnp.array([1/(x[-1]-x[-2])]))))
    else:
        return spdiagm(1/jnp.diff(x), k = 1) - spdiagm(jnp.concatenate((1/jnp.diff(x), jnp.array([0]))))

def compute_backward_derivative(x:Array, ghost_node:bool=False) -> BCOO:
    """
    Computes the matrix D such that D@f = df/dx using backward differences. Given a vector
    x[i] for i = 0,...,n-1 and a function f defined on those points, then
    df_i/dx = (f_i - f_{i-1}) / (x_i - x_{i-1})

    if ghost_node is False, then df[0]/dx = 0. If ghost_node is True
    then df[0]/dx = f[0] / (x[1] - x[0]). That is, assumes a ghost node
    at position x[-1] = x[0] - (x[1]-x[0]) with f[-1] = 0.
    """
    if ghost_node:
        return spdiagm(jnp.concatenate((jnp.array([1/(x[1]-x[0])]), 1/jnp.diff(x)))) - spdiagm(1/jnp.diff(x), k=-1)
    else:
        return spdiagm(jnp.concatenate((jnp.array([0]), 1/jnp.diff(x)))) - spdiagm(1/jnp.diff(x), k=-1)
    

def compute_second_derivative(x:Array) -> BCOO:
    n = x.shape[0]
    dx = jnp.diff(x)
    dx2 = jnp.concatenate((dx[:1],dx))*jnp.concatenate((dx,dx[-1:]))
    return spdiagm(1/dx2[:-1], k=1) + spdiagm(1/dx2[1:], k=-1) - spdiagm(jnp.full(n,2).at[0].set(1).at[-1].set(1)/dx2)
