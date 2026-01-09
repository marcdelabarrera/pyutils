import jax
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




@jax.jit
def compute_interpolation_weights(x: float | Array, grid: Array) -> tuple[Array, Array, Array]:
    """
    Given a grid and a scalar value x, returns the interpolation indices and weights
    for linear interpolation between the two grid points surrounding x.

    Args:
        x: Scalar value to interpolate (will be clipped to grid bounds)
        grid: 1D array of grid points (must be sorted)

    Returns:
        Tuple of (indices, grid_values, weights) where:
        - indices: Array of shape (2,) with left and right indices
        - grid_values: Array of shape (2,) with corresponding grid values
        - weights: Array of shape (2,) with interpolation weights summing to 1
    """
    x = jnp.clip(x, grid[0], grid[-1])
    ix = jnp.searchsorted(grid, x)
    w_left = jnp.clip((grid[ix] - x) / (grid[ix] - grid[ix-1]), 0, 1)
    w_right = 1 - w_left

    indices = jnp.array([ix-1, ix])
    grid_values = jnp.array([grid[ix-1], grid[ix]])
    weights = jnp.array([w_left, w_right])

    return indices, grid_values, weights

@jax.jit
def compute_interpolation_weights_dense(x: float | Array, grid: Array) -> Array:
    """
    Given a grid and a scalar value x, returns a dense row vector of interpolation weights.

    Args:
        x: Scalar value to interpolate (will be clipped to grid bounds)
        grid: 1D array of grid points (must be sorted)

    Returns:
        Array of shape (n,) where n = len(grid), with weights at the two relevant
        indices such that grid @ weights approximates x via linear interpolation.
    """
    x = jnp.clip(x, grid[0], grid[-1])
    ix = jnp.searchsorted(grid, x)
    w_left = jnp.clip((grid[ix] - x) / (grid[ix] - grid[ix-1]), 0, 1)
    w_right = 1 - w_left

    row = jnp.zeros_like(grid)
    row = row.at[ix-1].set(w_left)
    row = row.at[ix].set(w_right)
    return row
