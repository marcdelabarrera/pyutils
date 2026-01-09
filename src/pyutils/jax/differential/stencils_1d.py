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
    """
    Computes the matrix D2 such that D2@f = d2f/dx2 using centered differences.
    For interior points i (1 <= i <= n-2), uses the 3-point stencil:

    d2f_i/dx2 = 2 * [f_{i-1}/(h_L*(h_L+h_R)) - f_i/(h_L*h_R) + f_{i+1}/(h_R*(h_L+h_R))]

    where h_L = x_i - x_{i-1} and h_R = x_{i+1} - x_i.

    Boundary points use one-sided stencils.
    """
    dx = jnp.diff(x)  # dx[i] = x[i+1] - x[i], shape (n-1,)

    # For interior points, compute the correct coefficients
    # h_L[i] = dx[i-1], h_R[i] = dx[i]
    h_L = dx[:-1]  # shape (n-2,), corresponds to points i=1,...,n-2
    h_R = dx[1:]   # shape (n-2,), corresponds to points i=1,...,n-2

    # Coefficients for the centered 3-point stencil (interior points)
    # Multiply by 2 for the standard second derivative formula
    coef_left = 2.0 / (h_L * (h_L + h_R))      # coefficient for f_{i-1}
    coef_center = -2.0 / (h_L * h_R)            # coefficient for f_i
    coef_right = 2.0 / (h_R * (h_L + h_R))      # coefficient for f_{i+1}

    # Build diagonal coefficients for the full matrix
    # Main diagonal: [boundary_left, coef_center[0], ..., coef_center[-1], boundary_right]
    diag_main = jnp.concatenate([
        jnp.array([-1.0 / dx[0]]),  # boundary at i=0
        coef_center,
        jnp.array([-1.0 / dx[-1]])  # boundary at i=n-1
    ])

    # Upper diagonal (k=1): [boundary_left, coef_right[0], ..., coef_right[-1]]
    diag_upper = jnp.concatenate([
        jnp.array([1.0 / dx[0]]),   # boundary at i=0
        coef_right
    ])

    # Lower diagonal (k=-1): [coef_left[0], ..., coef_left[-1], boundary_right]
    diag_lower = jnp.concatenate([
        coef_left,
        jnp.array([1.0 / dx[-1]])   # boundary at i=n-1
    ])

    return spdiagm(diag_upper, k=1) + spdiagm(diag_lower, k=-1) + spdiagm(diag_main, k=0)





@jax.jit
def compute_interpolation_weights(x: float | Array, grid: Array) -> tuple[Array, Array]:
    """
    Given a grid and a scalar value x, returns the interpolation indices and weights
    for linear interpolation between the two grid points surrounding x.

    Args:
        x: Scalar value to interpolate (will be clipped to grid bounds)
        grid: 1D array of grid points (must be sorted)

    Returns:
        Tuple of (indices, weights) where:
        - indices: Array of shape (2,) with left and right indices
        - weights: Array of shape (2,) with interpolation weights summing to 1
    """
    x = jnp.clip(x, grid[0], grid[-1])
    ix = jnp.searchsorted(grid, x, side='right')
    ix = jnp.clip(ix, 1, len(grid) - 1)  # Ensure ix is in [1, n-1] to avoid negative indices

    ix_left = ix - 1
    ix_right = ix

    # Linear interpolation weights
    dx = grid[ix_right] - grid[ix_left]
    w_right = jnp.where(dx > 0, (x - grid[ix_left]) / dx, 1.0)
    w_left = 1.0 - w_right

    indices = jnp.array([ix_left, ix_right])
    weights = jnp.array([w_left, w_right])

    return indices, weights

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
    ix = jnp.searchsorted(grid, x, side='right')
    ix = jnp.clip(ix, 1, len(grid) - 1)  # Ensure ix is in [1, n-1] to avoid negative indices

    ix_left = ix - 1
    ix_right = ix

    # Linear interpolation weights
    dx = grid[ix_right] - grid[ix_left]
    w_right = jnp.where(dx > 0, (x - grid[ix_left]) / dx, 1.0)
    w_left = 1.0 - w_right

    row = jnp.zeros_like(grid)
    row = row.at[ix_left].set(w_left)
    row = row.at[ix_right].set(w_right)
    return row


def compute_interpolation_weights_2d(x, y, grid_x: Array, grid_y: Array):
    """
    Compute 2D bilinear interpolation weights using 1D interpolation.
    
    Returns:
        indices: Array of flat indices (length 4) for the surrounding grid points
        weights: Array of weights (length 4) that sum to 1
    
    Grid flattening: uses ravel_multi_index with C-order
    """
    # Get 1D interpolation for each dimension
    ix_indices, ix_weights = compute_interpolation_weights(x, grid_x)
    iy_indices, iy_weights = compute_interpolation_weights(y, grid_y)
    
    # Combine into 2D: we have 2x2 = 4 points
    # Create all combinations of (ix, iy) pairs
    ix_grid = jnp.array([ix_indices[0], ix_indices[0], ix_indices[1], ix_indices[1]])
    iy_grid = jnp.array([iy_indices[0], iy_indices[1], iy_indices[0], iy_indices[1]])
    
    # Flatten 2D indices to 1D
    indices = jnp.ravel_multi_index((ix_grid, iy_grid), (len(grid_x), len(grid_y)), mode="clip")
    
    # Combine weights using outer product structure
    weights = jnp.array([
        ix_weights[0] * iy_weights[0],
        ix_weights[0] * iy_weights[1],
        ix_weights[1] * iy_weights[0],
        ix_weights[1] * iy_weights[1]
    ])
    
    return indices, weights