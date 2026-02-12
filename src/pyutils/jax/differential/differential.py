from typing import Callable
from collections.abc import Sequence

import jax
from jax import Array
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from .sparse import speye, kron
from .stencils_1d import compute_forward_derivative, compute_backward_derivative, compute_second_derivative

def vvmap(fun: Callable, in_axes: int | None | Sequence[int | None] = 0, out_axes: int | None | Sequence[int | None] = 0) -> Callable:
    return jax.vmap(jax.vmap(fun, in_axes = in_axes, out_axes = out_axes), in_axes = in_axes, out_axes = out_axes)

def compute_vec_index(index:tuple, shape:tuple, order = "C")-> Array:
    return jnp.ravel_multi_index(index, shape, order=order).astype(jnp.int32)

def compute_D_x(x:Array, y:Array, direction:str, ghost_node:bool=False)-> BCOO:
    """
    Computes the matrix D such that D@f = df/dx using finite differences. Given a vector
    x[i] for i = 0,...,n-1 and a function f defined on those points, then  df_i/dx = (f_{i+1} - f_{i}) / (x_{i+1} - x_{i}) for forward differences,
    and df_i/dx = (f_i - f_{i-1}) / (x_i - x_{i-1}) for backward differences."""
    if direction == 'forward':
        return kron(compute_forward_derivative(x, ghost_node=ghost_node), speye(y.shape[0]))
    elif direction == 'backward':
        return kron(compute_backward_derivative(x, ghost_node=ghost_node), speye(y.shape[0]))
    else:
        raise ValueError("Direction must be 'forward' or 'backward'")

def compute_D_y(x:Array, y:Array, direction:str, ghost_node:bool=False)-> BCOO:
    """
    Computes the matrix D such that D@f = df/dy using finite differences. Given a vector
    y[j] for j = 0,...,m-1 and a function f defined on those points, then  df_j/dy = (f_{j+1} - f_{j}) / (y_{j+1} - y_{j}) for forward differences,
    and df_j/dy = (f_j - f_{j-1}) / (y_j - y_{j-1}) for backward differences.
    """

    if direction == 'forward':
        return kron(speye(x.shape[0]), compute_forward_derivative(y, ghost_node=ghost_node))
    elif direction == 'backward':
        return kron(speye(x.shape[0]), compute_backward_derivative(y, ghost_node=ghost_node))
    else:
        raise ValueError("Direction must be 'forward' or 'backward'")
    
def compute_D_xx(x:Array, y:Array)-> BCOO:
    return kron(compute_second_derivative(x), speye(y.shape[0]))

def compute_D_yy(x:Array, y:Array)-> BCOO:
    return kron(speye(x.shape[0]), compute_second_derivative(y))

# def compute_D_xy(x:Array, y:Array, direction_x:str, direction_y:str, ghost_node:bool=False)-> BCOO:
#     if direction_x == 'forward' and direction_y == 'forward':
#         return kron(compute_forward_derivative(x, ghost_node=ghost_node), compute_forward_derivative(y, ghost_node=ghost_node))
#     elif direction_x == 'backward' and direction_y == 'backward':
#         return kron(compute_backward_derivative(x, ghost_node=ghost_node), compute_backward_derivative(y, ghost_node=ghost_node))
#     elif direction_x == 'forward' and direction_y == 'backward':
#         return kron(compute_forward_derivative(x, ghost_node=ghost_node), compute_backward_derivative(y, ghost_node=ghost_node))
#     elif direction_x == 'backward' and direction_y == 'forward':
#         return kron(compute_backward_derivative(x, ghost_node=ghost_node), compute_forward_derivative(y, ghost_node=ghost_node))
#     else:
#         raise ValueError("Directions must be 'forward' or 'backward'")
    

def compute_gradient(f:Array, x:Array, y:Array, direction_x:str, direction_y:str)-> Array:
    if f.shape != (x.shape[0], y.shape[0]):
        raise ValueError("f must be of shape (len(x), len(y))")
    if direction_x not in ['forward', 'backward']:
        raise ValueError("direction_x must be 'forward' or 'backward'")
    if direction_y not in ['forward', 'backward']:
        raise ValueError("direction_y must be 'forward' or 'backward'")
    
    return jnp.stack(((compute_D_x(x, y, direction_x)@f.flatten()).reshape(f.shape),
                      (compute_D_y(x, y, direction_y)@f.flatten()).reshape(f.shape)), axis = -1)

def compute_hessian(f:Array, x:Array, y:Array, direction_x:str, direction_y:str)->Array:
    if f.shape != (x.shape[0], y.shape[0]):
        raise ValueError("f must be of shape (len(x), len(y))")
    if direction_x not in ['forward', 'backward']:
        raise ValueError("direction_x must be 'forward' or 'backward'")
    if direction_y not in ['forward', 'backward']:
        raise ValueError("direction_y must be 'forward' or 'backward'")
    D_xx = compute_D_xx(x, y)
    f_xx = (D_xx @ f.flatten()).reshape(f.shape)
    D_yy = compute_D_yy(x, y)
    f_yy = (D_yy @ f.flatten()).reshape(f.shape)
    D_xy = compute_D_xy(x, y)
    f_xy = (D_xy @ f.flatten()).reshape(f.shape)

    return  jnp.stack((jnp.stack((f_xx, f_xy), axis=-1),
                    jnp.stack((f_xy, f_yy), axis=-1)), axis=-1)


def compute_D_xy(x:Array, y:Array) -> BCOO:
    """
    Computes D_xy such that D_xy @ f = d^2f/dxdy using a 5-point stencil.
    For interior points, the stencil is:D_xy f_{ij} = (f_{i+1,j+1} - f_{i+1,j-1} - f_{i-1,j+1} + f_{i-1,j-1}) / (4 * dx * dy)where dx = x_{i+1} - x_{i-1} and dy   
    """

    x = jnp.asarray(x); y = jnp.asarray(y)
    nx, ny = int(x.size), int(y.size)
    shape2d = (nx, ny); n = nx * ny

    cap = 4 * n
    D_row = jnp.zeros(cap, dtype=jnp.int32)
    D_col = jnp.zeros(cap, dtype=jnp.int32)
    D_val = jnp.zeros(cap, dtype=jnp.float32)

    k = 0
    for ix in range(nx):
        il = max(ix - 1, 0)
        ir = min(ix + 1, nx - 1)
        dx = float(x[ir] - x[il])
        for iy in range(ny):
            jl = max(iy - 1, 0)
            jr = min(iy + 1, ny - 1)
            dy = float(y[jr] - y[jl])

            if dx == 0.0 or dy == 0.0:
                continue  # malla degenerada al node

            denom = dx * dy
            row = compute_vec_index((ix, iy), shape2d)

            # (+) f_{ir,jr}
            D_row = D_row.at[k].set(row)
            D_col = D_col.at[k].set(compute_vec_index((ir, jr), shape2d))
            D_val = D_val.at[k].set( 1.0 / denom); k += 1

            # (-) f_{ir,jl}
            D_row = D_row.at[k].set(row)
            D_col = D_col.at[k].set(compute_vec_index((ir, jl), shape2d))
            D_val = D_val.at[k].set(-1.0 / denom); k += 1

            # (-) f_{il,jr}
            D_row = D_row.at[k].set(row)
            D_col = D_col.at[k].set(compute_vec_index((il, jr), shape2d))
            D_val = D_val.at[k].set(-1.0 / denom); k += 1

            # (+) f_{il,jl}
            D_row = D_row.at[k].set(row)
            D_col = D_col.at[k].set(compute_vec_index((il, jl), shape2d))
            D_val = D_val.at[k].set( 1.0 / denom); k += 1

    idx = jnp.column_stack((D_row[:k], D_col[:k]))
    D_xy = BCOO((D_val[:k], idx), shape=(n, n))
    return D_xy


def compute_D_xi(x: Array, y: Array) -> BCOO:
    """Second derivative along the (1,1) diagonal direction.

    Stencil at interior point (i,j):
        D_xi V_{ij} = c_- V_{i-1,j-1} + c_0 V_{ij} + c_+ V_{i+1,j+1}

    where c_-, c_+ > 0, c_0 < 0, and c_- + c_0 + c_+ = 0 (M-matrix).

    This approximates:
        (h_x^2 V_xx + 2 h_x h_y V_xy + h_y^2 V_yy) / (h_x^2 + h_y^2)

    Boundary rows are zero.
    """
    x = jnp.asarray(x); y = jnp.asarray(y)
    nx, ny = int(x.size), int(y.size)
    n = nx * ny
    shape2d = (nx, ny)

    ix = jnp.arange(1, nx - 1)
    iy = jnp.arange(1, ny - 1)
    IX, IY = jnp.meshgrid(ix, iy, indexing='ij')
    IX = IX.ravel()
    IY = IY.ravel()

    # Arc-length step sizes along (1,1) diagonal
    s_plus = jnp.sqrt((x[IX + 1] - x[IX])**2 + (y[IY + 1] - y[IY])**2)
    s_minus = jnp.sqrt((x[IX] - x[IX - 1])**2 + (y[IY] - y[IY - 1])**2)

    coef_bwd = 2.0 / (s_minus * (s_plus + s_minus))
    coef_ctr = -2.0 / (s_plus * s_minus)
    coef_fwd = 2.0 / (s_plus * (s_plus + s_minus))

    row = jnp.ravel_multi_index((IX, IY), shape2d)
    col_bwd = jnp.ravel_multi_index((IX - 1, IY - 1), shape2d)
    col_fwd = jnp.ravel_multi_index((IX + 1, IY + 1), shape2d)

    rows = jnp.concatenate([row, row, row])
    cols = jnp.concatenate([col_bwd, row, col_fwd])
    vals = jnp.concatenate([coef_bwd, coef_ctr, coef_fwd])

    indices = jnp.column_stack([rows, cols])
    return BCOO((vals, indices), shape=(n, n))


def compute_D_eta(x: Array, y: Array) -> BCOO:
    """Second derivative along the (1,-1) anti-diagonal direction.

    Stencil at interior point (i,j):
        D_eta V_{ij} = c_- V_{i-1,j+1} + c_0 V_{ij} + c_+ V_{i+1,j-1}

    where c_-, c_+ > 0, c_0 < 0, and c_- + c_0 + c_+ = 0 (M-matrix).

    This approximates:
        (h_x^2 V_xx - 2 h_x h_y V_xy + h_y^2 V_yy) / (h_x^2 + h_y^2)

    Boundary rows are zero.
    """
    x = jnp.asarray(x); y = jnp.asarray(y)
    nx, ny = int(x.size), int(y.size)
    n = nx * ny
    shape2d = (nx, ny)

    ix = jnp.arange(1, nx - 1)
    iy = jnp.arange(1, ny - 1)
    IX, IY = jnp.meshgrid(ix, iy, indexing='ij')
    IX = IX.ravel()
    IY = IY.ravel()

    # Arc-length step sizes along (1,-1) anti-diagonal
    s_plus = jnp.sqrt((x[IX + 1] - x[IX])**2 + (y[IY] - y[IY - 1])**2)
    s_minus = jnp.sqrt((x[IX] - x[IX - 1])**2 + (y[IY + 1] - y[IY])**2)

    coef_bwd = 2.0 / (s_minus * (s_plus + s_minus))
    coef_ctr = -2.0 / (s_plus * s_minus)
    coef_fwd = 2.0 / (s_plus * (s_plus + s_minus))

    row = jnp.ravel_multi_index((IX, IY), shape2d)
    col_bwd = jnp.ravel_multi_index((IX - 1, IY + 1), shape2d)
    col_fwd = jnp.ravel_multi_index((IX + 1, IY - 1), shape2d)

    rows = jnp.concatenate([row, row, row])
    cols = jnp.concatenate([col_bwd, row, col_fwd])
    vals = jnp.concatenate([coef_bwd, coef_ctr, coef_fwd])

    indices = jnp.column_stack([rows, cols])
    return BCOO((vals, indices), shape=(n, n))





def build_differential_matrices(x:Array, y:Array, ghost_node:bool=False)->tuple:
    """
    Computes the differential matrices for 2D grid defined by vectors x and y.
    """
    D_x_forward = compute_D_x(x, y, 'forward', ghost_node=ghost_node) 
    D_x_backward = compute_D_x(x, y, 'backward', ghost_node=ghost_node)
    D_y_forward = compute_D_y(x, y, 'forward', ghost_node=ghost_node)
    D_y_backward = compute_D_y(x, y, 'backward', ghost_node=ghost_node)
    D_xx = compute_D_xx(x, y)
    D_yy = compute_D_yy(x, y)
    D_xi = compute_D_xi(x, y)
    D_eta = compute_D_eta(x, y)
    D_xy = compute_D_xy(x, y)
    return (D_x_forward, D_x_backward, D_y_forward, D_y_backward,
            D_xx, D_yy, D_xi, D_eta, D_xy)



def is_monotonic(A: Array)-> bool:
    diag = jnp.diag(A)
    off_diag = A - jnp.diag(diag)
    return bool(jnp.all(diag <= 0)) and bool(jnp.all(off_diag >= 0))


