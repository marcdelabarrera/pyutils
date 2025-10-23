import jax
from jax import Array
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, eye, empty


def vvmap(fun, in_axes):
    return jax.vmap(jax.vmap(fun, in_axes = in_axes), in_axes = in_axes)

def compute_vec_index(index:tuple, shape:tuple, order = "C")-> int:
    if jnp.any(jnp.array(index) >= jnp.array(shape)):
        raise ValueError("Index out of bounds")
    if order == "C":
        for i in range(len(index)):
            if i == 0:
                vec_index = index[-1]
            else:
                vec_index += index[-(i+1)] * jnp.prod(jnp.array(shape[-i:]))
        return int(vec_index)


def spdiagm(x:Array, k:int=0) -> BCOO:
    n = x.shape[0]+abs(k)
    if k >= 0:
        row = jnp.arange(n-k)
        col = jnp.arange(k, n)
    else:
        row = jnp.arange(abs(k), n)
        col = jnp.arange(n-abs(k))
    return BCOO((x, jnp.column_stack((row, col))), shape=(n, n))

def speye(n:int, **kwargs)->BCOO:
    return eye(n, **kwargs)

def spzeros(shape:tuple, **kwargs) -> BCOO:
    if isinstance(shape, int):
        shape = (shape, shape)
    elif len(shape) == 1:
        shape = (shape[0], shape[0])
    return empty(shape, **kwargs)



def kron(A: BCOO, B: BCOO) -> BCOO:
    """Kronecker producte de dues matrius disperses BCOO."""
    (m, n) = A.shape
    (p, q) = B.shape
    idxA, valA = A.indices, A.data
    idxB, valB = B.indices, B.data      

    IA, IB = jnp.meshgrid(jnp.arange(idxA.shape[0]), jnp.arange(idxB.shape[0]), indexing="ij")
    IA = IA.reshape(-1)
    IB = IB.reshape(-1)

    new_rows = idxA[IA, 0] * p + idxB[IB, 0]
    new_cols = idxA[IA, 1] * q + idxB[IB, 1]
    new_idx = jnp.stack([new_rows, new_cols], axis=1)
    
    new_data = valA[IA] * valB[IB]

    return BCOO((new_data, new_idx), shape=(m * p, n * q))

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
        return spdiagm(1/jnp.diff(x), k = 1) - spdiagm(jnp.concatenate((1/jnp.diff(x), 1/(x[-1]-x[-2]))))
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
        return spdiagm(jnp.concatenate((1/(x[1]-x[0]), 1/jnp.diff(x)))) - spdiagm(1/jnp.diff(x), k=-1)
    else:
        return spdiagm(jnp.concatenate((jnp.array([0]), 1/jnp.diff(x)))) - spdiagm(1/jnp.diff(x), k=-1)

def compute_second_derivative(x:Array) -> BCOO:
    n = x.shape[0]
    dx = jnp.diff(x)
    dx2 = jnp.concatenate((dx[:1],dx))*jnp.concatenate((dx,dx[-1:]))
    return spdiagm(1/dx2[:-1], k=1) + spdiagm(1/dx2[1:], k=-1) - spdiagm(jnp.full(n,2).at[0].set(1).at[-1].set(1)/dx2)

def compute_D_x(x:Array, y:Array, direction:str)-> BCOO:
    if direction == 'forward':
        return kron(compute_forward_derivative(x), eye(y.shape[0]))
    elif direction == 'backward':
        return kron(compute_backward_derivative(x), eye(y.shape[0]))
    else:
        raise ValueError("Direction must be 'forward' or 'backward'")
    
def compute_D_y(x:Array, y:Array, direction:str)-> BCOO:
    if direction == 'forward':
        return kron(eye(x.shape[0]), compute_forward_derivative(y))
    elif direction == 'backward':
        return kron(eye(x.shape[0]), compute_backward_derivative(y))
    else:
        raise ValueError("Direction must be 'forward' or 'backward'")
    
def compute_D_xx(x:Array, y:Array|None=None)-> BCOO:
    if y is None:
        return compute_second_derivative(x)
    else:
        return kron(compute_second_derivative(x), eye(y.shape[0]))

def compute_D_yy(x:Array, y:Array)-> BCOO:
    return kron(eye(x.shape[0]), compute_second_derivative(y))

def compute_D_xy(x:Array, y:Array, direction_x:str, direction_y:str)-> BCOO:
    if direction_x == 'forward' and direction_y == 'forward':
        return kron(compute_forward_derivative(x), compute_forward_derivative(y))
    elif direction_x == 'backward' and direction_y == 'backward':
        return kron(compute_backward_derivative(x), compute_backward_derivative(y))
    elif direction_x == 'forward' and direction_y == 'backward':
        return kron(compute_forward_derivative(x), compute_backward_derivative(y))
    elif direction_x == 'backward' and direction_y == 'forward':
        return kron(compute_backward_derivative(x), compute_forward_derivative(y))
    else:
        raise ValueError("Directions must be 'forward' or 'backward'")
    

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
    D_xy = compute_D_xy(x, y, direction_x, direction_y)
    f_xy = (D_xy @ f.flatten()).reshape(f.shape)

    return  jnp.stack((jnp.stack((f_xx, f_xy), axis=-1),
                    jnp.stack((f_xy, f_yy), axis=-1)), axis=-1)


# def build_D_xy(x: Array, y: Array) -> BCOO:
#     shape = (len(x), len(y))
#     D_row = jnp.zeros(4*len(x)*len(y), dtype=int)
#     D_col = jnp.zeros(4*len(x)*len(y), dtype=int)
#     D_val = jnp.zeros(4*len(x)*len(y))
#     row_index, k = 0, 0
#     for ix in range(len(x)):
#         for iy in range(len(y)):
#             if 0<ix<(len(x)-1) and 0<iy<(len(y)-1):
#                 denom = (x[ix+1]-x[ix-1])*(y[iy+1]-y[iy-1])
#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy+1), shape)
#                 D_val[k] = 1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy-1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy+1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy-1), shape)
#                 D_val[k] = 1/denom
#                 k += 1
#             elif ix == 0 and 0<iy<(len(y)-1):
#                 denom = (x[ix+1]-x[ix])*(y[iy+1]-y[iy-1])
#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix, iy+1), shape)
#                 D_val[k] = 1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix, iy-1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy+1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy-1), shape)
#                 D_val[k] = 1/denom
#                 k += 1
#             elif ix == (len(x)-1) and 0<iy<(len(y)-1):
#                 denom = (x[ix]-x[ix-1])*(y[iy+1]-y[iy-1])
#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix, iy+1), shape)
#                 D_val[k] = 1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix, iy-1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy+1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy-1), shape)
#                 D_val[k] = 1/denom
#                 k += 1
#             elif 0<ix<(len(x)-1) and iy == 0:
#                 denom = (x[ix+1]-x[ix-1])*(y[iy+1]-y[iy])
#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy), shape)
#                 D_val[k] = 1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy+1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy+1), shape)
#                 D_val[k] = 1/denom
#                 k += 1
#             elif 0<ix<(len(x)-1) and iy == (len(y)-1):
#                 denom = (x[ix+1]-x[ix-1])*(y[iy]-y[iy-1])
#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy), shape)
#                 D_val[k] = 1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy-1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy-1), shape)
#                 D_val[k] = 1/denom
#                 k += 1
#             elif ix == 0 and iy == 0:
#                 denom = (x[ix+1]-x[ix])*(y[iy+1]-y[iy])
#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix, iy+1), shape)
#                 D_val[k] = 1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy+1), shape)
#                 D_val[k] = 1/denom
#                 k += 1
#             elif ix == 0 and iy == (len(y)-1):
#                 denom = (x[ix+1]-x[ix])*(y[iy]-y[iy-1])
#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix, iy-1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix+1, iy-1), shape)
#                 D_val[k] = 1/denom
#                 k += 1
#             elif ix == (len(x)-1) and iy == 0:  
#                 denom = (x[ix]-x[ix-1])*(y[iy+1]-y[iy])
#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix, iy+1), shape)
#                 D_val[k] = 1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy+1), shape)
#                 D_val[k] = 1/denom
#                 k += 1
#             elif ix == (len(x)-1) and iy == (len(y)-1):
#                 denom = (x[ix]-x[ix-1])*(y[iy]-y[iy-1])
#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix, iy-1), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy), shape)
#                 D_val[k] = -1/denom
#                 k += 1

#                 D_row[k] = row_index
#                 D_col[k] = compute_vec_index((ix-1, iy-1), shape)
#                 D_val[k] = 1/denom
#                 k += 1
#             row_index += 1  
#     return BCOO((D_val[:k], jnp.column_stack((D_row[:k], D_col[:k]))), shape=(len(x)*len(y), len(x)*len(y)))


def build_D_xy(x, y) -> BCOO:
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
    mat = BCOO((D_val[:k], idx), shape=(n, n))
    # Opcional: combina duplicats (pot aparèixer a les vores)
    # mat = mat.sum_duplicates()
    return mat


