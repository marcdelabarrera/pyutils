from jax import Array
import jax.numpy as jnp
from jax.experimental.sparse import BCOO


def spdiagm(x:Array, k:int=0) -> BCOO:
    n = x.shape[0]+abs(k)
    if k >= 0:
        row = jnp.arange(n-k)
        col = jnp.arange(k, n)
    else:
        row = jnp.arange(abs(k), n)
        col = jnp.arange(n-abs(k))
    return BCOO((x, jnp.column_stack((row, col))), shape=(n, n))

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

def compute_forward_derivative(x:Array) -> BCOO:
    return spdiagm(1/jnp.diff(x), k = 1) - spdiagm(jnp.concatenate((1/jnp.diff(x), jnp.array([0]))))

def compute_backward_derivative(x:Array) -> BCOO:
    return spdiagm(jnp.concatenate((jnp.array([0]), 1/jnp.diff(x)))) - spdiagm(1/jnp.diff(x), k=-1)

def compute_second_derivative(x:Array) -> BCOO:
    n = x.shape[0]
    dx = jnp.diff(x)
    dx2 = jnp.concatenate((dx[:1],dx))*jnp.concatenate((dx,dx[-1:]))
    return spdiagm(1/dx2[:-1], k=1) + spdiagm(1/dx2[1:], k=-1) - spdiagm(jnp.full(n,2).at[0].set(1).at[-1].set(1)/dx2)

def compute_D_x(x:Array, y:Array, direction:str)-> BCOO:
    if direction == 'forward':
        return kron(compute_forward_derivative(x), jnp.eye(y.shape[0]))
    elif direction == 'backward':
        return kron(compute_backward_derivative(x), jnp.eye(y.shape[0]))
    else:
        raise ValueError("Direction must be 'forward' or 'backward'")
    
def compute_D_y(x:Array, y:Array, direction:str)-> BCOO:
    if direction == 'forward':
        return kron(jnp.eye(x.shape[0]), compute_forward_derivative(y))
    elif direction == 'backward':
        return kron(jnp.eye(x.shape[0]), compute_backward_derivative(y))
    else:
        raise ValueError("Direction must be 'forward' or 'backward'")
    
def compute_D_xx(x:Array, y:Array)-> BCOO:
    return kron(compute_second_derivative(x), jnp.eye(y.shape[0]))

def compute_D_yy(x:Array, y:Array)-> BCOO:
    return kron(jnp.eye(x.shape[0]), compute_second_derivative(y))

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