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

def speye(n:int, **kwargs)->BCOO:
    return BCOO._eye(n,n,k=0, **kwargs)

def spzeros(shape:tuple, **kwargs) -> BCOO:
    if isinstance(shape, int):
        shape = (shape, shape)
    elif len(shape) == 1:
        shape = (shape[0], shape[0])
    return BCOO._empty(shape, **kwargs)


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