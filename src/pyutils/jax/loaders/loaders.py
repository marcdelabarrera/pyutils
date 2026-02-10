import json
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from pathlib import Path

def to_numpy(obj):
    if isinstance(obj, dict):
        return {k: to_numpy(v) for k,v in obj.items()}
    if isinstance(obj, Array):
        return np.array(obj)
    return obj

def to_jax(obj):
    if isinstance(obj, dict):
        return {k: to_jax(v) for k,v in obj.items()}
    if isinstance(obj, Array):
        return jnp.array(obj)
    return obj

# def save_dict(path, data):
#     data_numpy = to_numpy(data)
#     with open(path, "wb") as f:
#         pickle.dump(data_numpy, f)

def load_dict(path):
    with open(path, "rb") as f:
        data_numpy = pickle.load(f)
    return to_jax(data_numpy)

def save_pytree(path:Path, pytree:dict, overwrite:bool=False, mkdir:bool=False):
    """
    Saves a pytree (dict of dicts of jax.Arrays) to a JSON file
    """
    path = Path(path) if isinstance(path, str) else path

    if path.exists() and overwrite is False:
        raise FileExistsError(f"File {path} already exists. Use overwrite=True to overwrite.")
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    pytree = jax.tree_util.tree_map(
                    lambda x: x.tolist() if isinstance(x, Array) else x,
                    pytree)
    
    with open(path, "w") as f:
        json.dump(pytree, f, indent=4)


def load_pytree(path: Path) -> dict:
    with open(path, "r") as f:
        pytree = json.load(f)

    # Convert *every list* (including nested ones) into jnp.array
    def to_array(x):
        return jnp.array(x) if isinstance(x, list) else x

    # Use tree_map with is_leaf to descend into lists
    return jax.tree_util.tree_map(
        to_array,
        pytree,
        is_leaf=lambda x: not isinstance(x, dict)
    )