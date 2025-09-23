import pickle
import jax.numpy as jnp
import numpy as np
from jax import Array

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

def save_dict(path, data):
    data_numpy = to_numpy(data)
    with open(path, "wb") as f:
        pickle.dump(data_numpy, f)

def load_dict(path):
    with open(path, "rb") as f:
        data_numpy = pickle.load(f)
    return to_jax(data_numpy)