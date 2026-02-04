from collections.abc import Callable
from dataclasses import dataclass
import warnings

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import Partial

ALPHA = (jnp.sqrt(5)-1)/2

OptState = dict

@dataclass
class RootResults:
  x: Array
  fun: Array
  it: int
  success: bool

def init_ab(f: Callable[[Array], Array], a, b) -> tuple[Array, Array]:
  '''
  Initializes the lower and upper bounds of the bracket
  '''
  a = jnp.atleast_1d(a).astype(float)
  b = jnp.atleast_1d(b).astype(float)

  if len(a)==1 and len(b)>1:
    a = jnp.ones_like(b)*a
  elif len(b)==1 and len(a)>1:
    b = jnp.ones_like(a)*b
  elif len(a)==len(b)==1:
    a = jnp.ones_like(f(a))*a
    b = jnp.ones_like(f(b))*b
  else:
    return ValueError('TBD') # type: ignore
  if jnp.any(a>b):
    raise ValueError('The lower bound of the bracket must be less than the upper bound')
  return a,b

def init_state(f: Callable[[Array], Array], a: Array, b: Array) -> OptState:
  '''
  '''
  d = ALPHA*(b-a)
  x1 = a+d
  x2 = b-d
  f1 = f(x1)**2
  f2 = f(x2)**2
  return {'d':d,'x1':x1,'x2':x2,'f1':f1,'f2':f2, 'it':0}

def update(f: Callable[[Array], Array], state: OptState) -> OptState:
  a,b = state['a'], state['b']
  d = ALPHA*(b-a)
  x1 = a+d
  x2 = b-d
  f1 = f(x1)**2
  f2 = f(x2)**2
  idx = f1<f2
  a = jnp.where(idx, x2, a)
  b = jnp.where(~idx, x1, b)
  return {"a":a, "b":b, 'it': state['it']+1}

def root_scalar(f: Callable[..., Array],
                a: Array,
                b: Array, tol: float = 1e-5, maxit: int = 100, errors: str = 'warn', **kwargs) -> RootResults:
  """
  Find a root
  """
  if errors not in ['warn','ignore','raise']:
    raise ValueError('The warnings parameter must be either "warn", "ignore" or "raise"')
  f = Partial(f, **kwargs)
  a,b = init_ab(f, a, b)
  if jnp.any(f(a)*f(b)>0):
    if errors=='warn':
        warnings.warn('The function has the same sign at the bounds of the bracket')
    elif errors == 'raise':
        raise ValueError('The function has the same sign at the bounds of the bracket')
  state = {"a":a, "b":b, 'it':0}
  opt_results = jax.lax.while_loop(cond_fun = lambda x: (x['it']<maxit)*(jnp.max(x['b']-x['a'])>tol),
                            body_fun = lambda x: update(f, x), 
                            init_val = state)
  
  x = (opt_results['a']+opt_results['b'])/2
  x = x.at[jnp.abs(f(x))>tol].set(jnp.nan)
  if jnp.any(jnp.isnan(x)) and errors=='warn':
    warnings.warn('The function did not converge')
  return RootResults(x = x,
                     fun = f(x),
                     it = opt_results['it'],
                     success = bool(jnp.all(jnp.abs(f(x))<tol)))