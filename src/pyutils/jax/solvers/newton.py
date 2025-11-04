import warnings
from dataclasses import dataclass
import jax
from jax import Array
import jax.numpy as jnp


@dataclass
class NewtonResult:
  x: Array
  fun: Array
  it: int
  success: bool
  grad: Array


class SolutionNotFoundError(Exception):
    """Custom exception for when a solution is not found."""
    pass


def newton_step(f:callable, J:callable, x:Array, config={"steps":{"min":1e-8,"max":4,"n":1000}})->Array:
    delta = jnp.linalg.solve(J(x), -f(x))
    step = jnp.logspace(jnp.log10(config["steps"]["max"]), jnp.log10(config["steps"]["min"]), config["steps"]["n"])
    x_new = x.reshape(-1,1) + step * delta.reshape(-1,1)
    candidates = jnp.linalg.norm(jax.vmap(f, in_axes=1)(x_new), axis=1)
    if jnp.any(jnp.isnan(candidates)):
        candidates = jnp.where(jnp.isnan(candidates), jnp.inf, candidates)
        warnings.warn("NaN encountered in line search candidates, ignoring those candidates.")
    elif jnp.all(jnp.isnan(candidates)):
        raise SolutionNotFoundError("All candidates in line search are NaN.")
    x_new = x_new[:,jnp.argmin(candidates)]
    step_size = step[jnp.argmin(candidates)]
    if jnp.linalg.norm(f(x_new)) > jnp.linalg.norm(f(x)):
        warnings.warn(f"Line search failed: no acceptable step size found and error is {jnp.linalg.norm(f(x))}")
        stop = True
    else:
        stop = False
    return x_new, stop, step_size


def newton_solver(f:callable, x0:Array, tol=1e-6, maxit=100, has_aux=False,verbose=True, **kwargs)->NewtonResult:
    """
    Looks for f(x)=0 by using Newton's method. Finds x such that jnp.linalg.norm(f(x))<tol.
    """
    if not jax.config.read("jax_enable_x64"):
        warnings.warn("JAX is not configured to use 64-bit precision. This may lead to numerical instability in Newton's method. Consider setting jax_enable_x64=True in your JAX configuration.")

    x = x0
    J = jax.jacfwd(f, has_aux = has_aux)
    for it in range(maxit):
        x, stop, step_size = newton_step(f,J, x, **kwargs)
        f_x = f(x)
        if stop:
            break
        if verbose:
            print(f"it={it}, error = {jnp.linalg.norm(f_x)}, step size = {step_size}", end = "\r")
        if jnp.linalg.norm(f_x) < tol:
            break
        if jnp.isnan(jnp.linalg.norm(f_x)):
           raise ValueError("NaN encountered in function evaluation.")
    if it == maxit-1:
        print("Warning: Maximum iterations reached without convergence.")
    return NewtonResult(x=x, fun=f_x, it=it, success= (jnp.linalg.norm(f_x) < tol).item(), grad = J(x))


def inspect_gradient(grad:Array)->None:
    i, j = jnp.unravel_index(jnp.argmin(grad), grad.shape)
    print(f"Min gradient at equation {i}, variable {j} with value {grad[i, j]}")
    i, j = jnp.unravel_index(jnp.argmax(grad), grad.shape)
    print(f"Max gradient at equation {i}, variable {j} with value {grad[i, j]}")
    