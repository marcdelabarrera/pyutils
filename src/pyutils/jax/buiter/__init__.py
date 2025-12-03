import warnings
import jax


if not jax.config.read("jax_enable_x64"):
        warnings.warn("JAX is not configured to use 64-bit precision. This may lead to numerical instability in Newton's method. Consider setting jax_enable_x64=True in your JAX configuration.")

from .buiter import *