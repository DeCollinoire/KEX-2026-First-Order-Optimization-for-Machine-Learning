
import jax
import jax.numpy as jnp

def rosenbrock(x):
    """
    Computes the Rosenbrock function
    Input: x - vector of values
    Output: Result of Rosenbrock's function
    """
    return jnp.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

