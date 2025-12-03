import numpy as np
import jax.numpy as jnp


def rescale_nodes(nodes, domain):
    return 0.5 * (domain[0] - domain[1]) * nodes + 0.5 * (domain[1] + domain[0])

def chebyshev_nodes(n, domain = (-1.0, 1.0), type='first-kind')->jnp.ndarray:
    """
    first-kind includes the endpoints
    second-kind does not include the endpoints
    """
    if type == 'first-kind':
        x = jnp.cos(jnp.pi * jnp.arange(n) / (n-1))
    elif type == 'second-kind':
        x = jnp.cos((2 * jnp.arange(n) + 1) * jnp.pi / (2 * n))
    else:
        raise ValueError("Invalid type. Use 'first-kind' or 'second-kind'.")
    x = rescale_nodes(x, domain)
    return x


def clenshaw_curtis_weights(n, domain=(-1., 1.)):
    theta = (jnp.pi * jnp.arange(n) / (n - 1)).reshape(-1,1)
    m_max = (n - 1) // 2
    m = jnp.arange(1, m_max + 1).reshape(-1, 1) 
    coeffs = 2 / (1 - 4 * m**2)                 
    cos_terms = jnp.cos(2 * m * theta.T)         
    s = jnp.sum(coeffs * cos_terms, axis=0)    
    w = (2 / (n - 1)) * (1 + s)
    w = w.at[0].set(w[0] / 2)
    w = w.at[-1].set(w[-1] / 2)
    return 0.5 * (domain[1] - domain[0]) * w  

def chebyshev_diff_matrix(n, domain=(-1.0, 1.0), type='first-kind'):
    if n == 0:
        x = jnp.array([1.0])
        D = jnp.array([[0.0]])
        return D, x

    x = chebyshev_nodes(n, type=type)
    X = x[:, None]
    dX = X - X.T
    c = jnp.ones(n)
    c = c.at[0].set(2.0)
    c = c.at[-1].set(2.0)
    c = c * (-1.0) ** jnp.arange(n)
    if type == 'first-kind':        
        C = (c[:, None] / c[None, :]) / (dX + jnp.eye(n))  # Off-diagonal
        D = C - jnp.diag(jnp.sum(C, axis=1))                  # Diagonal
    elif type == 'second-kind':
        C = c[:, None] / c[None, :]
        D = (C / (dX + jnp.eye(n))) - jnp.diag(jnp.sum(C / (dX + jnp.eye(n)), axis=1))
    else:
        raise ValueError("Invalid type. Use 'first-kind' or 'second-kind'.")
    
    D = (2.0 / (domain[1] - domain[0])) * D  # Scale for arbitrary domain
    return D



def chebyshev_eval_1d(x, coeffs, domain):
    """
    Evaluate Chebyshev interpolant at point x using Clenshaw's algorithm.
    coeffs: Chebyshev coefficients (from values via DCT or similar)
    domain: (a, b) tuple for original function domain
    """
    x_hat = 2.0 * (x - domain[0]) / (domain[1] - domain[0]) - 1.0
    b_kp1 = 0.0
    b_k = 0.0
    x2 = 2.0 * x_hat
    for a_k in reversed(coeffs[1:]):
        b_km1 = a_k + x2 * b_k - b_kp1
        b_kp1 = b_k
        b_k = b_km1
    return coeffs[0] + x_hat * b_k - b_kp1




# def evaluate_chebyshev(X:np.ndarray, n:int=None) -> np.ndarray:
#     """
#     Evaluate the first n Chebyshev polynomials for all values in matrix X.

#     Parameters
#     ----------
#     X : np.ndarray
#         Input array.
#     n : int
#         Number of Chebyshev polynomials to evaluate.

#     Returns
#     -------
#     np.ndarray
#         Evaluated Chebyshev polynomials. T[n,i] = T_n(x_i). That is, 
#         the n-th row of T_n corresponds to the n-th Chebyshev polynomial
#     """
#     n = len(X) if n is None else n
#     T_n = np.empty((len(X), n))
#     T_n[:, 0] = 1
#     T_n[:, 1] = X

#     for i in range(2, n):
#         T_n[:, i] = 2 * X * T_n[:, i - 1] - T_n[:, i - 2]

#     return T_n


# function U_n_store = evaluate_chebyshev_second_degree(X,n)

# % This function evaluates the first n chebyshev polynomials for all values
# % in matrix X

# U_n_store = NaN([length(X),n]);
# U_n_store(:,1) = ones(size(X));
# U_n_store(:,2) = 2*X;

# for i = 3:n
#     U_n_store(:,i) = 2*X.*U_n_store(:,i-1)-U_n_store(:,i-2);
# end

# end

def evaluate_chebyshev_second_degree(X:np.ndarray, n:int=None) -> np.ndarray:
    """
    Evaluate the first n Chebyshev polynomials of the second kind for all values in matrix X.

    Parameters
    ----------
    X : np.ndarray
        Input array.
    n : int
        Number of Chebyshev polynomials to evaluate.

    Returns
    -------
    np.ndarray
        Evaluated Chebyshev polynomials of the second kind.
    """
    n = len(X) if n is None else n
    U_n = np.empty((len(X), n))
    U_n[:, 0] = 1
    U_n[:, 1] = 2 * X

    for i in range(2, n):
        U_n[:, i] = 2 * X * U_n[:, i - 1] - U_n[:, i - 2]

    return U_n

def chebyshev_first_derivative(U_n):
    N = U_n.shape[1]
    return np.arange(N)*np.column_stack([np.zeros(N),U_n[:,0:-1]])


def chebyshev_second_derivative(n):

    x = chebyshev_nodes(n)
    T_n = evaluate_chebyshev(x)
    U_n = evaluate_chebyshev_second_degree(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        T_n_prime_prime = n*((n+1)*T_n - U_n)/(x.reshape(-1,1)**2-1)

    T_n_prime_prime[0] = (n**4-n**2)/3
    T_n_prime_prime[-1] = (-1)**n*(n**4-n**2)/3
    return T_n_prime_prime

