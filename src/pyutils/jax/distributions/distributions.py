from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm
from jax.scipy.special import gamma


@dataclass
class BrownianWithReset:
    mu: float
    sigma: float
    chi: float
    x_entry: float


    def pdf(self, x:Array)->Array:
        mu, sigma, chi, x_entry = self.mu, self.sigma, self.chi, self.x_entry
        r_plus = (mu + jnp.sqrt(mu**2 + 2*sigma**2*chi)) / sigma**2
        r_minus = (mu - jnp.sqrt(mu**2 + 2*sigma**2*chi)) / sigma**2
        coeff = chi / jnp.sqrt(mu**2 + 2*sigma**2*chi)
        return jnp.where(x < x_entry, coeff * jnp.exp(r_plus * (x - x_entry)),
                                   coeff * jnp.exp(r_minus * (x - x_entry)))


@dataclass
class OrnsteinUhlenbeck:
   theta: float
   sigma:float

   def cdf(self, x:Array)->Array:
       theta, sigma = self.theta, self.sigma
       return norm.cdf(x, 0, jnp.sqrt(sigma**2/(2*theta)))
   


@dataclass
class CIR:
    a: float
    b: float
    sigma: float

    def pdf(self, x:Array)->Array:
        a, b, sigma = self.a, self.b, self.sigma
        beta = 2*a/sigma**2
        alpha = 2*a*b/sigma**2
        return beta**alpha/gamma(alpha)*x**(alpha-1)*jnp.exp(-beta*x)
    
@dataclass
class BrownianWithReset2D:
    mu: Array
    Sigma: Array
    chi: Array
    entry: Array

    def pdf(self, x,y):
        Sigma = self.Sigma
        sigma_x, sigma_y = Sigma[0,0], Sigma[1,1]
        rho_xy = Sigma[0,1]/(sigma_x*sigma_y)
        chi = self.chi
        x_entry, y_entry = self.entry
        coeff = chi/(jnp.pi*sigma_x*sigma_y*jnp.sqrt(1-rho_xy**2))
        exp_term = jnp.exp(-1/(1-rho_xy**2)*((x - x_entry)**2/sigma_x**2 - 2*rho_xy*(x - x_entry)*(y - y_entry)/(sigma_x*sigma_y) + (y - y_entry)**2/sigma_y**2))
        return coeff*exp_term
