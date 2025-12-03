
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import special as sp


@dataclass
class CIRModel:
    """
    Cox-Ingersoll-Ross model parameters
    dz = theta*(z_bar - z) dt+sigma*sqrt(z-z_min)*
    """
    theta: float
    z_bar: float
    sigma: float
    z_min: float = 0.0

    def stationary_distribution(self, z:np.ndarray)->np.ndarray:
        return cir_stationary_distribution(z, self.theta, self.sigma, self.z_bar, self.z_min)

def fit_cir(z, dt:float, z_min:float = 0)->CIRModel:
    """
    Fits a Cox-Ingersoll-Ross model
    dz = theta*(z_bar - z) dt+sigma*sqrt(z-z_min)*dW using OLS
    """
    if z_min != 0:
        z = z - z_min

    if z.min() < 0:
        raise ValueError("x must be non-negative")
    rs = z[:- 1]  
    rt = z[1:]
    model = LinearRegression()
    y = (rt - rs) / np.sqrt(rs)
    z1 = dt / np.sqrt(rs)
    z2 = dt * np.sqrt(rs)
    X = np.column_stack((z1, z2))
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    y_hat = model.predict(X)
    residuals = y - y_hat
    beta_1, beta_2 = model.coef_    
    theta = -beta_2
    z_bar = beta_1/theta
    sigma = np.std(residuals)/np.sqrt(dt)
    if z_min != 0:
        z_bar += z_min
    return CIRModel(theta=theta.item(), z_bar=z_bar.item(), sigma=sigma.item(), z_min=z_min)


def gamma_pdf(x, alpha, beta):
    return (x**(alpha - 1) * np.exp(-x / beta)) / (
        beta**alpha * sp.gamma(alpha)
    )

def cir_stationary_distribution(z, theta, sigma, z_bar = 0, z_min = 0):
    """
    Analytical pd fof the process
    dz = theta * (z_bar - z) + sigma*sqrt(z-z_L)*dW
    """
    alpha = 2 * theta * (z_bar - z_min) / sigma**2
    beta = sigma**2 / (2 * theta)
    return gamma_pdf(z-z_min, alpha, beta)