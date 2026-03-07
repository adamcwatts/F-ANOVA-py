import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import pytest


def logistic_function(L: float, k: float, x_0: float, x: np.ndarray) -> np.ndarray:
    return L / (1+ np.exp(-k*(x-x_0)))

def rbf_kernel(s: float, t: float, l: float) -> float:
    return np.exp(-(np.linalg.norm(s-t)**2) /  (2* l**2))

def generate_logistic_data(eta_t: np.ndarray, cov_matrix: np.ndarray, n_reals: int, rng: np.random.Generator) -> np.ndarray:
    noise = multivariate_normal.rvs(mean=np.zeros(len(eta_t)), cov=cov_matrix, size=n_reals, random_state=rng)
    return (eta_t + noise).T

def covariance_matrix(kernel_fn, t_grid: np.ndarray) -> np.ndarray:
    N = len(t_grid)
    cov = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1):
            cov[i,j] = kernel_fn(t_grid[i], t_grid[j])
    return cov+cov.T - np.diag(np.diag(cov))

def init_rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))
