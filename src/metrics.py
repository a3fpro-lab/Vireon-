import numpy as np
import torch
from scipy.stats import entropy, kstest

# Target PDFs / CDFs
def wigner_gue_pdf(s: np.ndarray) -> np.ndarray:
    return (32/np.pi**2) * s**2 * np.exp(-4*s**2/np.pi)

def wigner_gue_cdf(s: np.ndarray) -> np.ndarray:
    from scipy.integrate import cumulative_trapezoid
    pdf = wigner_gue_pdf(s)
    cdf = cumulative_trapezoid(pdf, s, initial=0.0)
    cdf /= cdf[-1]
    return cdf

def poisson_pdf(s: np.ndarray) -> np.ndarray:
    return np.exp(-s)

def poisson_cdf(s: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-s)

# Evaluation-only (NumPy)
def empirical_pdf(samples: np.ndarray, grid: np.ndarray, bandwidth: float = None) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    if n < 5:
        pdf = np.ones_like(grid)
        return pdf / np.trapz(pdf, grid)
    if bandwidth is None:
        std = np.std(samples)
        bandwidth = n**(-1/5) * std if std > 0 else 0.1
    diffs = (grid[:, None] - samples[None, :]) / bandwidth
    kern = np.exp(-0.5 * diffs**2) / (np.sqrt(2*np.pi) * bandwidth)
    pdf = kern.mean(axis=1)
    pdf = np.maximum(pdf, 1e-12)
    pdf /= np.trapz(pdf, grid)
    return pdf

def I_struct_KL(samples: np.ndarray, target_pdf_fn, grid: np.ndarray) -> float:
    p_emp = empirical_pdf(samples, grid)
    p_tar = target_pdf_fn(grid)
    p_tar = np.maximum(p_tar, 1e-12)
    p_tar /= np.trapz(p_tar, grid)
    return float(entropy(p_emp, p_tar))

def KS_distance(samples: np.ndarray, target_cdf_fn) -> float:
    stat, _ = kstest(samples, target_cdf_fn)
    return float(stat)

# Torch-native differentiable loss
def soft_empirical_cdf(spacings_t: torch.Tensor, grid_t: torch.Tensor, tau: float) -> torch.Tensor:
    diff = (grid_t[:, None] - spacings_t[None, :]) / tau
    return torch.sigmoid(diff).mean(dim=1)

def cdf_wasserstein_L1_torch(
    spacings_t: torch.Tensor,
    grid_t: torch.Tensor,
    target_cdf_t: torch.Tensor,
    tau: float = 0.02
) -> torch.Tensor:
    emp_cdf_t = soft_empirical_cdf(spacings_t, grid_t, tau)
    return torch.mean(torch.abs(emp_cdf_t - target_cdf_t))
