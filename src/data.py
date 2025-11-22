import numpy as np

from .metrics import wigner_gue_pdf, poisson_pdf

def unfold_spacings(vals: np.ndarray) -> np.ndarray:
    vals = np.sort(np.asarray(vals, dtype=float))
    spacings = np.diff(vals)
    m = spacings.mean()
    return spacings / m if m > 0 else spacings

# ----- Toy GUE generator -----

def sample_gue_spacings(n_mats: int = 64, dim: int = 128, seed: int = 0) -> np.ndarray:
    """
    Generate GUE matrices, compute eigenvalue spacings, unfold.
    """
    rng = np.random.default_rng(seed)
    all_spacings = []
    for _ in range(n_mats):
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        H = (A + A.conj().T) / 2.0
        eigs = np.linalg.eigvalsh(H).real
        s = unfold_spacings(eigs)
        all_spacings.append(s)
    return np.concatenate(all_spacings)

# ----- Odlyzko loader -----

def load_odlyzko_spacings(path: str) -> np.ndarray:
    """
    Expects file of zeros t_n (one per line).
    Unfold by dividing spacings by mean.
    """
    zeros = np.loadtxt(path, dtype=float)
    return unfold_spacings(zeros)

# ----- Null controls -----

def sample_poisson_spacings(n: int = 10000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.exponential(scale=1.0, size=n)

def shuffled_spacings(spacings: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.array(spacings, copy=True)
    rng.shuffle(s)
    return s

def phase_randomized_gue_spacings(n_mats: int = 64, dim: int = 128, seed: int = 0) -> np.ndarray:
    """
    Randomize phases of GUE by rotating eigenvectors; for spacings this mostly
    acts like a strong null.
    Simplest strong null: random Hermitian with phases scrambled.
    """
    rng = np.random.default_rng(seed)
    all_spacings = []
    for _ in range(n_mats):
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        # scramble phases
        ph = np.exp(1j * rng.uniform(0, 2*np.pi, size=(dim,)))
        A = (A * ph[None, :])
        H = (A + A.conj().T) / 2.0
        eigs = np.linalg.eigvalsh(H).real
        s = unfold_spacings(eigs)
        all_spacings.append(s)
    return np.concatenate(all_spacings)
