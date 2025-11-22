import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from tqdm import trange

# --- make repo root importable even on Actions / script runs ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import set_global_seed, ensure_dir, save_json
from src.models import SpacingGenerator
from src.trp import TRPScheduler
from src.baselines import make_optimizer, CosineLRSchedule, KLAdaptiveSchedule
from src.metrics import (
    I_struct_KL, KS_distance,
    wigner_gue_pdf, wigner_gue_cdf, poisson_pdf, poisson_cdf,
    cdf_wasserstein_L1_torch
)
from src.data import (
    sample_gue_spacings, load_odlyzko_spacings,
    sample_poisson_spacings, shuffled_spacings, phase_randomized_gue_spacings
)
