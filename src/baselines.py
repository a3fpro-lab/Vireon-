import math
import torch

class CosineLRSchedule:
    def __init__(self, base_lr, steps, min_lr_ratio=0.05):
        self.base_lr = base_lr
        self.steps = steps
        self.min_lr_ratio = min_lr_ratio

    def lr(self, t):
        # cosine decay from base_lr to base_lr*min_lr_ratio
        c = 0.5 * (1 + math.cos(math.pi * t / self.steps))
        return self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio)*c)

class KLAdaptiveSchedule:
    """
    A simple KL-adaptive LR schedule:
      lr_t = base_lr * clamp(target_KL / (eps + I_struct_t), [min,max])
    This is not PPO; it is a fair baseline proxy for "adapt to KL."
    """
    def __init__(self, base_lr, target_kl=0.05, min_mult=0.2, max_mult=2.0, eps=1e-9):
        self.base_lr = base_lr
        self.target_kl = target_kl
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.eps = eps

    def lr(self, I_struct_t):
        mult = self.target_kl / (self.eps + float(I_struct_t))
        mult = max(self.min_mult, min(self.max_mult, mult))
        return self.base_lr * mult

def make_optimizer(model, baseline: str, lr: float, weight_decay: float = 1e-4):
    if baseline in ("sgd_flat", "trp"):
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    if baseline == "adamw_cos":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if baseline == "kl_adapt":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown baseline: {baseline}")
