import argparse, os
import numpy as np
import pandas as pd
import torch
from tqdm import trange

from .utils import set_global_seed, ensure_dir, save_json
from .models import SpacingGenerator
from .trp import TRPScheduler
from .baselines import make_optimizer, CosineLRSchedule, KLAdaptiveSchedule
from .metrics import (
    I_struct_KL, KS_distance,
    wigner_gue_pdf, wigner_gue_cdf, poisson_pdf, poisson_cdf,
    cdf_wasserstein_L1_torch
)
from .data import (
    sample_gue_spacings, load_odlyzko_spacings,
    sample_poisson_spacings, shuffled_spacings, phase_randomized_gue_spacings
)

def get_task(task, odlyzko_path=None, seed=0):
    if task == "gue":
        spacings = sample_gue_spacings(seed=seed)
        target_pdf_fn = wigner_gue_pdf
        target_cdf_fn = wigner_gue_cdf
        return spacings, target_pdf_fn, target_cdf_fn
    if task == "odlyzko":
        if odlyzko_path is None:
            raise ValueError("Need --odlyzko_path for odlyzko task.")
        spacings = load_odlyzko_spacings(odlyzko_path)
        target_pdf_fn = wigner_gue_pdf  # expected GUE-like
        target_cdf_fn = wigner_gue_cdf
        return spacings, target_pdf_fn, target_cdf_fn
    if task == "poisson_null":
        spacings = sample_poisson_spacings(seed=seed)
        target_pdf_fn = poisson_pdf
        target_cdf_fn = poisson_cdf
        return spacings, target_pdf_fn, target_cdf_fn
    if task == "shuffle_null":
        base = sample_gue_spacings(seed=seed)
        spacings = shuffled_spacings(base, seed=seed)
        target_pdf_fn = wigner_gue_pdf
        target_cdf_fn = wigner_gue_cdf
        return spacings, target_pdf_fn, target_cdf_fn
    if task == "phase_null":
        spacings = phase_randomized_gue_spacings(seed=seed)
        target_pdf_fn = wigner_gue_pdf
        target_cdf_fn = wigner_gue_cdf
        return spacings, target_pdf_fn, target_cdf_fn
    raise ValueError(f"Unknown task {task}")

def run_one_seed(args, seed):
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    target_spacings, target_pdf_fn, target_cdf_fn = get_task(args.task, args.odlyzko_path, seed=seed)

    # Fixed evaluation grid for KDE / CDF
    grid = np.linspace(0, args.grid_max, args.grid_n)
    grid_t = torch.from_numpy(grid).to(device=device, dtype=torch.float32)

    # Precompute target CDF on grid (numpy -> torch)
    tar_cdf = target_cdf_fn(grid)
    tar_cdf_t = torch.from_numpy(tar_cdf).to(device=device, dtype=torch.float32)

    # Model
    model = SpacingGenerator(latent_dim=args.latent_dim, hidden=args.hidden, depth=args.depth).to(device)
    opt = make_optimizer(model, args.baseline, args.lr)

    # Schedules
    trp_sched = TRPScheduler(base_lr=args.lr, alpha=args.alpha) if args.baseline == "trp" else None
    cos_sched = CosineLRSchedule(args.lr, args.steps) if args.baseline == "adamw_cos" else None
    kl_sched  = KLAdaptiveSchedule(args.lr, target_kl=args.target_kl) if args.baseline == "kl_adapt" else None

    logs = []
    for t in trange(args.steps, desc=f"seed {seed}", leave=False):
        z = torch.randn(args.batch, args.latent_dim, device=device)
        spacings = model(z)  # (B,) torch tensor with grads

        spac_np = spacings.detach().cpu().numpy()

        # structure metric (eval-only, numpy)
        I_struct = I_struct_KL(spac_np, target_pdf_fn, grid)

        # LR pacing
        if trp_sched is not None:
            lr_t = trp_sched.lr(I_struct)
        elif cos_sched is not None:
            lr_t = cos_sched.lr(t)
        elif kl_sched is not None:
            lr_t = kl_sched.lr(I_struct)
        else:
            lr_t = args.lr

        for pg in opt.param_groups:
            pg["lr"] = lr_t

        # differentiable training loss (torch)
        loss_t = cdf_wasserstein_L1_torch(
            spacings_t=spacings,
            grid_t=grid_t,
            target_cdf_t=tar_cdf_t,
            tau=args.tau
        )

        opt.zero_grad()
        loss_t.backward()
        opt.step()

        if (t % args.log_every) == 0 or t == args.steps-1:
            ks = KS_distance(spac_np, target_cdf_fn)
            logs.append({
                "step": t,
                "loss_cdfL1": float(loss_t.detach().cpu()),
                "I_struct_KL": I_struct,
                "KS": ks,
                "lr_t": lr_t
            })

    return pd.DataFrame(logs), model.state_dict(), grid, target_pdf_fn

def summarize_runs(run_dirs):
    rows = []
    for rd in run_dirs:
        summ = pd.read_csv(os.path.join(rd, "summary.csv"))
        summ["run_dir"] = rd
        rows.append(summ)
    out = pd.concat(rows, ignore_index=True)
    print(out.sort_values(["task","baseline","final_I_struct_mean"]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="gue",
                    choices=["gue","odlyzko","poisson_null","shuffle_null","phase_null"])
    ap.add_argument("--baseline", type=str, default="trp",
                    choices=["trp","sgd_flat","adamw_cos","kl_adapt"])
    ap.add_argument("--odlyzko_path", type=str, default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--target_kl", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--latent_dim", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--grid_n", type=int, default=256)
    ap.add_argument("--grid_max", type=float, default=5.0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--tau", type=float, default=0.02, help="soft CDF temperature")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", type=str, default="runs/demo_a")
    ap.add_argument("--summarize", nargs="*", default=None)
    args = ap.parse_args()

    if args.summarize is not None and len(args.summarize) > 0:
        summarize_runs(args.summarize)
        return

    ensure_dir(args.out)
    save_json(vars(args), os.path.join(args.out, "config.json"))

    seed_dfs = []
    seed0_state = None
    for i, seed in enumerate(args.seeds):
        df, state, grid, target_pdf_fn = run_one_seed(args, seed)
        path = os.path.join(args.out, f"seed_{seed}.csv")
        df.to_csv(path, index=False)
        seed_dfs.append(df)
        if i == 0:
            seed0_state = state

    # aggregate curves
    cat = pd.concat(seed_dfs, ignore_index=True)
    curves = cat.groupby(["step"]).agg({
        "I_struct_KL": ["mean","std"],
        "loss_cdfL1": ["mean","std"],
        "KS": ["mean","std"]
    }).reset_index()
    curves.columns = ["step","I_struct_mean","I_struct_std",
                      "loss_mean","loss_std","KS_mean","KS_std"]

    curves.to_csv(os.path.join(args.out, "curves.csv"), index=False)

    # summary at last logged step
    last_step = curves["step"].max()
    last_row = curves[curves["step"]==last_step].iloc[0].to_dict()
    summary = pd.DataFrame([{
        "task": args.task,
        "baseline": args.baseline,
        "steps": args.steps,
        "alpha": args.alpha,
        "lr": args.lr,
        "batch": args.batch,
        "tau": args.tau,
        "final_I_struct_mean": last_row["I_struct_mean"],
        "final_I_struct_std": last_row["I_struct_std"],
        "final_KS_mean": last_row["KS_mean"],
        "final_KS_std": last_row["KS_std"],
        "final_loss_mean": last_row["loss_mean"],
        "final_loss_std": last_row["loss_std"]
    }])
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    # plots
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(curves["step"], curves["I_struct_mean"])
    plt.fill_between(curves["step"],
                     curves["I_struct_mean"]-curves["I_struct_std"],
                     curves["I_struct_mean"]+curves["I_struct_std"],
                     alpha=0.2)
    plt.xlabel("step"); plt.ylabel("I_struct KL(emp||target)")
    plt.title(f"{args.task} / {args.baseline}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "curves.png"), dpi=160)
    plt.close()

    # final histogram using trained seed0 model
    if seed0_state is not None:
        model = SpacingGenerator(latent_dim=args.latent_dim, hidden=args.hidden, depth=args.depth)
        model.load_state_dict(seed0_state)
        model.eval()
        z = torch.randn(20000, args.latent_dim)
        spac_np = model(z).detach().numpy()

        bins = np.linspace(0, args.grid_max, args.grid_n)
        emp_pdf = np.histogram(spac_np, bins=bins, density=True)[0]
        mid = (bins[:-1]+bins[1:])/2
        tar_pdf = target_pdf_fn(mid)
        tar_pdf /= np.trapz(tar_pdf, mid)

        plt.figure()
        plt.plot(mid, emp_pdf, label="empirical")
        plt.plot(mid, tar_pdf, label="target")
        plt.xlabel("s"); plt.ylabel("pdf")
        plt.title("Final spacing pdf (seed0 trained)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "final_hist.png"), dpi=160)
        plt.close()

    print("Done. Summary:")
    print(summary)

if __name__ == "__main__":
    main()
