import argparse, os, json, math
import numpy as np
import pandas as pd
import subprocess
from .utils import ensure_dir

def run_cmd(cmd):
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="gue",
                    choices=["gue","odlyzko","poisson_null","shuffle_null","phase_null"])
    ap.add_argument("--odlyzko_path", type=str, default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--alpha_min", type=float, default=0.5)
    ap.add_argument("--alpha_max", type=float, default=20.0)
    ap.add_argument("--n_alphas", type=int, default=9)
    ap.add_argument("--baseline_ref", type=str, default="adamw_cos",
                    choices=["sgd_flat","adamw_cos","kl_adapt"])
    ap.add_argument("--min_improve_frac", type=float, default=0.40,
                    help="TRP must improve final I_struct by this fraction vs baseline on real tasks")
    ap.add_argument("--out", type=str, default="runs/alpha_sweep")
    args = ap.parse_args()

    ensure_dir(args.out)

    alphas = np.logspace(math.log10(args.alpha_min), math.log10(args.alpha_max), args.n_alphas)

    # 1) Run reference baseline once
    ref_dir = os.path.join(args.out, f"ref_{args.baseline_ref}")
    ensure_dir(ref_dir)
    if not os.path.exists(os.path.join(ref_dir, "summary.csv")):
        cmd = [
            "python","-m","src.run_demo_a",
            "--task", args.task,
            "--baseline", args.baseline_ref,
            "--steps", str(args.steps),
            "--lr", str(args.lr),
            "--batch", str(args.batch),
            "--tau", str(args.tau),
            "--out", ref_dir,
            "--seeds", *map(str,args.seeds)
        ]
        if args.odlyzko_path:
            cmd += ["--odlyzko_path", args.odlyzko_path]
        run_cmd(cmd)

    ref_summary = pd.read_csv(os.path.join(ref_dir, "summary.csv")).iloc[0]
    ref_I = float(ref_summary["final_I_struct_mean"])

    # 2) Sweep TRP alphas
    rows = []
    for a in alphas:
        trp_dir = os.path.join(args.out, f"trp_alpha_{a:.4g}")
        ensure_dir(trp_dir)
        if not os.path.exists(os.path.join(trp_dir, "summary.csv")):
            cmd = [
                "python","-m","src.run_demo_a",
                "--task", args.task,
                "--baseline", "trp",
                "--steps", str(args.steps),
                "--lr", str(args.lr),
                "--batch", str(args.batch),
                "--tau", str(args.tau),
                "--alpha", str(a),
                "--out", trp_dir,
                "--seeds", *map(str,args.seeds)
            ]
            if args.odlyzko_path:
                cmd += ["--odlyzko_path", args.odlyzko_path]
            run_cmd(cmd)

        summ = pd.read_csv(os.path.join(trp_dir, "summary.csv")).iloc[0]
        trp_I = float(summ["final_I_struct_mean"])
        improve_frac = (ref_I - trp_I) / max(ref_I, 1e-12)

        rows.append({
            "alpha": float(a),
            "ref_baseline": args.baseline_ref,
            "ref_final_I": ref_I,
            "trp_final_I": trp_I,
            "improve_frac": improve_frac
        })

    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(args.out, "alpha_sweep.csv"), index=False)

    # 3) Simple pass/fail based on prereg spirit
    passed = sweep["improve_frac"] >= args.min_improve_frac

    verdict = {
        "task": args.task,
        "baseline_ref": args.baseline_ref,
        "min_improve_frac": args.min_improve_frac,
        "alphas_tested": sweep["alpha"].tolist(),
        "improve_fracs": sweep["improve_frac"].tolist(),
        "any_alpha_passes": bool(passed.any()),
        "pass_alpha_values": sweep.loc[passed, "alpha"].tolist(),
    }

    with open(os.path.join(args.out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)

    # 4) Plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.semilogx(sweep["alpha"], sweep["improve_frac"], marker="o")
    plt.axhline(args.min_improve_frac, linestyle="--")
    plt.xlabel("alpha")
    plt.ylabel("TRP improvement fraction vs baseline")
    plt.title(f"Alpha sweep: {args.task} (ref={args.baseline_ref})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "alpha_sweep.png"), dpi=160)
    plt.close()

    print("Alpha sweep done.")
    print(json.dumps(verdict, indent=2))

if __name__ == "__main__":
    main()
