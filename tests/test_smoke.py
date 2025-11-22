import subprocess, sys, os, tempfile

def test_smoke_gue_trp_runs():
    with tempfile.TemporaryDirectory() as td:
        cmd = [
            sys.executable, "-m", "src.run_demo_a",
            "--task", "gue",
            "--baseline", "trp",
            "--steps", "50",
            "--seeds", "0",
            "--batch", "256",
            "--alpha", "6.0",
            "--tau", "0.02",
            "--out", td
        ]
        subprocess.check_call(cmd)
        assert os.path.exists(os.path.join(td, "summary.csv"))
