"""Phase 5 - reproduce the whole study end-to-end.

Runs, in order: TESS benchmark -> injection-recovery -> disagreement analysis
-> baselines/ablations -> 1D-CNN baseline. Each stage writes tables and figures
under research/results/ and docs/img/.

    python -m research.run_all --smoke     # fast sanity pass (small samples)
    python -m research.run_all             # full study (hours; heavy TLS + downloads)
    python -m research.run_all --figures-only   # redraw figures from existing results

The heavy stages (benchmark, injection) are resumable, so an interrupted full
run continues where it left off.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def stage(name: str, args: list[str]) -> None:
    print(f"\n{'=' * 70}\n▶ {name}\n{'=' * 70}", flush=True)
    t0 = time.time()
    subprocess.run([PY, "-m", f"research.{args[0]}", *args[1:]], cwd=ROOT, check=True)
    print(f"✔ {name} done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="Fast sanity pass on small samples")
    ap.add_argument("--figures-only", action="store_true", help="Redraw figures from existing results")
    args = ap.parse_args()

    if args.figures_only:
        for name, mod in [("TESS benchmark", "tess_benchmark"), ("Injection", "injection")]:
            stage(f"{name} figures", [mod, "--figures-only"])
        stage("Disagreement", ["disagreement"])
        stage("Baselines", ["baselines"])
        return

    if args.smoke:
        stage("TESS benchmark (smoke)", ["tess_benchmark", "--limit", "20"])
        stage("Injection-recovery (smoke)", ["injection", "--smoke"])
    else:
        stage("TESS benchmark (full)", ["tess_benchmark", "--limit", "160"])
        stage("Injection-recovery (full)", ["injection", "--full", "--n-hosts", "3"])
    stage("Disagreement triage", ["disagreement"])
    stage("Baselines + ablations", ["baselines"])
    stage("1D-CNN baseline", ["cnn_baseline", "--epochs", "40"])
    stage("Bootstrap confidence intervals", ["statistics"])
    stage("Triage efficiency", ["triage"])
    print("\n✅ Full study reproduced. See research/results/ and docs/img/.", flush=True)


if __name__ == "__main__":
    main()
