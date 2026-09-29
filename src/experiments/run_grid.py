"""Experiment grid runner.

Runs 7 primary configurations + optional ablations.
Primary Grid:
  - bm25-k1-settlein, bm25-k3-settlein, bm25-k5-settlein
  - dense-k1-settlein, dense-k3-settlein, dense-k5-settlein
  - closed_book

Usage:
  python -m src.experiments.run_grid           # run all
  python -m src.experiments.run_grid --dry-run  # print plan only
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 3 primary configurations for Phase 2 (k1/k3 can be added later)
PRIMARY_GRID = [
    {"method": "bm25", "top_k": 5, "variant": "settlein"},
    {"method": "dense", "top_k": 5, "variant": "settlein"},
    {"method": "closed_book", "top_k": 0, "variant": "closed_book"},
]


def config_name(cfg: dict) -> str:
    if cfg["method"] == "closed_book":
        return "closed_book"
    return f"{cfg['method']}-k{cfg['top_k']}-{cfg['variant']}"


def run_grid(dry_run: bool = False):
    results_dir = REPO_ROOT / "results"
    gen_dir = results_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "configurations": [],
    }

    # Try to get git hash
    try:
        import subprocess as sp
        git_hash = sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), stderr=sp.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    print(f"SettleIN Experiment Grid")
    print(f"{'='*60}")
    print(f"Configs: {len(PRIMARY_GRID)}")
    print(f"Git: {git_hash}")
    print(f"{'='*60}\n")

    for i, cfg in enumerate(PRIMARY_GRID, 1):
        name = config_name(cfg)
        gen_file = gen_dir / f"{name}.jsonl"

        # Check if already done (resumable)
        if gen_file.exists() and not dry_run:
            print(f"[{i}/{len(PRIMARY_GRID)}] {name} — SKIP (already exists)")
            manifest["configurations"].append({
                "name": name,
                "status": "skipped",
                "git_hash": git_hash,
            })
            continue

        if dry_run:
            print(f"[{i}/{len(PRIMARY_GRID)}] {name} — WOULD RUN")
            continue

        print(f"[{i}/{len(PRIMARY_GRID)}] {name} — RUNNING...")
        t0 = time.time()

        try:
            from src.experiments.batch_run import run_single_config
            run_single_config(
                method=cfg["method"],
                top_k=cfg["top_k"],
                variant=cfg["variant"],
                config_name=name,
            )
            elapsed = time.time() - t0
            status = "done"
            print(f"  Completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            status = f"error: {e}"
            print(f"  FAILED after {elapsed:.1f}s: {e}")

        manifest["configurations"].append({
            "name": name,
            "method": cfg["method"],
            "top_k": cfg["top_k"],
            "variant": cfg["variant"],
            "status": status,
            "elapsed_s": round(elapsed, 1),
            "git_hash": git_hash,
            "timestamp": datetime.now().isoformat(),
        })

    # Write manifest
    manifest_path = results_dir / "grid_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest written: {manifest_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_grid(dry_run=args.dry_run)
