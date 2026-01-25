#!/usr/bin/env python3
"""Batch runner for the timetabling solver.

This script runs the CP-SAT timetabling solver on multiple instances
and random seeds, and writes quantitative evaluation results to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from backend.solver import solve_timetabling_problem

EXAMPLE_DIR = PROJECT_ROOT / "backend" / "benchmarks"


# ---------- Helpers ----------

def load_instance(size: str) -> Dict[str, Any]:
    path = EXAMPLE_DIR / f"{size}.json"
    if not path.exists():
        raise FileNotFoundError(f"No example data found for '{size}' at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def total_periods(data: Dict[str, Any]) -> int:
    """Total number of lesson periods in an instance."""
    return sum(course["periods"] for course in data.get("courses", []))


# ---------- Benchmark Runner ----------

def run_benchmark(
    sizes: List[str],
    seed_count: int,
    time_limit: float,
    output: Path,
) -> None:
    records: List[Dict[str, Any]] = []

    print("Running benchmark (single-threaded, reproducible).")

    for size in sizes:
        data = load_instance(size)

        meta = {
            "instance": size,
            "n_classes": len(data.get("classes", [])),
            "n_teachers": len(data.get("teachers", [])),
            "n_rooms": len(data.get("rooms", [])),
            "n_courses": len(data.get("courses", [])),
            "n_days": len(data.get("days", [])),
            "num_slots_per_day": data.get("num_slots_per_day"),
            "total_periods": total_periods(data),
        }

        for seed in range(seed_count):
            solution = solve_timetabling_problem(
                data,
                time_limit_seconds=time_limit,
                random_seed=seed,
            )

            status = solution.get("status")
            obj = solution.get("objective_value")
            stats = solution.get("stats", {})
            penalties = solution.get("penalties", {})

            # Extract individual penalty components safely
            def p(key: str) -> float:
                entry = penalties.get(key, {})
                return float(entry.get("weighted", 0.0))

            record = {
                **meta,
                "seed": seed,
                "status": status,
                "objective_value": obj,
                "wall_time_s": stats.get("wall_time_s"),
                "conflicts": stats.get("conflicts"),
                "branches": stats.get("branches"),

                # ---- penalty components (flat, explicit) ----
                "penalty_unavailability": p("unavailability"),
                "penalty_unqualified": p("unqualified"),
                "penalty_double_period": p("double_period_singles"),
                "penalty_difficult_day": p("triple_difficult"),
                "penalty_subject_daily_cap": p("subject_daily_cap"),
                "penalty_total": penalties.get("total_weighted", {}).get("weighted"),
            }

            records.append(record)

            print(
                f"[{size}] seed={seed}: "
                f"status={status} obj={obj}"
            )

    # ---------- Write CSV ----------

    fieldnames = [
        "instance",
        "seed",
        "status",
        "objective_value",
        "wall_time_s",
        "conflicts",
        "branches",
        "n_classes",
        "n_teachers",
        "n_rooms",
        "n_courses",
        "n_days",
        "num_slots_per_day",
        "total_periods",
        "penalty_unavailability",
        "penalty_unqualified",
        "penalty_double_period",
        "penalty_difficult_day",
        "penalty_subject_daily_cap",
        "penalty_total",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print(f"Wrote benchmark results to {output}")


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        sizes=args.sizes,
        seed_count=args.seed_count,
        time_limit=args.time_limit,
        output=args.output,
    )


if __name__ == "__main__":
    main()
