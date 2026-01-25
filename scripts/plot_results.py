import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ============================================================
# Paths
# ============================================================python3 scripts/plot_results.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = PROJECT_ROOT / "results.csv"
EXAMPLE_DIR = PROJECT_ROOT / "backend" / "benchmarks"

print(f"Loading results from: {RESULTS_CSV}")

# ============================================================
# Load CSV
# ============================================================
df = pd.read_csv(RESULTS_CSV)
df.columns = df.columns.str.strip()

# Keep only solved runs
df = df[df["status"].isin(["OPTIMAL", "FEASIBLE"])].copy()

# ============================================================
# Load JSONs and compute total periods
# ============================================================
def total_periods_for_instance(instance_name: str) -> int:
    path = EXAMPLE_DIR / f"{instance_name}.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return sum(course["periods"] for course in data["courses"])

instance_to_periods = {
    inst: total_periods_for_instance(inst)
    for inst in df["instance"].unique()
}

df["total_periods"] = df["instance"].map(instance_to_periods)

# ============================================================
# Normalize objective
# ============================================================
df["penalty_per_period"] = df["objective_value"] / df["total_periods"]

# ============================================================
# PLOT 1 — Penalty per period across seeds
# ============================================================
plt.figure(figsize=(7, 4))
for inst in sorted(df["instance"].unique()):
    subset = df[df["instance"] == inst]
    plt.scatter(subset["seed"], subset["penalty_per_period"], label=inst, alpha=0.7)

plt.xlabel("Random seed")
plt.ylabel("Penalty per period")
plt.title("Penalty per period across seeds")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ============================================================
# PLOT 2 — Runtime across seeds (small instance)
# ============================================================
plt.figure(figsize=(7, 4))
small = df[df["instance"] == "small"]

plt.scatter(small["seed"], small["wall_time_s"], alpha=0.7)
plt.xlabel("Random seed")
plt.ylabel("Wall time (s)")
plt.title("Runtime across seeds (small instance)")
plt.grid(True)
plt.tight_layout()

# ============================================================
# PLOT 3 — Mean penalty per period by instance size
# ============================================================
plt.figure(figsize=(6, 4))
grouped = (
    df.groupby("instance")["penalty_per_period"]
      .agg(["mean", "std"])
      .reindex(["small", "medium", "large"])
)

plt.bar(grouped.index, grouped["mean"], yerr=grouped["std"], capsize=6)
plt.ylabel("Mean penalty per period")
plt.title("Solution quality by instance size")
plt.grid(axis="y")
plt.tight_layout()

# ============================================================
# PLOT 4 — Penalty composition by constraint type
# ============================================================
plt.figure(figsize=(7, 4))

penalty_components = {
    "double_period": "penalty_double_period",
    "subject_daily_cap": "penalty_subject_daily_cap",
    "difficult_day": "penalty_difficult_day",
    "unavailability": "penalty_unavailability",
    "unqualified": "penalty_unqualified",
}

stack_data = (
    df.groupby("instance")[list(penalty_components.values())]
      .mean()
      .reindex(["small", "medium", "large"])
)

bottom = None
for label, col in penalty_components.items():
    if bottom is None:
        plt.bar(stack_data.index, stack_data[col], label=label)
        bottom = stack_data[col]
    else:
        plt.bar(stack_data.index, stack_data[col], bottom=bottom, label=label)
        bottom += stack_data[col]

plt.ylabel("Mean penalty contribution")
plt.title("Penalty composition by constraint type")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()

# ============================================================
# SHOW ALL FIGURES AT ONCE
# ============================================================
plt.show()
