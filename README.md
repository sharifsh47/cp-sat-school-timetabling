## Solver Overview

The core solver is implemented in `backend/solver.py` and models the school timetabling problem using constraint programming.

### Hard constraints include:
- Exact fulfillment of required lesson periods
- No overlaps for classes, teachers, or rooms
- Daily minimum and maximum lesson bounds per class
- Weekly workload limits per teacher
- Contiguous daily lesson blocks for each class

### Soft constraints include:
- Teacher availability preferences
- Qualification preferences for specialist subjects
- Subject-specific daily limits
- Preferences for double periods
- Penalties for clustering multiple difficult subjects on the same day

All constraints and parameters are generated dynamically from structured JSON input.

---

## Benchmark Instances

Three benchmark instances are provided in `backend/benchmarks/`:
- `small.json`
- `medium.json`
- `large.json`

These instances differ in size and interaction density while preserving comparable constraint ratios. They are used to evaluate solver feasibility, solution quality, and runtime behavior under fixed solver configurations, as described in the thesis.

---

## Experimental Evaluation

The experimental workflow is implemented in the `scripts/` directory:
- `run_benchmark.py` executes the solver across multiple instances and random seeds
- `plot_results.py` aggregates results and generates plots for runtime and penalty analysis

All experiments are run with fixed solver parameters and a fixed time limit to ensure comparability.

---

## Frontend Visualization

A lightweight frontend implemented with Next.js is included to visualize generated time
