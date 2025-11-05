#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running Ruff checks (lint + import sort)..."
ruff check /Users/sharifalmasri/Documents/THESIS/Thesis/backend/ --fix

echo "Done. (Skipped mypy for a simpler workflow)"
