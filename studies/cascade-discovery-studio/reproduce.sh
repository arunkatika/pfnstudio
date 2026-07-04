#!/usr/bin/env bash
# Reproduce the cascade-discovery study: train an in-context structure-discovery
# PFN and score adjacency recovery (AUROC / SHD / F1) from its discovery head.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo "→ Validating..."
pfnstudio validate "studies/cascade-discovery-studio/priors/cascade_scm/"
echo "→ Training + scoring (8000 steps; auto-routes per compute.target)..."
pfnstudio run "studies/cascade-discovery-studio/runs/v0_1.yaml"
