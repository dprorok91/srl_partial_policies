#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$ROOT_DIR/.conda"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is not available in PATH. Install Miniconda or Anaconda first."
  exit 1
fi

if [ -d "$ENV_DIR" ]; then
  echo "Updating existing conda environment at $ENV_DIR"
  conda env update -p "$ENV_DIR" -f "$ROOT_DIR/environment.yml" --prune
else
  echo "Creating conda environment at $ENV_DIR"
  conda env create -p "$ENV_DIR" -f "$ROOT_DIR/environment.yml"
fi

"$ENV_DIR/bin/python" -m ipykernel install --user --name srl_partial_policies --display-name "Python (srl_partial_policies)"

echo ""
echo "Setup complete."
echo "Use VS Code to select the interpreter at: $ENV_DIR/bin/python"
echo "You may need to manually select the Jupyter kernel named 'Python (srl_partial_policies)' in the notebook UI."
