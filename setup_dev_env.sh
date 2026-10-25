#!/bin/bash
set -e
export UV_DYNAMIC_VERSIONING_BYPASS="2.2.0"
export PDM_BUILD_SCM_VERSION="2.2.0"
echo "Setting up local editable environment..."
pip install https://github.com/SamuelMarks/ml-ecosystem-snapshots/releases/download/v0.0.4/ml_ecosystem_snapshots-0.0.4-py3-none-any.whl
pip install --no-deps -e ../ml-switcheroo-ir
pip install --no-deps -e .
echo "Done!"
