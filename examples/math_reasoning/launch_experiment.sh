#!/bin/bash
# Backward-compatible shim — delegates to dashboard/launch_experiment.sh
# Forces --task-type math when invoked from old location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

exec bash "$REPO_ROOT/dashboard/launch_experiment.sh" --task-type math "$@"
