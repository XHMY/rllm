"""Backward-compatible shim — delegates to dashboard.evaluate_checkpoints."""

from dashboard.evaluate_checkpoints import main, parse_args  # noqa: F401

if __name__ == "__main__":
    args = parse_args()
    # Force math task type when invoked from old location
    args.task_type = "math"
    main(args)
