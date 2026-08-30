import argparse
from pathlib import Path

from agent.controller import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run autonomous research agent loop")
    parser.add_argument("--dry-run", action="store_true", help="Run the dry-run autonomous loop.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of dry-run iterations to execute.")
    args = parser.parse_args()

    if not args.dry_run:
        parser.error("This entry point is only supported for --dry-run mode.")

    summary = run_agent(max_iterations=args.max_iterations)
    print(f"iterations_run={summary['iterations_run']} accepted={summary['accepted']} rejected={summary['rejected']}")
    print(f"best_experiment_id={summary['best_experiment_id']} best_primary={summary['best_primary']:.4f}")
    print(f"log_path={summary['log_path']}")


if __name__ == "__main__":
    main()
