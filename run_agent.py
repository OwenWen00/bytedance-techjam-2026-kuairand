import argparse

from agent.controller import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous research agent loop")
    parser.add_argument("--dry-run", action="store_true", help="Run the dry-run autonomous loop.")
    parser.add_argument("--max-iterations", type=int, default=1, help="Maximum number of iterations to execute.")
    parser.add_argument("--experiment-config", default=None, help="Path to a real validation experiment config JSON file.")
    args = parser.parse_args()

    if args.dry_run:
        summary = run_agent(max_iterations=args.max_iterations)
        print(f"iterations_run={summary['iterations_run']} accepted={summary['accepted']} rejected={summary['rejected']}")
        print(f"best_experiment_id={summary['best_experiment_id']} best_primary={summary['best_primary']:.4f}")
        print(f"converged={summary['converged']} stop_reason={summary['stop_reason']}")
        print(f"log_path={summary['log_path']}")
        return

    summary = run_agent(max_iterations=args.max_iterations, mode="real", config_path=args.experiment_config)
    print(f"iterations_run={summary['iterations_run']} accepted={summary['accepted']} rejected={summary['rejected']}")
    print(f"best_experiment_id={summary['best_experiment_id']} best_primary={summary['best_primary']:.4f}")
    print(f"consecutive_no_improvement={summary['consecutive_no_improvement']}")
    print(f"converged={summary['converged']} stop_reason={summary['stop_reason']}")
    print(f"log_path={summary['log_path']}")


if __name__ == "__main__":
    main()
