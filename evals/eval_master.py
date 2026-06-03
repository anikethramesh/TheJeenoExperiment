import argparse
import sys
import subprocess
from pathlib import Path

from manifest import EVAL_SPECS, select_eval_specs


def main():
    parser = argparse.ArgumentParser(description="Run JEENOM eval probes.")
    parser.add_argument(
        "--suite",
        default="all",
        help="Eval suite to run: all, architecture, cleanup, smoke.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected evals without running them.",
    )
    args = parser.parse_args()

    evals_dir = Path(__file__).parent
    selected_specs = select_eval_specs(args.suite)
    if not selected_specs:
        known = sorted(
            {"all"}
            | {
                suite
                for spec in EVAL_SPECS
                for suite in spec.get("suites", [])
            }
        )
        print(f"Unknown or empty suite: {args.suite}")
        print(f"Known suites: {', '.join(known)}")
        sys.exit(2)

    eval_files = [evals_dir / str(spec["file"]) for spec in selected_specs]

    missing = [path.name for path in eval_files if not path.exists()]
    if missing:
        print("Manifest references missing eval files:")
        for name in missing:
            print(f"  - {name}")
        sys.exit(2)

    if args.list:
        print(f"Selected {len(eval_files)} eval scripts for suite={args.suite}:")
        for spec, path in zip(selected_specs, eval_files):
            suites = ", ".join(spec.get("suites", []))
            print(f"  - {path.name} [{suites}]")
        sys.exit(0)
    
    print(f"Found {len(eval_files)} eval scripts to run for suite={args.suite}.")
    
    failures = []
    fallback_enabled = []
    
    for eval_file in eval_files:
        print(f"\n{'='*60}")
        cmd = [sys.executable, str(eval_file)]

        # Check if script accepts --allow-fallback
        script_content = eval_file.read_text()
        if "--allow-fallback" in script_content:
            cmd.append("--allow-fallback")
            fallback_enabled.append(eval_file.name)

        mode = "offline fallback allowed" if "--allow-fallback" in cmd else "strict/offline"
        print(f"Running {eval_file.name} ({mode})...")
        print(f"{'='*60}")
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ FAILED: {eval_file.name}")
            print(result.stdout)
            print(result.stderr)
            failures.append((eval_file.name, result.returncode))
        else:
            print(f"✅ PASSED: {eval_file.name}")
            # Optionally print stdout if you want to see details, but keeping it quiet for passes
            # is usually better for a master script. We'll just show pass.
            
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total run: {len(eval_files)}")
    print(f"Passed: {len(eval_files) - len(failures)}")
    print(f"Failed: {len(failures)}")
    print(f"Fallback-enabled probes: {len(fallback_enabled)}")
    
    if failures:
        print("\nFailed scripts:")
        for name, code in failures:
            print(f"  - {name} (exit code: {code})")
        sys.exit(1)
    else:
        print("\nAll evals passed successfully! 🎉")
        sys.exit(0)

if __name__ == "__main__":
    main()
