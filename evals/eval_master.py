import os
import sys
import subprocess
from pathlib import Path

def main():
    evals_dir = Path(__file__).parent
    all_files = list(evals_dir.glob("*.py"))
    
    # Exclude itself and __init__.py
    excluded = {Path(__file__).name, "__init__.py"}
    eval_files = [f for f in all_files if f.name not in excluded]
    
    # Sort them to have some consistent ordering
    eval_files.sort(key=lambda x: x.name)
    
    print(f"Found {len(eval_files)} eval scripts to run.")
    
    failures = []
    
    for eval_file in eval_files:
        print(f"\n{'='*60}")
        print(f"Running {eval_file.name}...")
        print(f"{'='*60}")
        
        cmd = [sys.executable, str(eval_file)]
        
        # Check if script accepts --allow-fallback
        script_content = eval_file.read_text()
        if "--allow-fallback" in script_content:
            cmd.append("--allow-fallback")
            
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
