"""Phase 9 probe: eval coverage must catch known unit regressions.

This meta-probe runs the current known failing project-local tests. Until those
architectural regressions are fixed, including this probe in eval_master makes
the eval suite honest.
"""
from __future__ import annotations

import json
import subprocess
import sys

from harness import ROOT


KNOWN_REGRESSION_TESTS = [
    "tests/test_jeenom_minigrid.py::JeenomMiniGridTests::test_operator_station_go_there_again_resolves_after_success",
    "tests/test_jeenom_minigrid.py::JeenomMiniGridTests::test_operator_station_llm_intent_resolves_same_one_reference",
    "tests/test_jeenom_minigrid.py::JeenomMiniGridTests::test_operator_station_repeat_last_task_resolves_after_success",
    "tests/test_jeenom_minigrid.py::JeenomMiniGridTests::test_operator_station_reset_clears_reference_context_but_keeps_delivery_target",
    "tests/test_jeenom_minigrid.py::JeenomMiniGridTests::test_operator_station_unsupported_llm_intent_does_not_execute",
    "tests/test_jeenom_minigrid.py::JeenomMiniGridTests::test_spine_corrects_invalid_direct_action_template",
    "tests/test_jeenom_minigrid.py::TestClaimsFilterPrimitiveSynthesis::test_exact_synthesizable_registry_match_overrides_arbitrator_refusal",
]


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", *KNOWN_REGRESSION_TESTS]
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    metrics = {
        "known_regression_tests_executed": result.returncode in {0, 1},
        "known_regression_tests_green": result.returncode == 0,
        "eval_master_would_catch_known_regressions": result.returncode == 0,
    }
    details = {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-1000:],
    }
    print(json.dumps({"metrics": metrics, "details": details}, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
