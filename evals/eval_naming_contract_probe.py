"""Op 6 probe: eval naming contract.

Every eval file registered in manifest.py (excluding infrastructure files) must
have a name that matches one of the approved capability-based prefixes.

Approved prefixes:
  intent_fidelity_   — NLU / schema fidelity probes
  claim_custody_     — claim lifecycle, context, episode probes
  authority_         — command authority / ticket / gate probes
  repair_            — repair / truthfulness / mismatch probes
  synthesis_         — primitive / metric definition probes
  substrate_         — static architecture / adapter / schema probes
  pipeline_          — planning / dispatch / orchestrator probes
  regression_        — golden / live-operator / historical regression probes

Infrastructure files exempt: harness.py, manifest.py, eval_master.py, this probe.

Metrics:
  op6_all_evals_named_by_contract  — all registered files match an approved prefix
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVALS = ROOT / "evals"

_EXEMPT = frozenset({
    "harness.py",
    "manifest.py",
    "eval_master.py",
    "eval_naming_contract_probe.py",
})

_APPROVED_PREFIXES = (
    "intent_fidelity_",
    "claim_custody_",
    "authority_",
    "repair_",
    "synthesis_",
    "substrate_",
    "pipeline_",
    "regression_",
)


def main() -> int:
    metrics: dict[str, bool] = {}

    from evals.manifest import EVAL_SPECS

    offenders: list[str] = []
    for spec in EVAL_SPECS:
        fname = spec["file"]
        if fname in _EXEMPT:
            continue
        if not any(fname.startswith(prefix) for prefix in _APPROVED_PREFIXES):
            offenders.append(fname)

    if offenders:
        print(f"[op6] Non-compliant eval names ({len(offenders)}):", file=sys.stderr)
        for name in sorted(offenders):
            print(f"  {name}", file=sys.stderr)

    metrics["op6_all_evals_named_by_contract"] = len(offenders) == 0

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
