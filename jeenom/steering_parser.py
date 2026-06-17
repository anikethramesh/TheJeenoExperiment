"""Phase 13A — steering clause parser.

The single, sanctioned home for steering regex. Operators express HOW to approach a
task (budget / risk / scope / stopping-rule) in natural language, often embedded in the
same utterance as the WHAT ("go to the red door, query only, in at most 5 steps").

`parse_steering_clauses` splits that HOW off into a typed SteeringDirective and returns
the residual WHAT for the normal compiler path. This keeps steering regex OUT of the
intent compiler (no inline shadow NLU) — the directive is then validated/attached by
IntentVerifier (the gate) so a misparse can never silently shape a plan.

Pure, substrate-independent. No env/sense/memory/LLM, no minigrid imports.
"""
from __future__ import annotations

import re
from typing import Callable

from .schemas import SteeringDirective


# Each entry: (compiled pattern, setter(directive_state, match)). First match per field
# wins; every occurrence is stripped from the residual. Patterns are deliberately
# specific so they do not fire on ordinary task phrasing.
def _set_risk(value: str) -> Callable[[dict, re.Match[str]], None]:
    def setter(state: dict, _m: re.Match[str]) -> None:
        state.setdefault("risk", value)
    return setter


def _set_scope(value: str) -> Callable[[dict, re.Match[str]], None]:
    def setter(state: dict, _m: re.Match[str]) -> None:
        state.setdefault("scope", value)
    return setter


def _set_stopping(value: str) -> Callable[[dict, re.Match[str]], None]:
    def setter(state: dict, _m: re.Match[str]) -> None:
        state.setdefault("stopping_rule", value)
    return setter


def _set_max_steps(state: dict, m: re.Match[str]) -> None:
    budget = state.setdefault("budget", {})
    budget.setdefault("max_steps", int(m.group("n")))


def _set_no_clarify(state: dict, _m: re.Match[str]) -> None:
    budget = state.setdefault("budget", {})
    budget.setdefault("max_clarifications", 0)


def _set_max_clarify(state: dict, m: re.Match[str]) -> None:
    budget = state.setdefault("budget", {})
    budget.setdefault("max_clarifications", int(m.group("n")))


_CLAUSES: list[tuple[re.Pattern[str], Callable[[dict, re.Match[str]], None]]] = [
    # Budget — max_steps
    (re.compile(
        r",?\s*\b(?:in|within|using|with)\s+(?:at most\s+|no more than\s+|under\s+|up to\s+)?"
        r"(?P<n>\d+)\s+steps?\b", re.IGNORECASE), _set_max_steps),
    (re.compile(r",?\s*\bat most\s+(?P<n>\d+)\s+steps?\b", re.IGNORECASE), _set_max_steps),
    (re.compile(r",?\s*\b(?P<n>\d+)\s+steps?\s+(?:or fewer|max|maximum|budget)\b",
                re.IGNORECASE), _set_max_steps),
    # Budget — max_clarifications
    (re.compile(r",?\s*\bwithout asking\b", re.IGNORECASE), _set_no_clarify),
    (re.compile(r",?\s*\b(?:ask|asking)\s+(?:at most\s+)?(?P<n>\d+)\s+(?:times|questions)\b",
                re.IGNORECASE), _set_max_clarify),
    # Risk
    (re.compile(r",?\s*\bquery[\s-]?only\b", re.IGNORECASE), _set_risk("query_only")),
    (re.compile(r",?\s*\bread[\s-]?only\b", re.IGNORECASE), _set_risk("query_only")),
    (re.compile(r",?\s*\bwithout (?:acting|actuating|side[\s-]?effects)\b", re.IGNORECASE),
     _set_risk("query_only")),
    (re.compile(r",?\s*\bno (?:actuation|side[\s-]?effects)\b", re.IGNORECASE),
     _set_risk("query_only")),
    (re.compile(r",?\s*\breversible[\s-]?only\b", re.IGNORECASE), _set_risk("reversible_only")),
    (re.compile(r",?\s*\bnothing irreversible\b", re.IGNORECASE), _set_risk("reversible_only")),
    (re.compile(r",?\s*\b(?:without opening anything|don'?t open anything|no opening)\b",
                re.IGNORECASE), _set_risk("reversible_only")),
    (re.compile(r",?\s*\b(?:operator[\s-]?auth?ori[sz]ed|full authority|you may act)\b",
                re.IGNORECASE), _set_risk("operator_authorized")),
    # Scope
    (re.compile(r",?\s*\bvisible[\s-]?only\b", re.IGNORECASE), _set_scope("visible_only")),
    (re.compile(r",?\s*\b(?:only what you can see|don'?t search|no search(?:ing)?)\b",
                re.IGNORECASE), _set_scope("visible_only")),
    (re.compile(r",?\s*\b(?:you may search|feel free to search|search if needed|"
                r"explore if needed)\b", re.IGNORECASE), _set_scope("search_allowed")),
    (re.compile(r",?\s*\b(?:assume (?:the )?full (?:map|grid)|full (?:map|grid))\b",
                re.IGNORECASE), _set_scope("full")),
    # Stopping rule
    (re.compile(r",?\s*\b(?:stop at the first(?:\s+match)?|first match|"
                r"stop (?:after|at) the first)\b", re.IGNORECASE), _set_stopping("first_match")),
    (re.compile(r",?\s*\b(?:be exhaustive|exhaustive(?:ly)?|check all)\b", re.IGNORECASE),
     _set_stopping("exhaustive")),
    (re.compile(r",?\s*\bstop (?:when|if) ambiguous\b|,?\s*\bstop on ambiguity\b",
                re.IGNORECASE), _set_stopping("on_ambiguity")),
    (re.compile(r",?\s*\bstop when (?:the )?budget (?:runs out|is exhausted)\b",
                re.IGNORECASE), _set_stopping("on_budget_exhausted")),
]


def _clean_residual(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    # collapse repeated separators left behind by removed clauses
    text = re.sub(r"(?:\s*,\s*)+", ", ", text)
    text = re.sub(r"\s+(?:and|then)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(?:and|then)\s+", "", text, flags=re.IGNORECASE)
    return text.strip().strip(",").strip()


def parse_steering_clauses(utterance: str) -> tuple[SteeringDirective | None, str]:
    """Return (directive, residual). directive is None when no steering clause is found;
    residual is the utterance with steering clauses removed (== utterance when None)."""
    state: dict = {}
    residual = utterance
    for pattern, setter in _CLAUSES:
        m = pattern.search(residual)
        if m is None:
            continue
        setter(state, m)
        residual = pattern.sub(" ", residual)
    if not state:
        return None, utterance
    directive = SteeringDirective(
        budget=state.get("budget") or None,
        scope=state.get("scope"),
        risk=state.get("risk"),
        stopping_rule=state.get("stopping_rule"),
    )
    return directive, _clean_residual(residual)
