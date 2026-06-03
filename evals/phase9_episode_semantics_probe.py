"""Phase 9 probe: documented fresh-episode reference semantics.

Phase 6/9 docs currently say task turns use fresh task-episode semantics.
Reference and repeat requests must therefore replay the resolved task cleanly,
not continue from a completed human-render adapter and fail with no_path_found.
"""
from __future__ import annotations

from harness import emit_result, first_line, make_session, patched_env_builder


def _human_session():
    return make_session(render_mode="human", max_loops=64)


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, str] = {}

    with patched_env_builder():
        same_target = _human_session()
        first = same_target.handle_utterance("go to the red door")
        again = same_target.handle_utterance("go there again")
        metrics["initial_task_completes"] = "RUN COMPLETE" in first
        metrics["go_there_again_replays_fresh_episode"] = "RUN COMPLETE" in again
        metrics["go_there_again_does_not_no_path"] = "no_path_found" not in again
        details["go_there_again_response"] = first_line(again)

        repeat = _human_session()
        repeat.handle_utterance("go to the red door")
        repeat_response = repeat.handle_utterance("repeat the last task")
        metrics["repeat_last_task_replays_fresh_episode"] = (
            "RUN COMPLETE" in repeat_response
        )
        metrics["repeat_last_task_does_not_no_path"] = (
            "no_path_found" not in repeat_response
        )
        details["repeat_last_task_response"] = first_line(repeat_response)

        delivery = _human_session()
        delivery.handle_utterance("the red door is the delivery target")
        delivery.handle_utterance("go to the red door")
        delivery.handle_utterance("reset")
        delivery_response = delivery.handle_utterance("go to the delivery target")
        metrics["delivery_target_after_reset_replays_fresh_episode"] = (
            "RUN COMPLETE" in delivery_response
        )
        metrics["delivery_target_after_reset_does_not_no_path"] = (
            "no_path_found" not in delivery_response
        )
        details["delivery_target_response"] = first_line(delivery_response)

    metrics["fresh_episode_reference_semantics_hold"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="fresh_episode_reference_semantics_hold")


if __name__ == "__main__":
    raise SystemExit(main())
