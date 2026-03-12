# p_langchain/runtime/logger.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from p_langchain.core.types import Belief, Hypothesis, PosteriorResult, TraceEvent


def print_trace_summary(result: PosteriorResult, max_events: int = 50) -> None:
    """
    Prints high-level executor summary events.
    This is what you show the interviewer to explain the loop.
    """
    print("\n=== TRACE SUMMARY ===")
    for i, e in enumerate(result.trace_summary[:max_events]):
        _print_event(i, e)
    if len(result.trace_summary) > max_events:
        print(f"... ({len(result.trace_summary) - max_events} more events)")


def print_top_hypotheses(belief: Belief, k: int = 3, p_key: str = "_p", show_state_keys: Optional[List[str]] = None) -> None:
    """
    Print top-k hypotheses with logw and probability.
    show_state_keys: if provided, prints only those state fields (keeps output clean).
    """
    print("\n=== TOP HYPOTHESES ===")
    top = belief.topk(k).hyps
    for i, h in enumerate(top):
        p = h.state.get(p_key, None)
        print(f"\n[{i}] logw={h.logw:.3f}  p={p if p is None else round(float(p), 4)}")
        if show_state_keys is None:
            print("state:", _safe_repr(h.state))
        else:
            slim = {k: h.state.get(k) for k in show_state_keys}
            print("state:", _safe_repr(slim))


def print_hypothesis_trace(h: Hypothesis, max_events: int = 80) -> None:
    """
    Print per-hypothesis detailed trace events (proposals + scorer events).
    """
    print("\n=== HYPOTHESIS TRACE ===")
    for i, e in enumerate(h.trace[:max_events]):
        _print_event(i, e)
    if len(h.trace) > max_events:
        print(f"... ({len(h.trace) - max_events} more events)")


def _print_event(i: int, e: TraceEvent) -> None:
    msg = f"#{i:02d} [{e.kind}]"
    if e.message:
        msg += f" {e.message}"
    print(msg)
    if e.data:
        print("   data:", _safe_repr(e.data))


def _safe_repr(x: Any, max_len: int = 400) -> str:
    """
    Avoid dumping huge prompts or JSON blobs in console.
    """
    s = repr(x)
    if len(s) > max_len:
        return s[:max_len] + "...(truncated)"
    return s
