"""Stage 2: LLM refines the candidates, then code validates every answer.

The model is good at judging drama and standalone clarity, but it can invent
timestamps. So code never trusts its times: each segment is snapped to real
event boundaries in the script, cuts inside a dialogue line are repaired, and
length rules are enforced. Anything that cannot be repaired is flagged.
"""
from __future__ import annotations

import re
from pathlib import Path

from .heuristics import Candidate, score_window
from .llm import Backend, LLMError
from .parse import Episode, fmt

PROMPT = (Path(__file__).parent / "prompts" / "refine.txt").read_text(encoding="utf-8")
SNAP_TOLERANCE = 3.0   # seconds; further than this = the model made the time up
NEXT_EP_SECONDS = 45.0  # how much of the next episode the model may borrow from

TIME_RE = re.compile(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*$")


def parse_time(s) -> float | None:
    if isinstance(s, (int, float)):
        return float(s)
    m = TIME_RE.match(str(s))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def build_prompt(show_name: str, show: list[Episode], ep: Episode, cands: list[Candidate], n_promos: int) -> str:
    nxt = next((e for e in show if e.number == ep.number + 1), None)
    lines = [e.line() for e in ep.events]
    if nxt:
        lines.append(f"--- NEXT EPISODE (E{nxt.number}) OPENING ---")
        lines += [e.line() for e in nxt.events if e.start < NEXT_EP_SECONDS]
    sugg = "\n".join(
        f"- E{c.episode} {fmt(c.start)}-{fmt(c.end)} ({c.duration:.0f}s) rule score {c.score}; "
        f"hook cue: {c.hook_event[:110]} | ending cue: {c.cliff_event[:110]}"
        for c in cands) or "- (none)"
    return PROMPT.format(show=show_name, premise=show[0].logline, episode=ep.number, logline=ep.logline,
                         continuity=ep.continuity, suggestions=sugg, transcript="\n".join(lines), n_promos=n_promos)


def snap_segment(ep: Episode, a: float, b: float) -> tuple[float, float, list[str]]:
    """Move a, b onto real event boundaries and never split a dialogue line."""
    flags = []
    starts = [e.start for e in ep.events]
    ends = [e.end for e in ep.events]
    na = min(starts, key=lambda t: abs(t - a))
    nb = min(ends, key=lambda t: abs(t - b))
    if abs(na - a) > SNAP_TOLERANCE or abs(nb - b) > SNAP_TOLERANCE:
        flags.append("llm_time_not_in_script")
    for e in ep.events:
        if e.kind == "dialogue" and e.start < na < e.end:
            na = e.start
            flags.append("fixed_cut_into_line")
        if e.kind == "dialogue" and e.start < nb < e.end:
            nb = e.end
            flags.append("fixed_cut_out_of_line")
    return na, nb, flags


def validate(show: list[Episode], ep: Episode, promo: dict) -> dict:
    by_num = {e.number: e for e in show}
    segs, flags = [], []
    for s in promo.get("segments", [])[:3]:
        num = int(s.get("episode", ep.number))
        target = by_num.get(num)
        a, b = parse_time(s.get("start")), parse_time(s.get("end"))
        if target is None or num not in (ep.number, ep.number + 1) or a is None or b is None or b <= a:
            flags.append("invalid_segment_dropped")
            continue
        a, b, f = snap_segment(target, a, b)
        flags += f
        segs.append({"episode": num, "start": a, "end": b})
    segs.sort(key=lambda s: (s["episode"], s["start"]))
    # merge overlapping segments in the same episode
    merged = []
    for s in segs:
        if merged and merged[-1]["episode"] == s["episode"] and s["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
        else:
            merged.append(s)
    total = sum(s["end"] - s["start"] for s in merged)
    if merged and not 20 <= total <= 60:
        flags.append(f"length_out_of_range_{total:.0f}s")
    return {**promo, "segments": merged, "duration": round(total, 2), "flags": sorted(set(flags))}


def refine_episode(backend: Backend, show_name: str, show: list[Episode], ep: Episode,
                   cands: list[Candidate], n_promos: int = 2) -> tuple[list[dict], dict]:
    prompt = build_prompt(show_name, show, ep, cands, n_promos)
    raw = backend.complete_json(prompt)
    checked = [validate(show, ep, p) for p in raw.get("promos", []) if isinstance(p, dict)]
    promos, rejected = split_rejected(checked)
    raw["_rejected_by_validator"] = rejected  # kept in the raw output so failures stay visible
    return promos, raw


def split_rejected(checked: list[dict]) -> tuple[list[dict], list[dict]]:
    """A promo whose times are not in the script is dropped, not repaired: snapping a
    made-up time onto the nearest real line would produce a clip nobody chose."""
    keep, rejected = [], []
    for p in checked:
        if not p["segments"] or "llm_time_not_in_script" in p["flags"]:
            rejected.append({"promo": p, "reason": "no valid segments" if not p["segments"] else "timestamp not in script"})
        else:
            keep.append(p)
    return keep, rejected


def candidate_to_promo(c: Candidate) -> dict:
    """Fallback when there is no LLM: describe a rule-based window in the same schema."""
    return {
        "segments": [{"episode": c.episode, "start": c.start, "end": c.end}],
        "hook": c.hook_event, "cliffhanger": c.cliff_event,
        "why": f"Rule-based pick: hook cue {c.hook}, ending cue {c.cliff}, dialogue density {c.density}.",
        "hook_score": None, "cliffhanger_score": None, "standalone_score": None,
        "confusion_risk": "not assessed (offline mode)",
        "duration": round(c.duration, 2), "flags": list(c.flags),
    }
