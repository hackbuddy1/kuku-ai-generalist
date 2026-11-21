"""Stage 3: score, de-duplicate, rank and quality-check promos (plain code).

Final score mixes two independent opinions:
  * the LLM's own 1-10 scores (hook, cliffhanger, standalone)
  * the rule scorer re-run on the final cut points
When they disagree a lot, the promo is flagged for a human to look at.
"""
from __future__ import annotations

from .heuristics import cliff_strength, hook_strength, REPLY_OPENER
from .parse import Episode, fmt

SERIOUS = ("llm_time_not_in_script", "length_out_of_range", "invalid_segment_dropped", "cuts_into_line", "cuts_out_of_line")


def rule_check(show: list[Episode], promo: dict) -> dict:
    by_num = {e.number: e for e in show}
    first, last = promo["segments"][0], promo["segments"][-1]
    ep_a, ep_b = by_num[first["episode"]], by_num[last["episode"]]
    head = [e for e in ep_a.events if first["start"] - 1e-6 <= e.start < first["start"] + 3.0]
    tail = [e for e in ep_b.events if last["end"] - 4.0 < e.end <= last["end"] + 1e-6]
    hook = max((hook_strength(e) for e in head), default=0.0)
    cliff = max((cliff_strength(e) for e in tail), default=0.0)
    if last["end"] >= ep_b.duration - 1.5:
        cliff += 3.0
    flags = []
    first_line = next((e for e in ep_a.events if e.kind == "dialogue" and e.start >= first["start"] - 1e-6), None)
    if first_line and first_line.start < first["start"] + 6 and REPLY_OPENER.match(first_line.text):
        flags.append("opens_on_reply")
        if first_line.text.lstrip('"').startswith(("...", "…")):
            flags.append("opens_mid_sentence")
    if hook < 2:
        flags.append("rule_weak_hook")
    if cliff < 2:
        flags.append("rule_weak_ending")
    norm = 0.5 * min(hook, 8) / 8 + 0.5 * min(cliff, 8) / 8
    return {"rule_hook": round(hook, 2), "rule_cliff": round(cliff, 2), "rule_score": round(norm, 3), "rule_flags": flags}


def _llm_score(p: dict) -> float | None:
    vals = [p.get(k) for k in ("hook_score", "cliffhanger_score", "standalone_score")]
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    return sum(vals) / (10 * len(vals)) if vals else None


def _overlap(p: dict, q: dict) -> float:
    inter = 0.0
    for s in p["segments"]:
        for t in q["segments"]:
            if s["episode"] == t["episode"]:
                inter += max(0.0, min(s["end"], t["end"]) - max(s["start"], t["start"]))
    return inter / max(min(p["duration"], q["duration"]), 1e-6)


def score_and_rank(show: list[Episode], promos: list[dict]) -> list[dict]:
    for p in promos:
        p.update(rule_check(show, p))
        llm = _llm_score(p)
        p["llm_score"] = round(llm, 3) if llm is not None else None
        base = 0.65 * llm + 0.35 * p["rule_score"] if llm is not None else p["rule_score"]
        all_flags = p["flags"] + p["rule_flags"]
        penalty = sum(0.15 for f in all_flags if f.startswith(SERIOUS))
        penalty += 0.03 * sum(f.startswith("fixed_") for f in all_flags)
        penalty += 0.08 * ("opens_on_reply" in all_flags) + 0.15 * ("opens_mid_sentence" in all_flags)
        qc = []
        if llm is not None and abs(llm - p["rule_score"]) > 0.4:
            qc.append("llm_vs_rules_disagree")
        if isinstance(p.get("standalone_score"), (int, float)) and p["standalone_score"] < 6:
            qc.append("may_confuse_new_viewer")
        if "opens_mid_sentence" in all_flags:
            qc.append("opens_mid_sentence")
        if any(f.startswith(SERIOUS) for f in all_flags):
            qc.append("broken_cut")
        p["qc"] = qc
        p["needs_human_review"] = bool(qc)
        p["final_score"] = round(base - penalty, 3)

    promos.sort(key=lambda p: p["final_score"], reverse=True)
    kept = []
    for p in promos:  # drop near-duplicates of a better promo (same story beat)
        dup = next((k for k in kept if _overlap(p, k) > 0.5), None)
        if dup:
            dup.setdefault("alternates", []).append(segments_str(p))
        else:
            kept.append(p)

    # Posting order: best first, but spread the first 10 across different episodes
    order, used_eps, rest = [], set(), []
    for p in kept:
        eps = {s["episode"] for s in p["segments"]}
        if len(order) < 10 and not (eps & used_eps) and not p["needs_human_review"]:
            order.append(p)
            used_eps |= eps
        else:
            rest.append(p)
    ranked = order + sorted(rest, key=lambda p: p["final_score"], reverse=True)
    for i, p in enumerate(ranked, 1):
        p["rank"] = i
        p["post_first_reason"] = post_reason(p)
    return ranked


def segments_str(p: dict) -> str:
    return " + ".join(f"E{s['episode']} {fmt(s['start'])}-{fmt(s['end'])}" for s in p["segments"])


def post_reason(p: dict) -> str:
    bits = []
    if (p.get("hook_score") or 0) >= 8 or p["rule_hook"] >= 5:
        bits.append("very strong opening hook")
    if (p.get("cliffhanger_score") or 0) >= 8 or p["rule_cliff"] >= 6:
        bits.append("sharp unresolved ending")
    if (p.get("standalone_score") or 0) >= 8:
        bits.append("clear to a new viewer")
    if min(s["episode"] for s in p["segments"]) <= 5:
        bits.append("early episode, so new viewers land at the start of the story")
    if p["needs_human_review"]:
        bits.append("needs a human check first: " + ", ".join(p["qc"]))
    return "; ".join(bits) or "solid but unremarkable"
