"""Stage 1: cheap, rule-based candidate finder (no AI, runs offline).

Idea: a promo is a 20-60 s window that
  * opens on a hook (strong cue in its first 3 s),
  * closes on a cliffhanger (strong cue in its last ~4 s, or the episode's own ending),
  * is dense with dialogue/action (no dead air),
  * does not start halfway through an exchange.

We score every (start, end) pair built from event boundaries and keep the best
non-overlapping windows per episode. These are suggestions for the LLM stage and
also the fallback output when no LLM is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .parse import Episode, Event

# Visual / physical shocks (looked for in ACTION, CAMERA, SOUND text)
HOOK_ACTION = {
    r"\bslap": 3, r"\bstrik": 2, r"\bpunch": 3, r"\bkick": 2, r"\bshov": 2, r"\bpush(es|ed)? (her|him)": 2,
    r"\bgrab": 1.5, r"\bchok": 3, r"\bstrangl": 3, r"\bknife|\bdagger|\bblade|\bsword|\bpistol|\bgun\b": 3,
    r"\bblood": 2.5, r"\bcollaps": 2.5, r"\bfaint": 2, r"\bscream": 2, r"\bgasp": 1.5, r"\bsob": 1,
    r"\bkiss": 2, r"\bkneel": 1.5, r"\bthrow|\bhurl": 2, r"\bshatter|\bsmash|\bcrash": 2.5,
    r"\brip(s|ped)?\b|\btear(s|ing)? (off|open|up)": 2, r"\bburst": 2, r"\bslam": 2, r"\bunmask|\breveal": 2,
    r"\bfire\b|\bflames?": 2, r"\bpoison": 3, r"\bfall(s|ing)?\b": 1.5, r"\bsnatch": 1.5,
    r"\bdrag": 1.5, r"\bcuff|\barrest": 2, r"\bpregnan": 3, r"\bsilence": 1, r"\bfreez": 1,
}
# Shocking lines (looked for in dialogue)
HOOK_LINE = {
    r"\bdivorce": 3, r"\bmarr(y|ied|iage)": 1.5, r"\bkill": 3, r"\bdead\b|\bdie\b|\bmurder": 3,
    r"\bpregnan|\bbaby|\bchild\b": 2.5, r"\bliar\b|\blie[sd]?\b": 2, r"\bbetray": 2.5, r"\bduke\b": 1.5,
    r"\bnever\b": 1, r"\bhow dare": 3, r"\bget out": 2.5, r"\bkneel": 2, r"\bguards?\b": 1.5,
    r"\bworthless|\bplain|\bugly|\bdisgrac|\bshame": 2, r"\bwho are you": 2, r"\bbastard|\bwhore|\bharlot": 2.5,
    r"\bfather\b|\bmother\b": 0.5, r"\bstop\b": 1, r"\bmine\b": 1.5, r"\bwife\b|\bhusband\b": 1.5,
}
HOT_TONES = r"furious|enraged|shock|horrif|scream|menac|venom|cold|cruel|desperat|sneer|mock|threat|livid|terrif|panic|defiant|commanding|icy|outrag|hysteric"

# Unresolved endings
CLIFF_CUES = {
    r"cliffhanger": 4, r"cut to black": 2, r"ominous": 1.5, r"freeze": 1.5, r"\bsilence": 1,
    r"\bburst|\bbarges?|\bdoor(s)? (fly|swing|slam|open)": 2.5, r"\benters?\b|\barriv": 1.5,
    r"\bshock|\bstunned|\bhorrif|\bwiden": 1.5, r"\bdrum|\bsting|\bswell": 1.5, r"\breveal": 2,
    r"\bknife|\bblade|\bsword|\bgun\b|\bpistol": 2, r"\bcollaps|\bfaint": 2, r"\bkiss": 1,
}
# Lines that start mid-conversation (bad openers for a stranger)
REPLY_OPENER = re.compile(r'^"?(\.\.\.|…|(yes|no|because|and|but|so|then|that|it|he|she|they|this|indeed|very well|as you)\b)', re.I)


def _score(text: str, table: dict[str, float]) -> float:
    t = text.lower()
    return sum(w for pat, w in table.items() if re.search(pat, t))


def hook_strength(e: Event) -> float:
    if e.kind == "dialogue":
        s = _score(e.text, HOOK_LINE)
        if re.search(HOT_TONES, e.tone.lower()):
            s += 1.5
        if "!" in e.text:
            s += 1
        if "?" in e.text:
            s += 0.5
        if len(e.text.split()) <= 8:  # short punchy lines land in 3 s
            s += 0.5
        return s
    return _score(e.text, HOOK_ACTION) * (0.6 if e.kind == "camera" else 1.0)


def cliff_strength(e: Event) -> float:
    s = _score(e.text + " " + e.tone, CLIFF_CUES)
    if e.kind == "dialogue":
        if e.text.rstrip('"').endswith(("?", "...", "—", "-")):
            s += 1.5
        s += 0.5 * _score(e.text, HOOK_LINE) ** 0.5
    return s


@dataclass
class Candidate:
    episode: int
    start: float
    end: float
    hook: float
    cliff: float
    density: float
    score: float
    hook_event: str
    cliff_event: str
    flags: list[str]

    @property
    def duration(self) -> float:
        return self.end - self.start


def _window_events(ep: Episode, a: float, b: float) -> list[Event]:
    return [e for e in ep.events if e.start >= a - 1e-6 and e.end <= b + 1e-6]


def score_window(ep: Episode, a: float, b: float) -> Candidate | None:
    evs = _window_events(ep, a, b)
    if not evs:
        return None
    head = [e for e in evs if e.start < a + 3.0]
    tail = [e for e in evs if e.end > b - 4.0]
    hook_ev = max(head, key=hook_strength, default=None)
    cliff_ev = max(tail, key=cliff_strength, default=None)
    hook = hook_strength(hook_ev) if hook_ev else 0.0
    cliff = cliff_strength(cliff_ev) if cliff_ev else 0.0
    at_ep_end = b >= ep.duration - 1.5
    if at_ep_end:
        cliff += 3.0  # the writers already end every episode on a cliffhanger

    talk = sum(e.end - e.start for e in evs if e.kind in ("dialogue", "action"))
    density = min(talk / (b - a), 1.0)

    flags = []
    first_line = next((e for e in evs if e.kind == "dialogue"), None)
    if first_line and first_line.start < a + 6 and REPLY_OPENER.match(first_line.text):
        flags.append("opens_on_reply")
        if first_line.text.lstrip('"').startswith(("...", "…")):
            flags.append("opens_mid_sentence")
    # Splitting a line of dialogue in half is never OK
    if any(e.kind == "dialogue" and e.start < a < e.end for e in ep.events):
        flags.append("cuts_into_line")
    if any(e.kind == "dialogue" and e.start < b < e.end for e in ep.events):
        flags.append("cuts_out_of_line")
    if hook < 2:
        flags.append("weak_hook")
    if cliff < 2:
        flags.append("weak_ending")

    dur = b - a
    length_bonus = 1.0 if 25 <= dur <= 50 else 0.7
    score = (1.2 * min(hook, 8) + 1.0 * min(cliff, 8) + 3.0 * density) * length_bonus
    score -= 2.0 * ("opens_on_reply" in flags) + 4.0 * ("opens_mid_sentence" in flags) + 4.0 * ("cuts_into_line" in flags) + 4.0 * ("cuts_out_of_line" in flags)
    return Candidate(ep.number, a, b, round(hook, 2), round(cliff, 2), round(density, 2), round(score, 2),
                     hook_ev.line() if hook_ev else "", cliff_ev.line() if cliff_ev else "", flags)


def find_candidates(ep: Episode, min_len=20.0, max_len=60.0, keep=4) -> list[Candidate]:
    starts = sorted({e.start for e in ep.events if e.kind in ("dialogue", "action", "camera")})
    ends = sorted({e.end for e in ep.events})
    cands = []
    for a in starts:
        for b in ends:
            if min_len <= b - a <= max_len:
                c = score_window(ep, a, b)
                if c:
                    cands.append(c)
    cands.sort(key=lambda c: c.score, reverse=True)
    # non-maximum suppression: drop windows overlapping a better one by >40%
    chosen: list[Candidate] = []
    for c in cands:
        if all(_overlap(c, k) < 0.4 for k in chosen):
            chosen.append(c)
        if len(chosen) >= keep:
            break
    return chosen


def _overlap(x: Candidate, y: Candidate) -> float:
    inter = max(0.0, min(x.end, y.end) - max(x.start, y.start))
    return inter / min(x.duration, y.duration)
