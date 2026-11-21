"""Parse Kuku screenplay .txt files into timed events.

Each episode file has a header (EPISODE NUMBER, LOGLINE, CONTINUITY NOTES),
then scenes made of blocks like:

    **JULIANA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"

Timestamps are local to the episode (mm:ss.xx).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

LABEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
TIME_RE = re.compile(
    r"Start:\s*\((\d+):(\d+(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d+):(\d+(?:\.\d+)?)\)"
)
SCENE_RE = re.compile(r"^SCENE\s+(\d+)")
HEADING_RE = re.compile(r"^Scene heading:\s*(.*)")
EP_NUM_RE = re.compile(r"episode-(\d+)", re.I)

NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sound", "SOUND": "sound"}


def to_sec(mm: str, ss: str) -> float:
    return int(mm) * 60 + float(ss)


def fmt(t: float) -> str:
    """Seconds -> mm:ss.xx (same style as the scripts)."""
    m, s = divmod(round(t, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Event:
    episode: int
    idx: int
    kind: str          # dialogue | action | camera | sound
    speaker: str       # character name for dialogue, else the label
    tone: str          # text inside the parentheses, e.g. "Cold, Commanding"
    start: float
    end: float
    text: str
    scene: int
    location: str

    def line(self) -> str:
        """One transcript line for prompts and reports."""
        who = self.speaker + (f" ({self.tone})" if self.tone else "")
        return f"[E{self.episode} {fmt(self.start)}-{fmt(self.end)}] {who}: {self.text}"


@dataclass
class Episode:
    number: int
    logline: str
    continuity: str
    path: str
    events: list[Event] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((e.end for e in self.events), default=0.0)


def parse_episode(path: Path) -> Episode:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    m = EP_NUM_RE.search(path.name)
    number = int(m.group(1)) if m else 0
    logline = continuity = ""
    for ln in lines[:10]:
        if ln.startswith("EPISODE LOGLINE:"):
            logline = ln.split(":", 1)[1].strip()
        elif ln.startswith("CONTINUITY NOTES:"):
            continuity = ln.split(":", 1)[1].strip()

    ep = Episode(number, logline, continuity, str(path))
    scene, location = 0, ""
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if (sm := SCENE_RE.match(ln)):
            scene = int(sm.group(1))
        elif (hm := HEADING_RE.match(ln)):
            location = hm.group(1).strip()
        elif (lm := LABEL_RE.match(ln)):
            label = lm.group(1).strip()
            # find the timing line right after the label
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            tm = TIME_RE.search(lines[j]) if j < len(lines) else None
            if tm:
                start = to_sec(tm.group(1), tm.group(2))
                end = to_sec(tm.group(3), tm.group(4))
                body = []
                k = j + 1
                while k < len(lines) and lines[k].strip() and not LABEL_RE.match(lines[k].strip()):
                    body.append(lines[k].strip())
                    k += 1
                name, tone = label, ""
                if "(" in label and label.endswith(")"):
                    name, tone = label.split("(", 1)
                    name, tone = name.strip(), tone[:-1].strip()
                kind = NON_DIALOGUE.get(name.upper(), "dialogue")
                ep.events.append(Event(number, len(ep.events), kind, name, tone,
                                       start, max(end, start), " ".join(body), scene, location))
                i = k
                continue
        i += 1
    ep.events.sort(key=lambda e: (e.start, e.idx))
    for n, e in enumerate(ep.events):
        e.idx = n
    return ep


def load_show(folder: str | Path) -> list[Episode]:
    folder = Path(folder)
    files = sorted(folder.rglob("*.txt"), key=lambda p: (EP_NUM_RE.search(p.name) and int(EP_NUM_RE.search(p.name).group(1))) or 0)
    eps = [parse_episode(p) for p in files]
    return [e for e in eps if e.events]


if __name__ == "__main__":
    import sys, json
    show = load_show(sys.argv[1])
    for ep in show:
        kinds = {}
        for e in ep.events:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        print(ep.number, fmt(ep.duration), kinds)
