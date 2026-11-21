"""Resolve script speaker labels into real characters, for AI dubbing voice casting.

Problem: AI-written screenplays label the same character in different ways ("JULIANA" / "LADY JULIANA",
"GERALD" / "THE FORMER GROOM") and sometimes reuse one label for different people across story arcs.
A dubbing pipeline assigns one voice per label, so every alias becomes a wrong or new voice.

Pipeline
  1. code   : stats per label, split each label into "runs" (episode stretches; a gap > GAP episodes starts a new run)
  2. code   : hard constraint: labels that both speak in the same scene are different people
  3. code   : rule A, same proper name with/without a title (LADY JULIANA -> JULIANA)
  4. code   : rule B, the script names the label in an action line ("Blond Nobleman Rupert", "Rupert, the Blond Nobleman")
  5. LLM    : only descriptive labels left (LADY IN GOLD GOWN, MYSTERIOUS BEARDED MAN...) get an evidence pack
              and pick a candidate or DISTINCT; code checks the candidate is allowed and the quoted evidence exists
  6. code   : low-confidence or unverifiable decisions go to a human review queue instead of being applied
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

GAP = 8
TITLES = {"LADY", "LORD", "SIR", "THE", "MR", "MRS", "MISS", "MADAM", "DAME", "YOUNG", "OLD"}
ROLE_WORDS = {
    "PRIEST", "HIGH", "FATHER", "MOTHER", "NOBLEMAN", "NOBLEWOMAN", "GUARD", "ROYAL", "CONSTABLE", "POLICE", "MAID",
    "JUDGE", "THUG", "BALD", "TALL", "THIRD", "CAP", "WITH", "MAN", "WOMAN", "VOICE", "OFF-SCREEN", "OFFICIAL", "CROWN",
    "VILLAGER", "ELDER", "FIGHTER", "SCOUT", "DRIVER", "CART", "LOYALIST", "SENTRY", "FOOTMAN", "BUTLER", "HOUSEKEEPER",
    "INFORMANT", "TOWNSMAN", "CONSPIRATOR", "MATRIARCH", "NOBLE", "STRANGER", "BEARDED", "SCARRED", "MYSTERIOUS",
    "GROOM", "FORMER", "GOLD", "GOWN", "IN", "BLOND", "ARISTOCRATIC", "PEASANT", "DOWAGER", "DUCHESS", "DUKE", "KING",
    "QUEEN", "PRINCE", "PRINCESS", "COUNT", "COUNTESS", "BARON", "SERVANT", "SOLDIER", "CAPTAIN", "DOCTOR", "NURSE",
    "BOY", "GIRL", "CHILD", "OF", "AND", "A", "AN", "NARRATOR", "CROWD", "ALL", "LORD", "LADY", "THE",
}
NON_PERSON = {"OFF-SCREEN VOICE", "NARRATOR", "CROWD", "ALL"}


def words(label: str) -> list[str]:
    return re.findall(r"[A-Z][A-Z'\-]*", label.upper())


def proper_tokens(label: str) -> list[str]:
    """Tokens that look like a personal name (not a title, not a role word, not a possessive)."""
    return [w for w in words(label) if w not in ROLE_WORDS and w not in TITLES and not w.endswith("'S")]


def is_named(label: str) -> bool:
    return bool(proper_tokens(label))


@dataclass
class Run:
    label: str
    episodes: list[int]
    lines: int = 0
    scenes: set = field(default_factory=set)

    @property
    def key(self) -> str:
        e = self.episodes
        return f"{self.label}@E{e[0]}" + (f"-{e[-1]}" if e[-1] != e[0] else "")


class Resolver:
    def __init__(self, show):
        self.show = show
        self.ep = {e.number: e for e in show}
        self.speaks = defaultdict(list)         # label -> dialogue events
        self.scene_speakers = defaultdict(set)  # (ep, scene) -> labels
        for ep in show:
            for ev in ep.events:
                if ev.kind == "dialogue":
                    self.speaks[ev.speaker].append(ev)
                    self.scene_speakers[(ep.number, ev.scene)].add(ev.speaker)
        self.runs: list[Run] = []
        for label, evs in self.speaks.items():
            eps = sorted({e.episode for e in evs})
            chunk = [eps[0]]
            for n in eps[1:]:
                if n - chunk[-1] > GAP:
                    self.runs.append(self._mk_run(label, chunk))
                    chunk = []
                chunk.append(n)
            self.runs.append(self._mk_run(label, chunk))
        self.decisions: dict[str, dict] = {}

    def _mk_run(self, label, eps):
        evs = [e for e in self.speaks[label] if e.episode in eps]
        return Run(label, eps, len(evs), {(e.episode, e.scene) for e in evs})

    # ---- constraints -------------------------------------------------------------------------
    def co_speak(self, a: str, b: str) -> list:
        return [k for k, v in self.scene_speakers.items() if a in v and b in v]

    # ---- rule A: same proper name ------------------------------------------------------------
    def rule_title(self):
        named = [l for l in self.speaks if is_named(l)]
        by_name = defaultdict(list)
        for l in named:
            by_name[" ".join(proper_tokens(l))].append(l)
        for name, labels in by_name.items():
            if len(labels) < 2:
                continue
            canon = max(labels, key=lambda l: (len(self.speaks[l]), -len(l)))  # most lines, shortest label
            for l in labels:
                if l == canon:
                    continue
                g1, g2 = {"LADY", "LORD"} & set(words(l)), {"LADY", "LORD"} & set(words(canon))
                if g1 and g2 and g1 != g2:
                    continue  # Lady X and Lord X are a couple, not one person
                clash = self.co_speak(l, canon)
                for run in [r for r in self.runs if r.label == l]:
                    self.decisions[run.key] = {
                        "run": run.key, "label": l, "episodes": run.episodes, "lines": run.lines,
                        "canonical": None if clash else canon, "method": "rule:same_name",
                        "confidence": 0.0 if clash else 0.95,
                        "evidence": f"'{l}' and '{canon}' share the name '{name}'" + (f"; BUT they speak in the same scene {clash[:2]}" if clash else ""),
                        "review": bool(clash)}

    # ---- rule B: the script names the label in an action/camera line -------------------------
    def rule_apposition(self):
        names = {}
        for l in self.speaks:
            if is_named(l) and len(self.speaks[l]) >= 5:
                for t in proper_tokens(l):
                    names[t.title()] = self.canonical_of(l)
        for run in self.runs:
            if run.key in self.decisions or is_named(run.label) or run.label in NON_PERSON:
                continue
            phrase = run.label.title().replace("The ", "")
            pat_after = re.compile(rf"\b{re.escape(phrase)}\s+(?:named\s+)?([A-Z][a-z]+)\b(?!,\s*(?:and|the)\b)")
            pat_before = re.compile(rf"\b([A-Z][a-z]+),\s+the\s+{re.escape(phrase)}\b")
            hits = Counter()
            quote = {}
            for n in run.episodes:
                for ev in self.ep[n].events:
                    if ev.kind in ("camera", "action"):
                        for m in list(pat_after.finditer(ev.text)) + list(pat_before.finditer(ev.text)):
                            before = ev.text[:m.start()].split()
                            if m.re is pat_after and before and before[-1][:1].isupper() and before[-1].lower() not in ("the", "a", "an"):
                                continue  # 'Nobleman' inside 'Blond Nobleman Rupert' belongs to a longer label
                            nm = m.group(1).title()
                            if nm in names:
                                hits[names[nm]] += 1
                                quote.setdefault(names[nm], ev.line())
            if hits:
                canon, k = hits.most_common(1)[0]
                if not self.co_speak(run.label, canon) or all(s[0] not in run.episodes for s in self.co_speak(run.label, canon)):
                    self.decisions[run.key] = {
                        "run": run.key, "label": run.label, "episodes": run.episodes, "lines": run.lines,
                        "canonical": canon, "method": "rule:script_names_it", "confidence": 0.9,
                        "evidence": quote[canon][:240], "review": False}

    def canonical_of(self, label: str) -> str:
        for d in self.decisions.values():
            if d["label"] == label and d["canonical"]:
                return d["canonical"]
        return label

    # ---- LLM stage ---------------------------------------------------------------------------
    def unresolved(self) -> list[Run]:
        return [r for r in self.runs if r.key not in self.decisions and not is_named(r.label)
                and r.label not in NON_PERSON and r.lines >= 1]

    def candidates(self, run: Run) -> list[str]:
        cands = Counter()
        for l, evs in self.speaks.items():
            if l == run.label or l in NON_PERSON:
                continue
            canon = self.canonical_of(l)
            near = [e for e in evs if min(abs(e.episode - n) for n in run.episodes) <= 12]
            if not near or (not is_named(l) and canon == l and len(evs) < 15):
                continue  # only real named characters (or big recurring roles) are candidates
            cands[canon] += len(near)
        # a candidate is impossible if ANY of its labels (aliases included) talks with this speaker in these episodes
        out = []
        for c, _ in cands.most_common():
            group = {c} | {d["label"] for d in self.decisions.values() if d["canonical"] == c and not d["review"]}
            if any(k[0] in run.episodes for g in group - {run.label} for k in self.co_speak(run.label, g)):
                continue
            out.append(c)
        return out[:10]

    def evidence_pack(self, run: Run, cands: list[str]) -> str:
        out = []
        for n in sorted({max(1, run.episodes[0] - 1), *run.episodes})[:6]:
            ep = self.ep.get(n)
            if ep:
                out.append(f"EPISODE {n} LOGLINE: {ep.logline}\nCONTINUITY: {ep.continuity}")
        out.append(f"\nLINES SPOKEN BY '{run.label}':")
        out += [e.line() for e in self.speaks[run.label] if e.episode in run.episodes][:10]
        out.append(f"\nACTION/CAMERA LINES IN SCENES WHERE '{run.label}' SPEAKS:")
        act = [ev for (n, sc) in sorted(run.scenes) for ev in self.ep[n].events
               if ev.scene == sc and ev.kind in ("action", "camera")]
        key = [w.lower() for w in words(run.label) if w not in TITLES][:3]
        act.sort(key=lambda ev: -sum(k in ev.text.lower() for k in key))
        out += [ev.line() for ev in act[:14]]
        out.append("\nCANDIDATES (sample lines):")
        for c in cands:
            labels = {d["label"] for d in self.decisions.values() if d["canonical"] == c} | {c}
            sample = [e for l in labels for e in self.speaks.get(l, [])][:2]
            out.append(f"- {c}: " + " | ".join(e.text[:90] for e in sample))
        return "\n".join(out)

    def llm_resolve(self, llm, log=print):
        prompt_tpl = (
            "You are building the character list (cast sheet) for AI dubbing of a vertical microdrama. Each distinct "
            "character gets one voice, so wrongly merging two people is as bad as missing a merge.\n"
            "The screenplay labels one speaker with a description instead of a name: '{label}' (episodes {eps}).\n"
            "Decide if this speaker is the same person as ONE of the candidates, or a DISTINCT character.\n"
            "Rules: choose only from the candidates or DISTINCT. Base it on the evidence; quote the single most "
            "decisive sentence EXACTLY as it appears in the evidence. If the evidence is not clear, say DISTINCT or "
            "give low confidence.\n\nCANDIDATES: {cands}\n\nEVIDENCE\n{evidence}\n\n"
            'Return ONLY JSON: {{"decision": "<candidate name or DISTINCT>", "confidence": <0.0-1.0>, '
            '"evidence_quote": "<exact sentence from the evidence>", "reason": "<one line>"}}')
        for run in self.unresolved():
            cands = self.candidates(run)
            pack = self.evidence_pack(run, cands)
            prompt = prompt_tpl.format(label=run.label, eps=run.episodes, cands=", ".join(cands) or "(none)", evidence=pack)
            try:
                raw = llm.generate_json(prompt)
            except Exception as e:  # never lose the run: send it to review
                log(f"  {run.key}: LLM failed ({str(e)[:80]}), sent to review")
                self.decisions[run.key] = {"run": run.key, "label": run.label, "episodes": run.episodes, "lines": run.lines,
                                           "canonical": None, "method": "llm_failed", "confidence": 0.0,
                                           "evidence": str(e)[:200], "review": True}
                continue
            dec = str(raw.get("decision", "DISTINCT")).strip().upper()
            conf = float(raw.get("confidence", 0) or 0)
            quote = str(raw.get("evidence_quote", ""))
            flags = []
            if dec != "DISTINCT" and dec not in cands:
                flags.append("not_a_candidate")
                dec, conf = "DISTINCT", 0.0
            norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
            if quote and norm(quote)[:60] not in norm(pack):
                flags.append("quote_not_in_evidence")
                conf = min(conf, 0.4)
            review = conf < 0.7 or bool(flags)
            self.decisions[run.key] = {
                "run": run.key, "label": run.label, "episodes": run.episodes, "lines": run.lines,
                "canonical": None if dec == "DISTINCT" else dec, "method": "llm", "confidence": round(conf, 2),
                "evidence": quote[:240], "reason": str(raw.get("reason", ""))[:200], "flags": flags,
                "candidates": cands, "review": review, "model": f"{llm.name}:{llm.model}", "raw_model_output": raw}
            log(f"  {run.key}: {dec} ({conf:.2f}){' -> REVIEW' if review else ''}{' ' + str(flags) if flags else ''}")

    # ---- outputs -----------------------------------------------------------------------------
    def _final(self, label: str, seen=None) -> str:
        """Follow merge pointers (A->B->C). A loop (A->B, B->A) resolves to the label with more lines."""
        seen = seen or []
        if label in seen:
            return max(seen, key=lambda l: len(self.speaks.get(l, [])))
        nxt = [d["canonical"] for d in self.decisions.values()
               if d["label"] == label and d["canonical"] and not d["review"]]
        return self._final(nxt[0], seen + [label]) if nxt else label

    def alias_map(self) -> dict:
        """label@episodes -> voice/character. Only confident decisions are applied; the rest keep their own voice."""
        amap = {}
        for run in self.runs:
            d = self.decisions.get(run.key)
            applied = self._final(d["canonical"], [run.label]) if d and d["canonical"] and not d["review"] else None
            if applied == run.label:
                applied = None
            amap[run.key] = {"label": run.label, "episodes": run.episodes, "lines": run.lines,
                             "character": applied or run.label, "merged": bool(applied),
                             "needs_review": bool(d and d["review"]), "method": d["method"] if d else "own_name"}
        return amap

    def voice_profile(self, labels: set[str]) -> dict:
        tones = Counter()
        for l in labels:
            for e in self.speaks[l]:
                for t in re.split(r"[,/]", e.tone):
                    if t.strip():
                        tones[t.strip().lower()] += 1
        first = sorted(labels, key=lambda l: -len(self.speaks[l]))[0]
        name = (proper_tokens(first) or words(first))[0].title()
        he = she = 0
        for ep in self.show:
            for ev in ep.events:
                if ev.kind in ("action", "camera") and name in ev.text:
                    after = ev.text.split(name, 1)[1][:80].lower()
                    he += len(re.findall(r"\b(he|his|him)\b", after))
                    she += len(re.findall(r"\b(she|her|hers)\b", after))
        gender = "male" if he > she * 1.5 else "female" if she > he * 1.5 else "unclear"
        return {"gender_guess": gender, "he_she_counts": [he, she], "top_tones": [t for t, _ in tones.most_common(5)]}

    def cast_sheet(self) -> list[dict]:
        amap = self.alias_map()
        groups = defaultdict(list)
        for k, v in amap.items():
            groups[v["character"]].append((k, v))
        rows = []
        for char, items in groups.items():
            labels = {v["label"] for _, v in items}
            eps = sorted({n for _, v in items for n in v["episodes"]})
            rows.append({"character": char, "voice_id": re.sub(r"\W+", "_", char.lower()).strip("_"),
                         "script_labels": "; ".join(sorted(k for k, _ in items)), "lines": sum(v["lines"] for _, v in items),
                         "episodes": len(eps), "first_ep": eps[0], "last_ep": eps[-1],
                         "needs_review": any(v["needs_review"] for _, v in items), **self.voice_profile(labels)})
        rows.sort(key=lambda r: -r["lines"])
        return rows

    def metrics(self) -> dict:
        amap = self.alias_map()
        total = sum(v["lines"] for v in amap.values())
        merged_lines = sum(v["lines"] for v in amap.values() if v["merged"])
        review_lines = sum(v["lines"] for v in amap.values() if v["needs_review"])
        return {"labels_in_script": len(self.speaks), "label_runs": len(self.runs),
                "characters_after": len({v["character"] for v in amap.values()}), "dialogue_lines": total,
                "lines_fixed_by_merge": merged_lines, "pct_lines_fixed": round(100 * merged_lines / total, 1),
                "runs_needing_review": sum(v["needs_review"] for v in amap.values()), "lines_in_review": review_lines}
