"""The AI lawyer's brain: one model call per turn, then code checks everything.

What the model does  : understand the messy story (any language), structure the case,
                       explain options in plain words, draft documents, play the other side.
What code does       : decide which laws may be cited (kb.json only), strip any section that
                       is not verified, correct outdated law names, force a "see a lawyer"
                       banner for risky situations, and keep the case file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
KB = json.loads((HERE / "kb.json").read_text(encoding="utf-8"))["entries"]
KB_BY_ID = {e["id"]: e for e in KB}
ALLOWED_SECTIONS = {e["section"] for e in KB if e["section"]} | {s for e in KB for s in e.get("also_mentions", [])}

CATEGORIES = ["consumer", "deposit", "salary", "other"]
DOC_TYPES = {
    "notice": "a legal notice / demand letter to the other party",
    "consumer_complaint": "a draft consumer complaint for the District Consumer Commission (to file on e-Jagriti)",
    "wage_claim": "a draft application for unpaid wages under the Code on Wages, 2019, section 45",
}

# Situations where an app must not be the only help. Code decides, not the model.
ESCALATE_PATTERNS = {
    "violence or threats": r"\b(beat|hit|slap|threat|threaten|attack|violence|assault|kill)\w*|maar|dhamki|धमकी|मार",
    "police or criminal case": r"\b(police|fir|arrest|bail|criminal|jail)\b|पुलिस|गिरफ्तार",
    "court papers received": r"\b(summons|court notice|served|warrant|case filed against)\b|समन",
    "family law": r"\b(divorce|custody|dowry|alimony|maintenance case|domestic violence)\b|तलाक|दहेज",
    "forced eviction or lockout": r"\b(locked (me )?out|threw (out )?my|changed the lock|evict\w* by force)\b",
    "loan recovery / bank notice": r"\b(loan|emi|bank notice|recovery agent|sarfaesi|loan default)\b",
}
OUTDATED = [
    (r"\bIPC\b|Indian Penal Code", "Note: the IPC was replaced by the Bharatiya Nyaya Sanhita from 1 July 2024.", "BNS-2023"),
    (r"e-?Daakhil|e-?Dakhil", "Note: e-Daakhil was merged into e-Jagriti from 1 January 2025.", "E-JAGRITI"),
    (r"Payment of Wages Act|Minimum Wages Act", "Note: these Acts were replaced by the Code on Wages, 2019 from 21 November 2025.", "COW-STATUS"),
]
KNOWN_ACTS = {"consumer protection act", "consumer protection (consumer disputes redressal commissions) rules",
              "consumer protection (jurisdiction of the district commission, the state commission and the national commission) rules",
              "jurisdiction rules", "model tenancy act", "code on wages", "legal services authorities act", "bharatiya nyaya sanhita",
              "payment of wages act", "minimum wages act", "indian penal code"}
ACT_RE = re.compile(r"\b((?:[A-Z][A-Za-z()]*\s+){1,7}(?:Act|Code|Rules|Sanhita))\b(?:,?\s*(\d{4}))?")
SECTION_RE = re.compile(r"(?:\b[Ss]ection|\b[Ss]ec\.?|\bS\.|\bu/s|धारा|பிரிவு)\s*(\d+[A-Z]?)(?:\s*\(\s*\w+\s*\))*")


def kb_for_prompt(topic: str | None = None) -> str:
    rows = [e for e in KB if topic in (None, "other") or e["topic"] in (topic, "general")]
    return "\n".join(f"[{e['id']}] {e['act']}{' s.' + e['section'] if e['section'] else ''}: {e['title']}. {e['text']}"
                     for e in rows)


TURN_PROMPT = """You are "Nyay Saathi", a free legal helper for ordinary people in India. You are NOT a lawyer and you say so when it matters.
You help with three everyday problems: (1) consumer refunds / defective goods or services, (2) rental security deposit not returned, (3) salary not paid. Anything else: give general next steps only and send them to free legal aid.

HARD RULES
- Reply in the SAME language and script the user writes in (Hindi in Devanagari, Hinglish in Roman letters, Tamil, English...). Short sentences, no jargon.
- You may cite ONLY the verified facts below, by their [ID]. Never name any other Act, section, rule, case or judgment. If you are not sure a law exists, say you don't know.
- Never promise an outcome. Say what is likely and what depends on facts or the state.
- Ask for missing facts that change the advice (dates, amounts, city/state, what proof they have). Max 3 questions per turn.
- If there is violence, police, a criminal angle, court papers, family law, or a large amount (over Rs 50 lakh), set needs_lawyer=true.

VERIFIED FACTS YOU MAY CITE
{kb}

CASE FILE SO FAR (update it with anything new)
{case}

CONVERSATION
{history}

MODE: {mode}
{mode_rules}

Return ONLY JSON:
{{
  "reply": "<what you say to the user, in their language>",
  "language": "<english | hindi | hinglish | tamil | other: name>",
  "case": {{
    "category": "<consumer | deposit | salary | other>",
    "summary": "<2 lines, English>",
    "timeline": [{{"date": "<as stated or 'unknown'>", "event": "<what happened>"}}],
    "amount_claimed_rs": <number or null>,
    "other_party": "<who>",
    "location_state": "<state/city or 'unknown'>",
    "evidence_have": ["<proof they said they have>"],
    "evidence_missing": ["<proof worth collecting>"],
    "deadline_risk": "<e.g. consumer complaint within 2 years of ..., or 'none known'>"
  }},
  "options": [{{"title": "<option>", "steps": ["<step>"], "cites": ["<ID>"]}}],
  "cites": ["<every ID you relied on>"],
  "questions": ["<missing facts to ask, max 3>"],
  "needs_lawyer": <true|false>,
  "needs_lawyer_reason": "<why, or ''>"
}}"""

MODE_RULES = {
    "advise": "Build the case, explain where they stand and give 2-3 options with concrete steps.",
    "practice": ("ROLE-PLAY: play the OTHER side (landlord / company / employer) realistically and a bit stubborn, so the user "
                 "can practise negotiating. Stay in character in 'reply' (2-4 lines). After the in-character line add a line "
                 "starting with 'Coach:' with one tip on how the user could argue better, citing a verified [ID] if useful."),
}

DRAFT_PROMPT = """You are drafting {doc_desc} for an ordinary person in India, based ONLY on the case file below.
Rules:
- Write it in clear, formal English (documents to companies, landlords and authorities are usually in English) unless the user asked for {language}.
- Use ONLY these verified legal references, cited by name exactly as given; do not mention any other law or section:
{kb}
- Where a fact is missing, write a placeholder in square brackets like [date of vacating] instead of inventing it.
- Include: parties, facts in date order, the demand with amount, a deadline (15 days for a notice), what will follow if ignored, list of enclosed documents, signature block.
- Keep it to one page.

CASE FILE
{case}

Return ONLY JSON: {{"title": "<document title>", "body": "<full document text>", "placeholders": ["<each [placeholder] used>"], "cites": ["<IDs used>"]}}"""


def detect_escalation(text: str) -> list[str]:
    t = text.lower()
    return [label for label, pat in ESCALATE_PATTERNS.items() if re.search(pat, t)]


def check_citations(text: str, cites: list[str]) -> tuple[str, list[str], list[str], list[str]]:
    """Return (clean_text, valid_cites, flags, auto_cites). Unverified sections are removed from the text."""
    flags = []
    valid = [c for c in cites if c in KB_BY_ID]
    bad = [c for c in cites if c not in KB_BY_ID]
    if bad:
        flags.append(f"unknown_kb_ids_removed:{','.join(bad)}")

    def repl(m):
        if m.group(1) in ALLOWED_SECTIONS:
            return m.group(0)
        flags.append(f"unverified_section_removed:{m.group(1)}")
        return "[unverified section removed]"

    clean = SECTION_RE.sub(repl, text)

    def act_repl(m):
        name = re.sub(r"^(the|under|of|per|as)\s+", "", m.group(1).strip(), flags=re.I).lower()
        if any(name.endswith(k) or k.endswith(name) for k in KNOWN_ACTS) or name in ("act", "code", "rules"):
            return m.group(0)
        flags.append(f"unverified_law_flagged:{m.group(0)}")
        return m.group(0) + " (not in my verified list, treat as unconfirmed)"

    clean = ACT_RE.sub(act_repl, clean)
    clean = re.sub(r"\[(?:ID:\s*)?([A-Z][A-Z0-9-]+)\]", lambda m: m.group(0) if m.group(1) in KB_BY_ID
                   else (flags.append(f"fake_id_removed:{m.group(1)}") or ""), clean)
    auto = []
    for pat, note, kid in OUTDATED:
        if re.search(pat, clean, re.I) and note not in clean:
            clean += f"\n\n{note}"
            auto.append(kid)
            flags.append(f"outdated_law_corrected:{kid}")
    return clean, valid, list(dict.fromkeys(flags)), auto


def lawyer_banner(reasons: list[str], language: str) -> str:
    why = ", ".join(reasons)
    if "hindi" in language.lower() or "hinglish" in language.lower():
        return (f"⚠️ Is mamle mein ({why}) asli vakeel se baat karna zaroori hai. Muft kanooni madad: NALSA helpline 15100 "
                f"ya apne zile ka DLSA office. Agar khatra hai to turant 112 par call karein.")
    return (f"⚠️ This situation ({why}) needs a real lawyer, not just an app. Free legal aid: call NALSA 15100 or visit your "
            f"District Legal Services Authority. If you are in danger, call 112 now.")


class Session:
    """One user conversation. Holds the case file and history; every turn goes through the checks."""

    def __init__(self, backend):
        self.llm = backend
        self.history: list[tuple[str, str]] = []
        self.case: dict = {}
        self.log: list[dict] = []

    def transcribe(self, audio_bytes: bytes, mime: str = "audio/wav") -> str:
        txt = self.llm.generate("Transcribe this voice note exactly, in the language and script spoken. "
                                "Return only the transcript text.", json_mode=False, audio=(audio_bytes, mime))
        return txt.strip()

    def turn(self, user_text: str, mode: str = "advise") -> dict:
        self.history.append(("user", user_text))
        hist = "\n".join(f"{who.upper()}: {txt}" for who, txt in self.history[-12:])
        prompt = TURN_PROMPT.format(kb=kb_for_prompt(self.case.get("category")), case=json.dumps(self.case, ensure_ascii=False),
                                    history=hist, mode=mode, mode_rules=MODE_RULES[mode])
        raw = self.llm.generate_json(prompt)
        reply, cites, flags, auto = check_citations(str(raw.get("reply", "")), list(raw.get("cites", [])))
        for opt in raw.get("options", []) or []:
            opt["cites"] = [c for c in opt.get("cites", []) if c in KB_BY_ID]
            opt["steps"] = [check_citations(s, [])[0] for s in opt.get("steps", [])]

        if isinstance(raw.get("case"), dict):
            self.case = raw["case"]
        reasons = detect_escalation(user_text)
        amt = self.case.get("amount_claimed_rs")
        if isinstance(amt, (int, float)) and amt > 50_00_000:
            reasons.append("amount above Rs 50 lakh")
        if raw.get("needs_lawyer") and raw.get("needs_lawyer_reason"):
            reasons.append(str(raw["needs_lawyer_reason"])[:80])
        lang = str(raw.get("language", "english"))
        if reasons:
            reply = lawyer_banner(reasons, lang) + "\n\n" + reply
        out = {"reply": reply, "language": lang, "case": self.case, "options": raw.get("options", []),
               "cites": sorted(set(cites + auto)), "questions": raw.get("questions", []),
               "needs_lawyer": bool(reasons), "lawyer_reasons": reasons, "flags": flags, "mode": mode,
               "model": f"{self.llm.name}:{self.llm.model}", "raw_model_output": raw}
        self.history.append(("assistant", reply))
        self.log.append({"user": user_text, **out})
        return out

    def draft(self, doc_type: str) -> dict:
        topic = {"wage_claim": "salary", "consumer_complaint": "consumer"}.get(doc_type, self.case.get("category"))
        prompt = DRAFT_PROMPT.format(doc_desc=DOC_TYPES[doc_type], kb=kb_for_prompt(topic),
                                     case=json.dumps(self.case, ensure_ascii=False), language="their language")
        raw = self.llm.generate_json(prompt)
        body, cites, flags, auto = check_citations(str(raw.get("body", "")), list(raw.get("cites", [])))
        body += ("\n\n---\nDraft made by an AI helper, not a lawyer. Check every fact, fill in the blanks shown in square brackets, "
                 "and if the amount is large or the other side has a lawyer, get it reviewed (free: NALSA 15100).")
        out = {"doc_type": doc_type, "title": raw.get("title", DOC_TYPES[doc_type]), "body": body,
               "placeholders": list(dict.fromkeys(p for p in re.findall(r"\[[^\]]+\]", body)
                                                if p.strip("[]") not in KB_BY_ID and "unverified" not in p)), "cites": sorted(set(cites + auto)), "flags": flags,
               "model": f"{self.llm.name}:{self.llm.model}", "raw_model_output": raw}
        self.log.append({"draft": doc_type, **out})
        return out
