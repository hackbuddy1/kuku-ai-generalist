"""Run every test case through the real assistant and save unedited outputs + automatic checks.

  python run_tests.py                 # all cases, Gemini (GEMINI_API_KEY)
  python run_tests.py --only 01,05    # some cases
  python run_tests.py --resume        # skip cases already saved in outputs/
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from lawyer.core import KB_BY_ID, SECTION_RE, ALLOWED_SECTIONS, Session
from lawyer.llm import LLMError, get_backend

ROOT = Path(__file__).parent
SCRIPT = {"hindi": r"[ऀ-ॿ]", "tamil": r"[஀-௿]"}


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "a", encoding="utf-8"), sys.stdout

    def write(self, x):
        self.out.write(x)
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.out.flush()


def check(case, turns, draft):
    """Automatic checks. They test behaviour we can verify mechanically; the human review is in the doc."""
    exp, res = case.get("expect", {}), []
    last = turns[-1]
    all_replies = "\n".join(t["reply"] for t in turns) + ("\n" + draft["body"] if draft else "")
    all_cites = {c for t in turns for c in t["cites"]} | set(draft["cites"] if draft else [])

    res.append(("citations only from verified list", all(c in KB_BY_ID for c in all_cites)))
    left = [m.group(1) for m in SECTION_RE.finditer(all_replies) if m.group(1) not in ALLOWED_SECTIONS]
    res.append(("no unverified section numbers left in output", not left))
    if "category" in exp:
        res.append((f"category = {exp['category']}", last["case"].get("category") == exp["category"]))
    if "needs_lawyer" in exp:
        res.append((f"needs_lawyer = {exp['needs_lawyer']}", any(t["needs_lawyer"] for t in turns) == exp["needs_lawyer"]))
    if "must_cite_any" in exp:
        res.append((f"cites one of {exp['must_cite_any']}", bool(all_cites & set(exp["must_cite_any"]))))
    if "language_contains" in exp:
        lang = exp["language_contains"]
        ok = lang in last["language"].lower()
        if lang in SCRIPT:  # also check the reply is actually written in that script
            ok = ok and bool(re.search(SCRIPT[lang], last["reply"]))
        res.append((f"replies in {lang}", ok))
    if "reply_contains" in exp:
        res.append((f"reply contains '{exp['reply_contains']}'", exp["reply_contains"].lower() in all_replies.lower()))
    if case.get("draft"):
        res.append(("document drafted", bool(draft and len(draft["body"]) > 300)))
    return [{"check": n, "pass": bool(p)} for n, p in res]


def to_md(case, turns, draft, checks):
    md = [f"# {case['id']}", f"_{case['why']}_", ""]
    for t in turns:
        md += [f"**User:** {t['user']}", "", f"**Assistant:** {t['reply']}", ""]
        for o in t.get("options", []):
            md += [f"- _{o.get('title','')}_: " + " / ".join(o.get("steps", []))]
        if t["questions"]:
            md += ["", "Questions: " + " | ".join(t["questions"])]
        md += ["", f"<sub>cites: {t['cites']} · needs_lawyer: {t['needs_lawyer']} {t['lawyer_reasons']} · flags: {t['flags']} · model: {t['model']}</sub>", ""]
    if draft:
        md += [f"## Draft: {draft['title']}", "", "```", draft["body"], "```", f"<sub>flags: {draft['flags']}</sub>", ""]
    md += ["## Automatic checks", ""] + [f"- {'PASS' if c['pass'] else 'FAIL'}: {c['check']}" for c in checks]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=4)
    args = ap.parse_args()
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    sys.stdout = Tee(out / "run_log.txt")
    print(f"\n=== test run {datetime.now().isoformat(timespec='seconds')} ===")

    cases = json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
    if args.only:
        keep = args.only.split(",")
        cases = [c for c in cases if any(c["id"].startswith(k) for k in keep)]
    backend = get_backend()
    summary = []
    for case in cases:
        path = out / f"{case['id']}.json"
        if args.resume and path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            summary.append(rec["summary"])
            print(f"{case['id']}: reused")
            continue
        sess, turns, draft = Session(backend), [], None
        try:
            for msg in case["turns"]:
                turns.append({"user": msg, **sess.turn(msg, case.get("mode", "advise"))})
                time.sleep(args.sleep)
            if case.get("draft"):
                draft = sess.draft(case["draft"])
        except LLMError as e:
            print(f"{case['id']}: MODEL ERROR {e}")
            summary.append({"id": case["id"], "error": str(e)[:200], "passed": 0, "total": 0})
            continue
        checks = check(case, turns, draft)
        s = {"id": case["id"], "passed": sum(c["pass"] for c in checks), "total": len(checks),
             "failed": [c["check"] for c in checks if not c["pass"]], "model": backend.model}
        path.write_text(json.dumps({"case": case, "turns": turns, "draft": draft, "checks": checks, "summary": s},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
        (out / f"{case['id']}.md").write_text(to_md(case, turns, draft, checks), encoding="utf-8")
        summary.append(s)
        print(f"{case['id']}: {s['passed']}/{s['total']} checks passed" + (f"  FAILED: {s['failed']}" if s["failed"] else ""))

    tot_p = sum(s.get("passed", 0) for s in summary)
    tot = sum(s.get("total", 0) for s in summary)
    (out / "summary.json").write_text(json.dumps({"passed": tot_p, "total": tot, "cases": summary}, indent=2,
                                                 ensure_ascii=False), encoding="utf-8")
    print(f"\nDone: {tot_p}/{tot} automatic checks passed across {len(summary)} cases. See outputs/")


if __name__ == "__main__":
    main()
