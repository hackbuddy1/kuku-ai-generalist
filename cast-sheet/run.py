"""Build a dubbing cast sheet from a microdrama screenplay pack.

  python run.py --input "data/<show>"                    # Gemini for descriptive labels (GEMINI_API_KEY)
  python run.py --input "data/<show>" --backend offline  # rules only, no API
Outputs go to outputs/: cast_sheet.csv, alias_map.json, review.md, eval.md, decisions.json, run_log.txt
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

from castsheet.llm import LLMError, get_backend
from castsheet.parse import load_show
from castsheet.resolve import Resolver

ROOT = Path(__file__).parent


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "a", encoding="utf-8"), sys.stdout

    def write(self, x):
        self.out.write(x)
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.out.flush()


def evaluate(res: Resolver, gold_path: Path) -> tuple[str, dict]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))["items"]
    amap = res.alias_map()
    rows, correct, scored, review_ok, unsure = [], 0, 0, 0, 0
    gold_keys = set()
    for g in gold:
        runs = [v | {"key": k} for k, v in amap.items() if v["label"] == g["label"]
                and any(g["eps"][0] <= n <= g["eps"][1] for n in v["episodes"])]
        if not runs:
            rows.append(f"| {g['label']} E{g['eps'][0]}-{g['eps'][1]} | {g['expect']} | (label not found) | - |")
            continue
        for r in runs:
            gold_keys.add(r["key"])
        r = max(runs, key=lambda x: x["lines"])
        got = r["character"] if r["merged"] else ("REVIEW" if r["needs_review"] else "kept separate")
        if g["certainty"] == "certain":
            scored += 1
            ok = r["merged"] and r["character"] == g["expect"]
            correct += ok
            verdict = "correct" if ok else ("missed (sent to review)" if r["needs_review"] else "WRONG" if r["merged"] else "missed")
        else:
            unsure += 1
            ok = not r["merged"]
            review_ok += ok
            verdict = "ok, not auto-merged" if ok else f"auto-merged to {r['character']}: a human should check"
        rows.append(f"| {g['label']} E{g['eps'][0]}-{g['eps'][1]} | {g['expect']} ({g['certainty']}) | {got} | {verdict} |")
    extra = [v | {"key": k} for k, v in amap.items() if v["merged"] and k not in gold_keys]
    md = ["## Accuracy against the hand-made answer key", "",
          f"- Certain aliases correctly merged: **{correct}/{scored}**",
          f"- Unsure labels left for a human (not auto-merged): **{review_ok}/{unsure}**",
          f"- Merges outside the answer key (check by hand): {len(extra)}", "",
          "| Script label | Expected | Tool | Verdict |", "|---|---|---|---|"] + rows
    if extra:
        md += ["", "Merges not in the answer key:"] + [f"- {e['key']} -> {e['character']} ({e['method']})" for e in extra]
    return "\n".join(md), {"certain_correct": correct, "certain_total": scored, "unsure_left_for_review": review_ok,
                           "unsure_total": unsure, "extra_merges": len(extra)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--backend", default=None, help="gemini (default) | openai | offline")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    sys.stdout = Tee(out / "run_log.txt")
    print(f"\n=== run {datetime.now().isoformat(timespec='seconds')} ===")

    show = load_show(args.input)
    res = Resolver(show)
    res.rule_title()
    res.rule_apposition()
    print(f"{len(show)} episodes, {len(res.speaks)} speaker labels, {len(res.runs)} label runs; "
          f"rules decided {len(res.decisions)}; {len(res.unresolved())} descriptive runs left")
    label = "offline (rules only)"
    if args.backend != "offline":
        try:
            llm = get_backend(args.backend, args.model)
            res.llm_resolve(llm)
            label = f"{llm.name}:{llm.model}"
        except LLMError as e:
            print(f"LLM unavailable ({e}); rules-only result saved")

    sheet = res.cast_sheet()
    with open(out / "cast_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0].keys()))
        w.writeheader()
        for r in sheet:
            w.writerow({**r, "top_tones": ", ".join(r["top_tones"]), "he_she_counts": f"{r['he_she_counts'][0]}/{r['he_she_counts'][1]}"})
    (out / "alias_map.json").write_text(json.dumps(res.alias_map(), indent=2), encoding="utf-8")
    (out / "decisions.json").write_text(json.dumps(res.decisions, indent=2, ensure_ascii=False, default=list), encoding="utf-8")

    m = res.metrics()
    ev_md, ev = evaluate(res, ROOT / "castsheet" / "gold.json")
    review = sorted([d for d in res.decisions.values() if d["review"]], key=lambda d: -d["lines"])
    applied = sorted([d for d in res.decisions.values() if d["canonical"] and not d["review"]], key=lambda d: -d["lines"])
    md = [f"# Cast sheet review: {Path(args.input).name}", "", f"Model: {label}", "",
          f"- {m['labels_in_script']} speaker labels in the script became **{m['characters_after']} voices** "
          f"({m['label_runs']} label runs checked).",
          f"- **{m['lines_fixed_by_merge']} of {m['dialogue_lines']} dialogue lines ({m['pct_lines_fixed']}%)** were under an alias "
          f"and would have been dubbed in the wrong voice without this step.",
          f"- {m['runs_needing_review']} label runs ({m['lines_in_review']} lines) need a human decision.", "",
          "## Merges applied", "", "| Script label | Voice | How | Confidence | Evidence |", "|---|---|---|---|---|"]
    md += [f"| {d['run']} ({d['lines']} lines) | {d['canonical']} | {d['method']} | {d['confidence']} | {str(d['evidence'])[:150].replace('|','/')} |" for d in applied]
    md += ["", "## Needs a human decision", "", "| Script label | Suggestion | Why it was held back |", "|---|---|---|"]
    md += [f"| {d['run']} ({d['lines']} lines) | {d.get('canonical') or 'distinct?'} | conf {d['confidence']}; {', '.join(d.get('flags', [])) or d['method']}; {str(d.get('reason') or d['evidence'])[:120].replace('|','/')} |" for d in review]
    md += ["", "## Cast sheet (top 15 voices)", "", "| Voice | Lines | Episodes | Gender guess | Tones | Script labels |", "|---|---|---|---|---|---|"]
    md += [f"| {r['character']} | {r['lines']} | {r['first_ep']}-{r['last_ep']} | {r['gender_guess']} | {', '.join(r['top_tones'][:3])} | {r['script_labels'][:90]} |" for r in sheet[:15]]
    md += ["", ev_md]
    (out / "review.md").write_text("\n".join(md), encoding="utf-8")
    (out / "eval.md").write_text(ev_md, encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps({"model": label, **m, **ev}, indent=2), encoding="utf-8")
    print(f"\n{m['labels_in_script']} labels -> {m['characters_after']} voices; {m['pct_lines_fixed']}% of lines fixed; "
          f"answer key: {ev['certain_correct']}/{ev['certain_total']} certain aliases correct, "
          f"{ev['unsure_left_for_review']}/{ev['unsure_total']} unsure left for review. See {out}/review.md")


if __name__ == "__main__":
    main()
