"""Find and rank promo clips in a microdrama screenplay pack.

Examples
  python run.py --input "path/to/show_folder"                      # Gemini (needs GEMINI_API_KEY)
  python run.py --input show/ --backend offline                    # rules only, no API
  python run.py --input show/ --backend openai --model llama-3.3-70b-versatile   # e.g. Groq via OPENAI_BASE_URL
  python run.py --input show/ --episodes 1-5                       # quick test on a few episodes
  python run.py --input show/ --resume                             # continue after a rate-limit stop
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from promo_finder.heuristics import find_candidates
from promo_finder.llm import LLMError, get_backend
from promo_finder.parse import fmt, load_show
from promo_finder.rank import score_and_rank, segments_str
from promo_finder.refine import build_prompt, candidate_to_promo, refine_episode


def load_dotenv(path=".env"):
    p = Path(path)
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Tee:
    """Print to the console and to outputs/run_log.txt at the same time."""
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, x):
        self.out.write(x)
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.out.flush()


def parse_range(s: str | None):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder with episode .txt screenplays")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--backend", default=os.environ.get("PROMO_BACKEND", "gemini"), help="gemini | openai | offline")
    ap.add_argument("--model", default=os.environ.get("PROMO_MODEL"), help="model name for the backend")
    ap.add_argument("--episodes", help="e.g. 1-10 or 3,7,12")
    ap.add_argument("--per-episode", type=int, default=2, help="promos to ask for per episode")
    ap.add_argument("--sleep", type=float, default=float(os.environ.get("PROMO_SLEEP", 7)), help="seconds between LLM calls (free-tier rate limits)")
    ap.add_argument("--resume", action="store_true", help="reuse saved raw LLM replies in OUT/raw/")
    args = ap.parse_args()

    show_dir = Path(args.input)
    show = load_show(show_dir)
    if not show:
        sys.exit(f"No episodes found in {show_dir}")
    show_name = show_dir.name
    wanted = parse_range(args.episodes)
    episodes = [e for e in show if not wanted or e.number in wanted]

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "run_log.txt")
    print(f"\n=== run started {datetime.now().isoformat(timespec='seconds')} ===")
    try:
        backend = get_backend(args.backend, args.model)
    except LLMError as e:
        sys.exit(str(e))
    label = f"{backend.name}:{backend.model}" if backend else "offline (rules only)"
    print(f"Show: {show_name} | {len(episodes)} episodes | backend: {label}")

    all_promos, errors, prompt_chars, calls, rejected_total = [], [], 0, 0, 0
    t0 = time.time()
    for ep in episodes:
        cands = find_candidates(ep)
        (out / "raw" / f"ep{ep.number:03d}_rule_candidates.json").write_text(
            json.dumps([c.__dict__ for c in cands], indent=2), encoding="utf-8")
        if backend is None:
            all_promos += [candidate_to_promo(c) for c in cands[:args.per_episode]]
            print(f"  E{ep.number}: {len(cands)} rule candidates")
            continue
        raw_path = out / "raw" / f"ep{ep.number:03d}_llm.json"
        prompt_chars += len(build_prompt(show_name, show, ep, cands, args.per_episode))
        try:
            if args.resume and raw_path.exists():
                from promo_finder.refine import split_rejected, validate
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                promos, _ = split_rejected([validate(show, ep, x) for x in raw.get("promos", []) if isinstance(x, dict)])
            else:
                promos, raw = refine_episode(backend, show_name, show, ep, cands, args.per_episode)
                raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
                calls += 1
                time.sleep(args.sleep)
            for p in promos:
                p["source"] = "llm"
            all_promos += promos
            n_rej = len(raw.get("_rejected_by_validator", []))
            rejected_total += n_rej
            print(f"  E{ep.number}: {len(promos)} promos" + (f", {n_rej} rejected" if n_rej else "")
                  + (f" (skipped: {raw.get('skipped_reason')})" if not promos else ""))
        except LLMError as e:
            if any(f"HTTP {c}" in str(e) for c in (400, 401, 403, 404)) and not all_promos:
                sys.exit(f"\nThe API rejected the very first call, so stopping instead of failing every episode.\n"
                         f"Check the key/model. Details: {str(e)[:400]}")
            # Never lose an episode: fall back to the rule-based picks and say so
            errors.append({"episode": ep.number, "error": str(e)[:300]})
            fb = [candidate_to_promo(c) for c in cands[:1]]
            for p in fb:
                p["flags"].append("llm_failed_rule_fallback")
                p["source"] = "rules_fallback"
            all_promos += fb
            print(f"  E{ep.number}: LLM failed, used rule fallback ({str(e)[:80]})")

    ranked = score_and_rank(show, all_promos)
    write_outputs(out, show_name, ranked, episodes, errors, label)
    label = f"{backend.name}:{backend.model}" if backend else label
    meta = {"show": show_name, "backend": label, "episodes": len(episodes), "promos": len(ranked),
            "llm_calls": calls, "llm_promos_rejected_by_validator": rejected_total, "approx_input_tokens": prompt_chars // 4, "errors": errors,
            "seconds": round(time.time() - t0, 1), "run_at": datetime.now().isoformat(timespec="seconds")}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone: {len(ranked)} promos -> {out}/report.md, promos.csv, promos.json")
    if errors:
        print(f"{len(errors)} episode(s) fell back to rules; see run_meta.json")


def write_outputs(out: Path, show_name: str, ranked: list[dict], episodes, errors, label: str):
    (out / "promos.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")
    cols = ["rank", "final_score", "segments", "duration", "hook", "cliffhanger", "why", "hook_score",
            "cliffhanger_score", "standalone_score", "rule_hook", "rule_cliff", "confusion_risk",
            "needs_human_review", "qc", "flags", "post_first_reason"]
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in ranked:
            row = {**p, "segments": segments_str(p), "qc": ";".join(p["qc"]),
                   "flags": ";".join(p["flags"] + p["rule_flags"])}
            w.writerow([row.get(c, "") for c in cols])

    covered = {s["episode"] for p in ranked for s in p["segments"]}
    md = [f"# Promo report: {show_name}", "", f"Backend: {label}. Episodes: {len(episodes)}. Promos: {len(ranked)}.", ""]
    md += ["## Post these first", ""]
    for p in ranked[:10]:
        md += [f"**#{p['rank']}  {segments_str(p)}  ({p['duration']:.0f}s, score {p['final_score']})**",
               f"- Hook: {p['hook']}", f"- Cliffhanger: {p['cliffhanger']}", f"- Why: {p['why']}",
               f"- Why post it early: {p['post_first_reason']}", ""]
    md += ["## Quality check", ""]
    review = [p for p in ranked if p["needs_human_review"]]
    md += [f"{len(review)} of {len(ranked)} promos need a human look:", ""]
    for p in review:
        md += [f"- #{p['rank']} {segments_str(p)}: {', '.join(p['qc'])}"
               + (f" (flags: {', '.join(p['flags'] + p['rule_flags'])})" if p['flags'] or p['rule_flags'] else "")]
    missing = [e.number for e in episodes if e.number not in covered]
    md += ["", f"Episodes with no promo: {missing or 'none'}"]
    if errors:
        md += [f"Episodes where the LLM failed (rule fallback used): {[e['episode'] for e in errors]}"]
    md += ["", "## All promos", "", "| # | Segments | Len | Score | Hook | Review |", "|---|---|---|---|---|---|"]
    for p in ranked:
        hook = str(p["hook"]).replace("|", "/")[:90]
        md += [f"| {p['rank']} | {segments_str(p)} | {p['duration']:.0f}s | {p['final_score']} | {hook} | {'yes' if p['needs_human_review'] else ''} |"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
