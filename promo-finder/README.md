# Promo Finder

Reads microdrama screenplays (timestamped .txt episodes) and finds the best 20 to 60 second promo clips: where each one starts and ends, the hook, the cliffhanger, why it works, and a ranked posting order. Built for Kuku AI Generalist Assignment, Problem 1.

## How it works

```
episode .txt files
      |
1. parse.py       plain code   -> timed events (dialogue, action, camera, sound) per episode
2. heuristics.py  plain code   -> scores every 20-60 s window: hook cue in first 3 s, cliffhanger cue
                                  in last 4 s, dialogue density, penalties for cutting into a line or
                                  opening on a reply. Keeps the top 4 non-overlapping windows per episode.
3. refine.py      LLM          -> one call per episode: full transcript + next episode's opening + the
                                  rule suggestions. Model returns 1-2 promos as JSON (may join up to
                                  3 segments, incl. across episodes).
   validator      plain code   -> every timestamp is checked against the script. Cuts inside a line are
                                  moved to the line edge; invented times are rejected and logged.
4. rank.py        plain code   -> final score = 65% LLM scores + 35% rule scores, minus penalties.
                                  Removes duplicates, spreads the top 10 across episodes, flags promos
                                  for human review.
```

The AI only does what rules can't: judging drama and whether a new viewer would follow the clip. Parsing, timing, validation and ranking are deterministic code, so you can check them.

## Run it

Python 3.9+. No packages to install (standard library only).

```bash
# 1. put the episode folder somewhere local, e.g. data/<show name>/   (data/ is git-ignored)
# 2. add your free Gemini key: https://aistudio.google.com/apikey
cp .env.example .env        # then edit GEMINI_API_KEY=...

python run.py --input "data/I Married a Broken Soldier. He was a Duke"
```

Useful options:

| Option | What it does |
|---|---|
| `--backend offline` | rules only, no API key, runs in about 10 s |
| `--episodes 1-5` | quick test on a few episodes |
| `--resume` | reuse saved replies in `outputs/raw/` after a rate-limit stop |
| `--model gemini-3.8-pro` | swap the model |
| `--backend openai --model <name>` | any OpenAI-compatible API (OpenAI, Groq, OpenRouter, local Ollama). Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` |
| `--sleep 7` | pause between calls to stay inside free-tier limits |

Tests (no key needed): `python tests/test_pipeline.py "data/<show name>"`

## Swapping the model

All model code is in `promo_finder/llm.py`. A backend is one class with a `_call(prompt) -> str` method. To add a provider, write the class and add it to `BACKENDS`. The prompt is in `promo_finder/prompts/refine.txt` and can be edited without touching code.

## Outputs (`outputs/`)

- `report.md`: top 10 to post first with reasons, quality-check list, table of all promos
- `promos.csv` / `promos.json`: every promo with segments, hook, cliffhanger, why, LLM scores, rule scores, flags
- `raw/epNNN_llm.json`: unedited model reply per episode, including promos the validator rejected
- `raw/epNNN_rule_candidates.json`: what the rule stage suggested
- `run_meta.json`: backend, model, number of calls, approximate input tokens, errors

## Quality flags

| Flag | Meaning |
|---|---|
| `llm_vs_rules_disagree` | model and rules differ by more than 0.4 (out of 1), so a human should look |
| `may_confuse_new_viewer` | model rated standalone clarity below 6/10 |
| `opens_mid_sentence` / `opens_on_reply` | first line continues an earlier line or answers someone off-clip |
| `fixed_cut_into_line` | model cut inside a line of dialogue; moved to the line edge |
| `length_out_of_range_Ns` | total length not within 20-60 s |
| `rule_weak_hook` / `rule_weak_ending` | no strong keyword cue in the first 3 s / last 4 s |
| `llm_failed_rule_fallback` | API failed for this episode; rule pick used instead |

## Known limits

- Works from the script, not the video. Timings are the script's planned timings, and a strong visual moment the script describes weakly will be missed.
- Keyword cues are tuned on one show (period romance). Other genres need new cue words.
- Cross-episode promos are limited to an episode plus the next episode's first 45 s.
- Free-tier rate limits make a full 50-episode run take several minutes. Use `--resume` if it stops.

## Data

The input pack is confidential. Keep it in `data/` (git-ignored) and do not commit it.
