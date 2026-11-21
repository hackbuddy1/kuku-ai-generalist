# Cast Sheet: one voice per character for AI dubbing

Kuku AI Generalist Assignment, Problem 3.

**Who has the problem:** the localization / dubbing producer who takes a finished English microdrama and dubs it into other languages with AI voices. **The step:** voice casting, deciding which voice speaks each line. **The output:** a cast sheet and an alias map (`script label + episodes -> one voice`) that a dubbing pipeline can use directly.

**Why it is a real problem:** AI-written screenplays name the same person in different ways. In the input pack (50 episodes, 1,487 dialogue lines) there are 64 speaker labels: Juliana is also "LADY JULIANA" (110 lines), Gerald is "THE FORMER GROOM", Clarissa is "LADY IN GOLD GOWN", Alexander is "MYSTERIOUS BEARDED MAN" / "THE BEARDED STRANGER" / "SCARRED BEARDED MAN". The reverse also happens: "BLOND NOBLEMAN" is Rupert from episode 27 on, but in episode 2 he speaks as a member of Juliana's family. A pipeline that gives one voice per label puts the wrong voice on these lines.

## How it works

| Step | Who | What |
|---|---|---|
| Split labels into runs | code | a gap of more than 8 episodes starts a new run, so one label can map to different people in different arcs |
| Same scene = different people | code | hard constraint: labels that speak in the same scene are never merged (aliases included) |
| Same name with title | code | LADY JULIANA -> JULIANA (but Lady X and Lord X are kept apart) |
| Script names it | code | action lines like "Blond Nobleman Rupert" or "Rupert, the Blond Nobleman" |
| Descriptive labels | LLM | gets loglines, the label's lines, action lines from its scenes, and allowed candidates; picks one or DISTINCT and must quote evidence |
| Check the LLM | code | candidate must be allowed; the quote must exist in the evidence (else confidence capped at 0.4); below 0.7 goes to human review, not applied |

Outputs (`outputs/`): `cast_sheet.csv` (voice, all script labels, lines, episodes, gender guess, top delivery tones), `alias_map.json`, `review.md` (merges with evidence + human review queue), `eval.md` (accuracy against `castsheet/gold.json`, a hand-made answer key), `decisions.json` (raw model replies), `run_log.txt`.

## Run

```bash
set GEMINI_API_KEY=...      # free key: https://aistudio.google.com/apikey
python run.py --input "data/<show folder>"
python run.py --input "data/<show folder>" --backend offline   # rules only
```
Standard library only. Windows: double-click `RUN_CAST.bat`. Swap the model with `--model` or `--backend openai` (OPENAI_BASE_URL / OPENAI_API_KEY).

## Known limits

Works from the script only (no audio/face matching). The answer key covers the 14 hardest labels, not all 64. Gender guess is a pronoun count near the name and is "unclear" for descriptive-only characters. Generic walk-on roles (GUARD, MAID) are kept as their own voices on purpose.
