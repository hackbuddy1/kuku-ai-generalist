# Nyay Saathi: AI legal helper for everyday problems in India

Kuku AI Generalist Assignment, Problem 2.

A person types or speaks their problem in their own language. Nyay Saathi turns the messy story into a case file (what happened, when, how much, what proof), explains where they stand in plain words, drafts a legal notice / consumer complaint / wage claim, and can play the other side so they can practise. It covers three problems in depth: **refund not received, rent deposit not returned, salary not paid**. Everything else gets general next steps and a referral to free legal aid.

## How it avoids inventing law

AI models confidently make up Acts, sections and cases. Here the model is never trusted on law:

1. **Closed list.** The model only sees `lawyer/kb.json`: 13 legal facts, each checked against a source with a date. It must cite them by ID.
2. **Code checks every answer** (`lawyer/core.py`):
   - any cited ID not in the list is dropped;
   - any "Section N" / "धारा N" / "u/s N" whose number is not in the list is **removed** from the text;
   - any Act/Code/Rules name not in the list is **tagged** "not in my verified list";
   - outdated names are corrected automatically (IPC → BNS 2023, e-Daakhil → e-Jagriti, Payment of Wages Act → Code on Wages).
3. **Code decides when to send people to a lawyer**, not the model: threats/violence, police, court papers, family law, forced lockout, loan recovery, or amounts over Rs 50 lakh always show a banner with NALSA 15100 / 112.
4. Drafts use `[placeholders]` for missing facts instead of guessing, and end with a "check this / get it reviewed" note.

To add a law, add an entry with its source to `kb.json`. It becomes citable immediately.

## Run it

```bash
pip install -r requirements.txt
set GEMINI_API_KEY=...            # Windows (export on Mac/Linux). Free key: https://aistudio.google.com/apikey
streamlit run app.py              # chat app with voice input
python run_tests.py               # runs tests/cases.json, saves raw outputs + checks to outputs/
```

On Windows you can double-click `RUN_APP.bat` or `RUN_TESTS.bat` instead.

Swap the model: `set LAWYER_MODEL=gemini-flash-latest`, or `set LAWYER_BACKEND=openai` with `OPENAI_API_KEY` / `OPENAI_BASE_URL` (Groq, OpenRouter, Ollama...). If a Gemini model is overloaded or has no free quota, the tool switches to another Flash model by itself.

Deploy free: push to GitHub, create an app on Streamlit Community Cloud, add `GEMINI_API_KEY` under Secrets.

## Test cases (`tests/cases.json`)

10 cases: deposit (Hinglish), refund (English), salary (Hindi, Tamil), threat + lockout, a fake law ("Tenant Protection Act 2022"), outdated IPC 420, out-of-scope bank notice, Rs 85 lakh builder case, and practice mode. `run_tests.py` saves every unedited reply and runs automatic checks (citations only from the list, no unverified section numbers left, right category, right language and script, lawyer banner when expected, document drafted).

## What it does not do

No criminal, family, property-title or tax matters. It does not know state-specific rent laws beyond the Model Tenancy Act caveat. It does not track deadlines after the chat. It is not a lawyer and says so.
