"""Offline tests: no API key needed.  Run:  python tests/test_pipeline.py "path/to/show_folder"

Uses a fake LLM that returns deliberately bad answers to prove the validator
catches them (made-up timestamps, cuts inside a line, wrong lengths, markdown-wrapped JSON).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo_finder import llm  # noqa: E402

FAKE_REPLY = '''```json
{"promos":[
 {"segments":[{"episode":1,"start":"00:11.30","end":"00:47.00"}],"hook":"good","cliffhanger":"c","why":"w",
  "hook_score":8,"cliffhanger_score":7,"standalone_score":9,"confusion_risk":"none"},
 {"segments":[{"episode":1,"start":"09:99.00","end":"10:30.00"}],"hook":"made-up time","cliffhanger":"c","why":"w",
  "hook_score":9,"cliffhanger_score":9,"standalone_score":9,"confusion_risk":"none"},
 {"segments":[{"episode":1,"start":"00:50.00","end":"01:02.00"}],"hook":"too short","cliffhanger":"c","why":"w",
  "hook_score":6,"cliffhanger_score":6,"standalone_score":6,"confusion_risk":"none"}
]}```'''


class FakeBackend(llm.Backend):
    name, model = "fake", "fake-1"

    def _call(self, prompt):
        return FAKE_REPLY


def main(show_dir):
    llm.BACKENDS["fake"] = FakeBackend
    out = tempfile.mkdtemp()
    sys.argv = ["run.py", "--input", show_dir, "--backend", "fake", "--episodes", "1-2", "--sleep", "0", "--out", out]
    import run
    run.main()
    import json
    promos = json.loads((Path(out) / "promos.json").read_text())
    texts = [p["hook"] for p in promos]
    assert "made-up time" not in texts, "validator must reject invented timestamps"
    short = [p for p in promos if p["hook"] == "too short"]
    assert short and any(f.startswith("length_out_of_range") for f in short[0]["flags"]), "short clip must be flagged"
    for p in promos:
        assert p["segments"], "every promo needs segments"
    raw = json.loads((Path(out) / "raw" / "ep001_llm.json").read_text())
    assert raw["_rejected_by_validator"], "rejections must be logged in raw output"
    print("\nAll checks passed. Outputs in", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
