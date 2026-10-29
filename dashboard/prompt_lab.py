"""
prompt_lab.py — run the SAME questions against EVERY prompt variant and write
a side-by-side report. This is how prompt engineering is done properly:
change one thing, re-run the same test set, compare. (Phase 8 grows this into
real evaluation.)

    python dashboard\prompt_lab.py                       # all prompts, llama3
    python dashboard\prompt_lab.py --model qwen2.5:7b    # compare models
    python dashboard\prompt_lab.py --prompts v1_role v4_fewshot

Output: dashboard/lab_reports/<date_time>_<model>.md  (open it in VS Code preview)
Each run gets its own file, so you keep a history of experiments.
"""
import argparse
import time
from pathlib import Path

import llm

# A good test set mixes: questions the facts answer, a trap the facts do NOT
# answer (does it admit "I don't know"?), and a format/behaviour check.
QUESTIONS = [
    "What does the SDG stage do?",                                   # basic fact
    "Which active learning strategy is the default?",                # specific fact
    "What resolution should I use for synthetic images?",            # number from the facts
    "Where is the trained model saved after training?",              # path (hallucination risk!)
    "How many GPUs did we use for the experiments in the thesis?",   # TRAP: not in facts
    "What is an AAS passport?",                                      # domain concept
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=llm.DEFAULT_MODEL)
    ap.add_argument("--prompts", nargs="*", default=llm.list_prompts())
    ap.add_argument("--temperature", type=float, default=0.0)  # 0 = repeatable comparison
    a = ap.parse_args()
    for p in a.prompts:                 # check names BEFORE spending minutes of GPU time
        llm.load_prompt(p)

    out = [f"# Prompt lab — model `{a.model}`, temperature {a.temperature}\n"]
    for q in QUESTIONS:
        out.append(f"\n## ❓ {q}\n")
        for p in a.prompts:
            t = time.time()
            r = llm.chat([{"role": "user", "content": q}], model=a.model, prompt=p, temperature=a.temperature)
            ans = r["content"] if r["ok"] else "⚠ " + r["error"]
            st = r.get("stats", {})
            print(f"[{p:>12}] {q[:45]:<45} {time.time() - t:5.1f}s  prompt={st.get('prompt_tokens')} tok")
            out.append(f"\n### `{p}`  · prompt {st.get('prompt_tokens')} tok · output {st.get('output_tokens')} tok\n\n{ans}\n")

    d = Path(__file__).parent / "lab_reports"          # NOT inside prompts/ — or the report becomes a "prompt"
    d.mkdir(exist_ok=True)
    f = d / f"{time.strftime('%Y%m%d_%H%M')}_{a.model.replace(':', '-')}.md"
    f.write_text("".join(out), encoding="utf-8")
    print("\nreport ->", f)


if __name__ == "__main__":
    main()
