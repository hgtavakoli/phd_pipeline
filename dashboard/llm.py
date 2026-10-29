"""
llm.py — tiny client for a local LLM served by Ollama.

Phase 2: chat() / chat_stream()            (talking to the model)
Phase 3: prompts/*.md + sampling options   (prompt engineering)

Only the Python standard library is used (urllib + json), same as server.py.

Ollama API used here:
  GET  /api/tags   -> list of installed models
  POST /api/chat   -> {"model", "messages", "stream", "options"} -> answer
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")

# ---------------------------------------------------------------- prompts ---
# System prompts live as plain files in dashboard/prompts/. They are re-read on
# EVERY request, so you can edit a .md file and just ask again — no restart.
PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = "v4_fewshot"


def list_prompts():
    """Prompt variants available = the .md files in prompts/ (without extension)."""
    return sorted(p.stem for p in PROMPT_DIR.glob("*.md"))


def load_prompt(name=None):
    """
    Text of one prompt file. Accepts "v4_fewshot", "v4_fewshot.md", any letter case.
    An unknown name RAISES an error. (The first version silently fell back to the
    default prompt — so a typo looked like "my change had no effect". Fail loudly!)
    """
    name = Path(name or DEFAULT_PROMPT).name            # "../../x" -> "x" (no path tricks)
    if name.lower().endswith(".md"):
        name = name[:-3]
    for f in PROMPT_DIR.glob("*.md"):
        if f.stem.lower() == name.lower():
            return f.read_text(encoding="utf-8").strip()
    raise ValueError(f"unknown prompt '{name}'. Available: {', '.join(list_prompts())}")


def build_messages(messages, prompt=None):
    """[system prompt] + conversation. An empty prompt file = no system message at all."""
    system = load_prompt(prompt)
    head = [{"role": "system", "content": system}] if system else []
    return head + list(messages)


# --------------------------------------------------------------- sampling ---
# The model outputs a probability for every possible next token. "Sampling
# options" decide how we pick one:
#   temperature  0   = always the most likely token (deterministic, good for facts/tests)
#                0.8 = Ollama default, more varied
#                1.5 = creative / chaotic
#   num_ctx      context window in tokens. Ollama's default is small (2048-4096)!
#                If system prompt + history is longer, the OLDEST tokens are cut
#                silently — your carefully written prompt can vanish. llama3 supports 8192.
DEFAULT_OPTIONS = {"temperature": 0.3, "num_ctx": 8192}


def _options(temperature=None):
    o = dict(DEFAULT_OPTIONS)
    if temperature is not None:
        o["temperature"] = float(temperature)
    return o


# ------------------------------------------------------------------- HTTP ---
def _request(path, payload=None, timeout=300):
    """Send one HTTP request to Ollama and return the decoded JSON reply."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(OLLAMA_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def list_models():
    """Names of the models installed in Ollama (what `ollama list` shows)."""
    return [m["name"] for m in _request("/api/tags").get("models", [])]


def _stats(r):
    """Same numbers you computed by hand in PowerShell."""
    n, ns = r.get("eval_count", 0), r.get("eval_duration", 0)
    return {
        "model": r.get("model"),
        "prompt_tokens": r.get("prompt_eval_count", 0),   # prefill (system prompt + history!)
        "output_tokens": n,                               # decoding
        "tokens_per_s": round(n / (ns / 1e9), 1) if ns else None,
    }


# ------------------------------------------------------------------- chat ---
def chat(messages, model=None, prompt=None, temperature=None):
    """
    One full answer (no streaming). messages = conversation so far:
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, ...]
    The LLM is stateless — "memory" = the caller resends the whole history.
    """
    try:
        body = {"model": model or DEFAULT_MODEL, "messages": build_messages(messages, prompt),
                "stream": False, "options": _options(temperature)}
        r = _request("/api/chat", body)
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"Cannot reach Ollama at {OLLAMA_URL} — is it running? ({e.reason})"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "content": r["message"]["content"], "stats": _stats(r)}


def chat_stream(messages, model=None, prompt=None, temperature=None):
    """
    STREAMING version: a generator yielding events while the model writes.
    Ollama streams NDJSON = one JSON object per line:
        {"message":{"content":"Syn"},"done":false}
        ...
        {"done":true,"eval_count":595,...}      <- last line carries the stats
    We yield {"token": "..."}, then {"done": True, "stats": {...}}, or {"error": "..."}.
    Closing the generator early closes the connection -> Ollama stops generating.
    """
    try:
        msgs = build_messages(messages, prompt)
    except ValueError as e:
        yield {"error": str(e)}
        return
    body = json.dumps({"model": model or DEFAULT_MODEL, "messages": msgs,
                       "stream": True, "options": _options(temperature)}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:                       # iterating an HTTP response = reading line by line
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    yield {"error": chunk["error"]}
                    return
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield {"token": piece}
                if chunk.get("done"):
                    yield {"done": True, "stats": _stats(chunk)}
                    return
    except urllib.error.URLError as e:
        yield {"error": f"Cannot reach Ollama at {OLLAMA_URL} — is it running? ({e.reason})"}
    except Exception as e:
        yield {"error": str(e)}


if __name__ == "__main__":
    # Quick test without the dashboard:   python dashboard\llm.py
    print("models :", list_models())
    print("prompts:", list_prompts())
    print(chat([{"role": "user", "content": "In one sentence: what does the SDG stage do?"}]))
    print("--- streaming ---")
    for ev in chat_stream([{"role": "user", "content": "Count from 1 to 10."}]):
        print(ev.get("token", ev), end="", flush=True)
    print()
