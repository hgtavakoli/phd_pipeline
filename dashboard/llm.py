"""
llm.py — tiny client for a local LLM served by Ollama (Phase 2 of the LLM roadmap).

Why a separate file?  server.py is already big; every later phase
(system prompt, RAG, tool calling) will grow *this* file instead.

Only the Python standard library is used (urllib + json), same as server.py,
so nothing new has to be installed.

Ollama API used here:
  GET  /api/tags   -> list of installed models
  POST /api/chat   -> {"model", "messages":[{"role","content"},...], "stream"} -> answer
"""
import json
import os
import urllib.error
import urllib.request

# Where Ollama listens. Override with an env var if it runs on another machine/port.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# Default model; the dashboard can pick another one per request.
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:latest")

# The system prompt = the model's "job description". It is sent as the FIRST
# message of every conversation. Phase 3 is all about improving this text.
SYSTEM_PROMPT = (
    "You are the assistant of the SmartSDG pipeline, a PhD research project. "
    "The pipeline has stages: (1) synthetic data generation (SDG) with NVIDIA Isaac Sim, "
    "(2) YOLO object-detection training, (3) inference, (4) active learning, "
    "(5) continual learning. Every stage is described by an Asset Administration Shell (AAS) passport. "
    "Answer concisely. If you do not know something about this specific project, say so instead of guessing."
)


def _request(path, payload=None, timeout=300):
    """Send one HTTP request to Ollama and return the decoded JSON reply."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=data,                                   # data != None  ->  POST
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def list_models():
    """Names of the models installed in Ollama (what `ollama list` shows)."""
    return [m["name"] for m in _request("/api/tags").get("models", [])]


def chat(messages, model=None):
    """
    messages: the conversation so far, e.g.
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, ...]
    The browser keeps this list and sends ALL of it every time: the LLM itself
    is stateless — it has no memory between calls. "Memory" = resending history.

    Returns a dict for the dashboard: {"ok", "content", "stats"} or {"ok": False, "error"}.
    """
    full = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
    try:
        r = _request("/api/chat", {"model": model or DEFAULT_MODEL, "messages": full, "stream": False})
    except urllib.error.URLError as e:
        return {"ok": False, "error": f"Cannot reach Ollama at {OLLAMA_URL} — is it running? ({e.reason})"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {"ok": True, "content": r["message"]["content"], "stats": _stats(r)}


def _stats(r):
    """Same numbers you computed by hand in PowerShell."""
    n, ns = r.get("eval_count", 0), r.get("eval_duration", 0)
    return {
        "model": r.get("model"),
        "prompt_tokens": r.get("prompt_eval_count", 0),   # prefill (includes system prompt + history!)
        "output_tokens": n,                               # decoding
        "tokens_per_s": round(n / (ns / 1e9), 1) if ns else None,
    }


def chat_stream(messages, model=None):
    """
    STREAMING version of chat(): a generator that yields small dicts as soon as
    the model produces tokens, instead of waiting for the full answer.

    With "stream": true Ollama answers in NDJSON = one JSON object per line:
        {"message":{"content":"Syn"},"done":false}
        {"message":{"content":"thetic"},"done":false}
        ...
        {"done":true,"eval_count":595,...}      <- last line carries the stats
    We read it line by line and yield:
        {"token": "..."}   for every piece of text
        {"done": True, "stats": {...}}   at the end
        {"error": "..."}   if something fails
    The caller can stop early (user pressed Stop) — closing the generator
    closes the HTTP connection, and Ollama then stops generating.
    """
    full = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
    body = json.dumps({"model": model or DEFAULT_MODEL, "messages": full, "stream": True}).encode("utf-8")
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
    print("models:", list_models())
    print(chat([{"role": "user", "content": "In one sentence: what does the SDG stage do?"}]))
    print("--- streaming ---")
    for ev in chat_stream([{"role": "user", "content": "Count from 1 to 10. or explain the stage 1"}]):
        print(ev.get("token", ev), end="", flush=True)
    print()
