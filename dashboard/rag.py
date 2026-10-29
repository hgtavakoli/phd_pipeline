"""
rag.py — Retrieval-Augmented Generation over the project documentation (Phase 4).

THE IDEA
  The model can't read 140 KB of docs on every question (context window, speed).
  So we split the docs into small CHUNKS, turn each chunk into an EMBEDDING
  (a vector of numbers that represents its meaning), and store them.
  For a question we embed the question too, find the chunks whose vectors point
  in the most similar direction (cosine similarity), and paste only those few
  chunks into the prompt. The LLM then answers from them and cites the source.

        docs/*.md ──chunk──▶ chunks ──embed──▶ vectors ──▶ rag_index.json      (offline, once)
        question ──embed──▶ vector ──cosine vs all──▶ top-k chunks ──▶ prompt  (every question)

USAGE (run from the repo root)
    ollama pull nomic-embed-text                      # once: the embedding model
    python dashboard\\rag.py build                     # chunk + embed all docs
    python dashboard\\rag.py search "default AL strategy"   # see what retrieval finds
    python dashboard\\rag.py compare "why SDG"          # plain top-k vs MMR side by side
    python dashboard\\rag.py chunks                    # inspect how the docs were split

Pure standard library (no numpy) — a few hundred chunks is tiny.
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import llm  # reuse OLLAMA_URL + the HTTP helper

ROOT = Path(__file__).resolve().parent.parent          # repo root
INDEX_FILE = Path(__file__).parent / "rag_index.json"   # git-ignored, rebuild any time
EMBED_MODEL = "nomic-embed-text"                        # 137M params, 768-dim vectors, 8k context

# Which files are knowledge. Explicit list = you control what the assistant "knows".
SOURCES = ["README.md", "*/README.md", "docs/*.md"]
EXCLUDE = {"docs/LEGACY_README_SmartSDG.md"}            # outdated -> would retrieve wrong facts

CHUNK_CHARS = 1200      # ~300 tokens. Too big = vague vectors; too small = no context.
OVERLAP_CHARS = 200     # repeat a bit between neighbours so no sentence is cut in half


# ============================================================ 1. CHUNKING ===
def _files():
    seen = []
    for pat in SOURCES:
        for f in sorted(ROOT.glob(pat)):
            rel = f.relative_to(ROOT).as_posix()
            if rel not in EXCLUDE and rel not in seen:
                seen.append(rel)
    return seen


def chunk_markdown(text, source):
    """
    Split one markdown file into chunks.
    Step 1: split at headings (#, ##, ###) — sections are natural meaning units.
    Step 2: a section longer than CHUNK_CHARS is cut into overlapping windows
            on paragraph boundaries.
    Every chunk keeps its heading path ("GUIDE.md > Training > Outputs") — that
    text is embedded too, so a chunk "knows" what it is about.
    """
    chunks, path, buf = [], [], []

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if len(body) < 40:                       # skip empty / heading-only bits
            return
        title = " > ".join([source] + path)
        for piece in _windows(body):
            chunks.append({"id": f"{source}#{len(chunks)}", "source": source,
                           "title": title, "text": piece})

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code                # '#' inside code blocks is NOT a heading
        m = None if in_code else re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[:level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    flush()
    return chunks


def _windows(body):
    if len(body) <= CHUNK_CHARS:
        return [body]
    paras, out, cur = [], [], ""
    for p in re.split(r"\n\s*\n", body):
        if len(p) <= CHUNK_CHARS:
            paras.append(p)
        else:                                    # one huge paragraph (big table / code block):
            lines, part = p.splitlines(), ""     # fall back to splitting on single lines
            for ln in lines:
                if part and len(part) + len(ln) > CHUNK_CHARS:
                    paras.append(part)
                    part = ""
                part += ("\n" if part else "") + ln
            paras.append(part)
    for p in paras:
        if cur and len(cur) + len(p) > CHUNK_CHARS:
            out.append(cur.strip())
            cur = cur[-OVERLAP_CHARS:]           # overlap: carry the tail into the next chunk
        cur += "\n\n" + p
    if cur.strip():
        out.append(cur.strip())
    return out


def all_chunks():
    chunks = []
    for rel in _files():
        chunks += chunk_markdown((ROOT / rel).read_text(encoding="utf-8", errors="replace"), rel)
    return chunks


# =========================================================== 2. EMBEDDING ===
# nomic-embed-text was trained with TASK PREFIXES: documents and questions get
# different prefixes. Forgetting them silently lowers retrieval quality.
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "


def embed(texts):
    """List of texts -> list of unit-length vectors (via Ollama /api/embed)."""
    r = llm._request("/api/embed", {"model": EMBED_MODEL, "input": texts})
    return [_normalize(v) for v in r["embeddings"]]


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def build(batch=32, log=print):
    """Chunk every doc, embed all chunks, save the index. Re-run after docs change."""
    chunks = all_chunks()
    t = time.time()
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        vecs = embed([DOC_PREFIX + c["title"] + "\n" + c["text"] for c in part])
        for c, v in zip(part, vecs):
            c["vec"] = [round(x, 5) for x in v]  # 5 decimals: half the file size, same results
        log(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)}")
    INDEX_FILE.write_text(json.dumps({"model": EMBED_MODEL, "built": time.strftime("%Y-%m-%d %H:%M"),
                                      "chunks": chunks}), encoding="utf-8")
    msg = f"{len(chunks)} chunks from {len(_files())} files in {time.time() - t:.1f}s -> {INDEX_FILE.name}"
    log(msg)
    return msg


# =========================================================== 3. RETRIEVAL ===
_cache = {"mtime": None, "index": None}


def _index():
    if not INDEX_FILE.is_file():
        raise RuntimeError("no RAG index yet — run: python dashboard\\rag.py build  (or the button in the Assistant tab)")
    m = INDEX_FILE.stat().st_mtime
    if _cache["mtime"] != m:                      # reload only when the file changed
        _cache["index"], _cache["mtime"] = json.loads(INDEX_FILE.read_text(encoding="utf-8")), m
    return _cache["index"]


# MMR settings (Maximal Marginal Relevance, Carbonell & Goldstein 1998)
USE_MMR = True
MMR_FETCH_K = 20     # first take the 20 most relevant candidates ...
MMR_LAMBDA = 0.1     # ... then trade relevance (1.0) against diversity (0.0)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))       # = cosine, because all vectors have length 1


def search(query, k=4, mmr=None):
    """
    Top-k chunks for a question.  Score ~0.7+ = very related, ~0.4 = weak.

    mmr=False  plain similarity: the k chunks closest to the question.
               Problem: they are often near-copies of each other (same fact in
               README, GUIDE and PIPELINE) -> 4 slots, 1 piece of information.
    mmr=True   MMR: pick chunks one at a time; each pick maximises
                   lambda * sim(question, chunk)  -  (1 - lambda) * max sim(chunk, already picked)
               = "relevant, but NOT a repeat of what I already have".
    mmr=None   use the module setting USE_MMR.
    """
    mmr = USE_MMR if mmr is None else mmr
    q = embed([QUERY_PREFIX + query])[0]
    scored = sorted(((_dot(q, c["vec"]), c) for c in _index()["chunks"]),
                    key=lambda s: s[0], reverse=True)

    if not mmr:
        picked = [(s, s, c) for s, c in scored[:k]]
    else:
        pool, picked = scored[:MMR_FETCH_K], []
        while pool and len(picked) < k:
            def mmr_score(item):
                rel, c = item
                redundancy = max((_dot(c["vec"], p[2]["vec"]) for p in picked), default=0.0)
                return MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * redundancy
            best = max(pool, key=mmr_score)
            picked.append((best[0], mmr_score(best), best[1]))
            pool.remove(best)

    return [{"score": round(rel, 3), "mmr": round(m, 3), "id": c["id"], "source": c["source"],
             "title": c["title"], "text": c["text"]} for rel, m, c in picked]


def format_context(hits):
    """Retrieved chunks -> text block for the prompt, each tagged with its source."""
    return "\n\n".join(f'<doc source="{h["source"]}" section="{h["title"]}">\n{h["text"]}\n</doc>' for h in hits)


# ================================================================== CLI ===
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "build":
        build()
    elif cmd == "search":           # what the assistant would get (k=4, setting USE_MMR)
        print("mode:", f"MMR (lambda={MMR_LAMBDA}, from top {MMR_FETCH_K})" if USE_MMR else "PLAIN top-4 (USE_MMR = False)")
        for h in search(" ".join(sys.argv[2:]) or "default active learning strategy", k=4):
            print(f"\n[{h['score']}] {h['title']}\n" + "-" * 60 + "\n" + h["text"][:400])
    elif cmd == "compare":          # ALWAYS runs both (ignores USE_MMR) — for learning the difference
        q = " ".join(sys.argv[2:]) or "what is the reason for synthetic data generation"
        for name, hits in (("PLAIN (similarity only)", search(q, k=4, mmr=False)),
                           (f"MMR (lambda={MMR_LAMBDA}, from top {MMR_FETCH_K})", search(q, k=4, mmr=USE_MMR))):
            print(f"\n=== {name}")
            for h in hits:
                print(f"  sim={h['score']:.3f}  mmr={h['mmr']:+.3f}  {h['title'][:90]}")
    elif cmd == "chunks":
        cs = all_chunks()
        sizes = sorted(len(c["text"]) for c in cs)
        print(f"{len(cs)} chunks, size min/median/max = {sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]} chars")
        for c in cs[:15]:
            print(f"  {len(c['text']):5d}  {c['title']}")
    else:
        print(__doc__)
