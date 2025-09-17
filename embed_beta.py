#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG over Notion export (hallucination-hardened):
- GPU0: your GGUF instruct model (served by llama.cpp / Text Generation WebUI)
- GPU1: embeddings + reranking (SentenceTransformers & CrossEncoder)

Run modes:
  python app.py index                # build (or rebuild with --force) the Qdrant index
  python app.py serve                # chat only; NO file loading or bulk embeddings
  python app.py purge-noise          # delete obviously noisy points from the collection

Key changes vs original:
- Noise gating: filters CSV/Jira/error-code docs BEFORE reranking
- Group-by-path: keeps only the best chunk per document to reduce sub-feature drift
- Smaller, cleaner context: defaults to 150 → 10 (initial → final)
- Optional CSV indexing off by default (toggle with --include-csv)
- Safer system prompt to avoid sub-features and tickets
"""

import os
import sys
import csv
import re
import qdrant_client
from pydantic import BaseModel
from typing import List, Tuple, Optional
import uvicorn
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointIdsList
from fastapi import FastAPI
from fastapi.responses import JSONResponse


# ======================
# Config
# ======================
COLLECTION = "notion_docs"
NOTION_EXPORT_DIR = "Notion-Export"

# Qdrant (HTTP default). If you use gRPC, set prefer_grpc=True below.
QDRANT_URL = "http://localhost:6333"

# LLM server (OpenAI-compatible, e.g., llama.cpp server)
LLM_API = "http://localhost:5000/v1/chat/completions"
MODEL_NAME = "qwen2.5-14b-instruct"  # adjust to whatever your server expects

# Embeddings / Reranker on GPU 1
EMBED_MODEL = "intfloat/multilingual-e5-large"
# Tip (optional upgrade): RERANK_MODEL = "BAAI/bge-reranker-v2-m3" for stronger precision
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cuda:1")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cuda:1")

# Retrieval sizes (higher initial recall → better extractions)
# Forward-looking default: small, clean final context
TOP_K_INITIAL = int(os.getenv("TOP_K_INITIAL", "150"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "10"))

# Behavior toggles
ENABLE_NOISE_FILTER = os.getenv("ENABLE_NOISE_FILTER", "1") == "1"
GROUP_BY_PATH = os.getenv("GROUP_BY_PATH", "1") == "1"
PRE_GROUP_LIMIT_FACTOR = int(os.getenv("PRE_GROUP_LIMIT_FACTOR", "3"))  # keep top 3×K before grouping

# ======================
# Clients & Models
# ======================
client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

# Instantiate models once; cheap for "serve", heavy for "index" only on encode()
embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
reranker = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)

# ======================
# Utilities
# ======================

def clean_content(raw: str) -> str:
    """Strip out <think> blocks or similar hidden reasoning."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return raw.strip()


def chunk_text(t: str, max_chars: int = 2000, overlap: int = 120) -> List[str]:
    """Roughly 450-600 tokens per chunk; good trade-off for retrieval."""
    t = t.strip()
    if len(t) <= max_chars:
        return [t]
    chunks, i = [], 0
    step = max_chars - overlap
    while i < len(t):
        chunks.append(t[i:i + max_chars])
        i += step
    return chunks


def load_notion_export(base_dir: str = NOTION_EXPORT_DIR, include_csv: bool = False):
    """Load .md and (optionally) .csv from Notion export and chunk them."""
    docs = []
    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)

        if fname.endswith(".md"):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            for idx, ch in enumerate(chunk_text(text)):
                docs.append({"path": fname, "chunk_id": idx, "text": ch})

        elif include_csv and fname.endswith(".csv"):
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row_i, row in enumerate(reader):
                    text = " | ".join([f"{k}: {v}" for k, v in row.items()])
                    for idx, ch in enumerate(chunk_text(text)):
                        docs.append({"path": f"{fname}#row{row_i}", "chunk_id": idx, "text": ch})
    return docs


def collection_ready(name: str) -> bool:
    """Return True if collection exists and has points."""
    try:
        client.get_collection(name)
        count = client.count(name, exact=True).count
        return count > 0
    except Exception:
        return False


def build_context(hits) -> str:
    """Label each chunk so the LLM can separate sources."""
    blocks = []
    for h in hits:
        p = h.payload
        blocks.append(f"### SOURCE path={p['path']} chunk={p['chunk_id']}\n{p['text']}")
    return "\n\n".join(blocks)


# --- Noise gating ---
NOISE_PATH_DENY = re.compile(r"(?i)(error\s*codes?|jira|issues?|backlog|meeting|minutes|\\.csv$)")
NOISE_TEXT_DENY = re.compile(r"(?i)(error code|unique error|jira|ticket|issue|bug|severity|priority|sprint|epic|story)")
ALLOW_HINT = re.compile(r"(?i)(microservice|service|api|architecture|system|infrastructure|overview)")


def is_noise(payload: dict) -> bool:
    """Heuristic: block CSV/Jira/error-code docs unless they clearly look like service/API pages."""
    path = payload.get("path", "")
    text = payload.get("text", "")

    if NOISE_PATH_DENY.search(path) or NOISE_TEXT_DENY.search(text):
        return not (ALLOW_HINT.search(path) or ALLOW_HINT.search(text))

    # Default: drop CSV rows unless allowed by hint
    if path.endswith(".csv"):
        return not (ALLOW_HINT.search(path) or ALLOW_HINT.search(text))

    # Extremely short or metadata-only chunks are also likely noise
    if len(text.strip()) < 40:
        return True

    return False


def llm_call(messages: list, temperature: float = 0.1, json_only: bool = False) -> str:
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 4068,
        "temperature": temperature,
        "seed": 42,
    }
    # If your llama.cpp server supports grammars, you can enforce JSON:
    # if json_only:
    #     body["grammar"] = "json"
    resp = requests.post(LLM_API, json=body, timeout=120)
    data = resp.json()
    if "choices" in data:
        return clean_content(data["choices"][0]["message"]["content"])
    return f"[API error: {data}]"


# ======================
# Indexing
# ======================

def index_collection(force: bool = False, batch_size: int = 256, include_csv: bool = False):
    """Build or rebuild the Qdrant collection."""
    if collection_ready(COLLECTION) and not force:
        print("ℹ️ Collection exists, skipping indexing.")
        return

    docs = load_notion_export(include_csv=include_csv)
    print(f"Loaded {len(docs)} chunks from Notion export")

    embeddings = embedder.encode(
        [d["text"] for d in docs],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Recreate guarantees correct vector size
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
    )

    points = []
    for i, (doc, emb) in enumerate(zip(docs, embeddings)):
        payload = {"text": doc["text"], "path": doc["path"], "chunk_id": doc["chunk_id"]}
        points.append({"id": i, "vector": emb.tolist(), "payload": payload})

    # Upsert in batches
    BATCH_SIZE = 1000
    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start:start + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION, points=batch)
        print(f"Inserted {start + len(batch)}/{len(points)} chunks")

    print("✅ Indexed Notion documents into Qdrant")


# ======================
# Query / Serve
# ======================
EXTRACTION_SCHEMA_HINT = (
    # NOTE: keep EXACT keys, including the requested 'feature_decription' spelling.
    'Return ONLY a JSON object of the form: '
    '{"organization_name":"", "organization_description":"", '
    '"main_features":[{"feature_name":"", "feature_decription":""}]}. '
    'Use only information present in the context. '
    'If any field is missing in the context, set it to an empty string (""), '
    'and if no features are found, use an empty array []. '
    'Do not invent content. Do not add extra keys, text, code fences, or comments.'
)


def ask(query: str, want_json: bool = False) -> str:
    """Dense retrieve → noise filter → rerank → (pre-group prune) → group-by-path → answer with LLM."""
    # Encode query on GPU 1
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]

    # Initial recall from Qdrant
    initial = client.query_points(
        collection_name=COLLECTION,
        query=q_emb.tolist(),
        limit=TOP_K_INITIAL
    ).points

    # 🔒 Drop noisy docs upfront (CSV, Jira, error-code lists, etc.)
    candidates = initial
    if ENABLE_NOISE_FILTER:
        candidates = [h for h in initial if not is_noise(h.payload)]
        if not candidates:  # fallback if we filtered too aggressively
            candidates = initial

    # Rerank on the filtered set
    pairs = [[query, h.payload["text"]] for h in candidates]
    scores = reranker.predict(pairs)  # higher = better
    scored: List[Tuple[float, any]] = list(zip(scores, candidates))

    # Optional: drop long tail before grouping for speed (keep top 3×K)
    if PRE_GROUP_LIMIT_FACTOR > 0:
        scored.sort(key=lambda x: x[0], reverse=True)
        keep = PRE_GROUP_LIMIT_FACTOR * TOP_K_FINAL
        scored = scored[:keep]

    # Keep only the best chunk per document (path)
    if GROUP_BY_PATH:
        best_by_path = {}
        for s, h in scored:
            p = h.payload["path"]
            if p not in best_by_path or s > best_by_path[p][0]:
                best_by_path[p] = (s, h)
        ranked_docs = sorted(best_by_path.values(), key=lambda x: x[0], reverse=True)
        hits = [h for _, h in ranked_docs[:TOP_K_FINAL]]
    else:
        # Fallback: just take top-K by score
        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [h for _, h in scored[:TOP_K_FINAL]]

    context = build_context(hits)

    # Safer system prompt to avoid sub-features / Jira / error lists
    system = {
        "role": "system",
        "content": (
            "You are a documentation extraction assistant. Use ONLY the provided context.\n"
            "Return only top-level microservices/services/APIs. Ignore Jira issues, tickets, error-code lists, "
            "endpoint enumerations, or sub-features. If a page lists sub-components, include only the parent microservice.\n"
            "When structured output is requested, you MUST return valid JSON that follows the requested schema.\n"
            "If required fields are not present, leave them empty. Do NOT invent facts."
        ),
    }

    # If the user asks for a format, we append the schema hint to enforce precise keys.
    user_text = f"Context:\n{context}\n\nQuestion: {query}\n\nList all matching items."
    if want_json:
        user_text += "\n\n" + EXTRACTION_SCHEMA_HINT

    user = {"role": "user", "content": user_text}

    # Lower temp in JSON mode for determinism
    answer = llm_call([system, user], temperature=0.0 if want_json else 0.1, json_only=want_json)
    return answer


# ======================
# Maintenance: purge noisy points already indexed
# ======================

def purge_noise(batch: int = 1000) -> int:
    """Scan the collection and delete points matching is_noise(). Returns deleted count."""
    print("Scanning for noisy points…")
    deleted = 0
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            limit=batch,
            with_payload=True,
            offset=offset,
        )
        if not points:
            break

        to_delete = []
        for p in points:
            try:
                if is_noise(p.payload):
                    to_delete.append(p.id)
            except Exception:
                # If payload is malformed, treat as noise
                to_delete.append(p.id)

        if to_delete:
            client.delete(collection_name=COLLECTION, points_selector=PointIdsList(points=to_delete))
            deleted += len(to_delete)
            print(f"Deleted {deleted} noisy points so far…")

        if next_offset is None:
            break
        offset = next_offset

    print(f"✅ Purge complete. Deleted {deleted} points.")
    return deleted


# ======================
# API
# ======================
app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.post("/ask")
def ask_api(req: QueryRequest):
    return JSONResponse(content={"answer": ask(req.question)})


@app.get("/health")
def health():
    return {"status": "ok"}


# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "serve", "purge-noise"], default="serve")
    parser.add_argument("--force", action="store_true", help="Rebuild index even if it exists")
    parser.add_argument("--batch", type=int, default=256, help="Embedding batch size during indexing")
    parser.add_argument("--include-csv", action="store_true", help="Index CSV files (off by default)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "index":
        index_collection(force=args.force, batch_size=args.batch, include_csv=args.include_csv)
    elif args.mode == "serve":
        uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
    elif args.mode == "purge-noise":
        purge_noise()
