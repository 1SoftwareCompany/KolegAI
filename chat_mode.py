#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG over Notion export:
- GPU0: your GGUF instruct model (served by llama.cpp / Text Generation WebUI)
- GPU1: embeddings + reranking (SentenceTransformers & CrossEncoder)

Run modes:
  python app.py index        # build (or rebuild with --force) the Qdrant index
  python app.py serve        # chat only; NO file loading or bulk embeddings
"""

import os
import sys
import csv
import re
from typing import List

import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# ======================
# Config
# ======================
COLLECTION = "notion_docs_unicom"
NOTION_EXPORT_DIR = "Notion-Export"

# Qdrant (HTTP default). If you use gRPC, set prefer_grpc=True below.
QDRANT_URL = "http://localhost:6333"

# LLM server (OpenAI-compatible, e.g., llama.cpp server)
LLM_API = "http://localhost:5000/v1/chat/completions"
MODEL_NAME = "qwen2.5-14b-instruct"  # adjust to whatever your server expects

# Embeddings / Reranker on GPU 1
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-large"
EMBED_DEVICE = "cuda:1"
RERANK_DEVICE = "cuda:1"


# Retrieval sizes (higher initial recall → better extractions)
TOP_K_INITIAL = 500
TOP_K_FINAL = 30

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


def load_notion_export(base_dir: str = NOTION_EXPORT_DIR):
    """Load .md and .csv from Notion export and chunk them."""
    docs = []
    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)

        if fname.endswith(".md"):
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()
            for idx, ch in enumerate(chunk_text(text)):
                docs.append({"path": fname, "chunk_id": idx, "text": ch})

        elif fname.endswith(".csv"):
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


def llm_call(messages: list, temperature: float = 0.1, json_only: bool = False) -> str:
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 2048,
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
def index_collection(force: bool = False, batch_size: int = 256):
    """Build or rebuild the Qdrant collection."""
    if collection_ready(COLLECTION) and not force:
        print("ℹ️ Collection exists, skipping indexing.")
        return

    docs = load_notion_export()
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


def ask(query: str, want_json: bool = False) -> str:
    """Dense retrieve → rerank → answer with LLM."""
    # Encode query on GPU 1
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    initial = client.query_points(
        collection_name=COLLECTION,
        query=q_emb.tolist(),
        limit=TOP_K_INITIAL
    ).points

    # Rerank on GPU 1
    pairs = [[query, h.payload["text"]] for h in initial]
    scores = reranker.predict(pairs)  # higher = better
    ranked = [h for _, h in sorted(zip(scores, initial), key=lambda x: x[0], reverse=True)]
    hits = ranked[:TOP_K_FINAL]

    context = build_context(hits)

    # Less rigid system prompt: never block JSON extraction with "Not found"
    system = {
        "role": "system",
        "content": (
            "You are a documentation extraction assistant. Use ONLY the provided context.\n"
            "When structured output is requested, you MUST return valid JSON that follows the requested schema.\n"
            "If some required fields are not present in the context, leave them as empty strings and still return JSON.\n"
            "Do NOT invent facts that are not in the context."
        ),
    }

    # If the user asks for a format, we append the schema hint to enforce precise keys.
    user_text = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer concisely."

    user = {"role": "user", "content": user_text}

    # Lower temp in JSON mode for determinism
    answer = llm_call([system, user], temperature=0.0 if want_json else 0.1, json_only=want_json)
    return answer


def serve():
    """Chat loop; does NOT touch Notion files or bulk embeddings."""
    if not collection_ready(COLLECTION):
        print("❌ No index found. Run:  python app.py index")
        sys.exit(1)

    print("\n--- CHAT MODE ---\nType 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        want_json = ("format" in q.lower()) or ("{" in q and "}" in q)
        print("Angel:", ask(q, want_json=want_json), "\n")


# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "serve"], default="serve")
    parser.add_argument("--force", action="store_true", help="Rebuild index even if it exists")
    parser.add_argument("--batch", type=int, default=256, help="Embedding batch size during indexing")
    args = parser.parse_args()

    if args.mode == "index":
        index_collection(force=args.force, batch_size=args.batch)
    elif args.mode == "serve":
        serve()