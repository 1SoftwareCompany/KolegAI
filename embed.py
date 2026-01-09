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
import qdrant_client
from pydantic import BaseModel
from typing import List
import uvicorn
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, SparseVectorParams, models, PointStruct
from fastapi import FastAPI, Depends, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from security import JwtConfig, JwtVerifier
from log import get_logger

# ======================
# Config
# ======================
COLLECTION = "notion_docs_unicom_test_hybrid_search_i_hope_it_works"
NOTION_EXPORT_DIR = "Notion-Export-Unicom-Only" # is this the one?

# Qdrant (HTTP default). If you use gRPC, set prefer_grpc=True below.
QDRANT_URL = "http://localhost:6333"

# LLM server (OpenAI-compatible, e.g., llama.cpp server)
LLM_API = "http://localhost:5000/v1/chat/completions"
MODEL_NAME = "qwen2.5-14b-instruct"  # adjust to whatever your server expects

# Embeddings / Reranker on GPU 1
EMBED_MODEL = "intfloat/multilingual-e5-large"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_DEVICE = "cuda:1"
RERANK_DEVICE = "cuda:1"

SPARSE_MODEL = "Qdrant/bm25"
SPARSE_DEVICE = os.getenv("SPARSE_DEVICE", "cuda:1")

# Retrieval sizes (higher initial recall → better extractions)
TOP_K_INITIAL = 250
TOP_K_FINAL = 20

# ======================
# Clients & Models
# ======================
client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)

# Instantiate models once; cheap for "serve", heavy for "index" only on encode()
embedder = SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE)
reranker = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)

bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25", device=SPARSE_DEVICE)

logger = get_logger(__name__)
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
    except Exception as ex:
        logger.error("Something went wrong while getting collection from qdrant")
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
def index_collection(collection_name: str, docs: List, force: bool = False, batch_size: int = 256):
    """Build or rebuild the Qdrant collection."""
    if collection_ready(collection_name) and not force:
        print("ℹ️ Collection exists, skipping indexing.")
        return

    print(f"Loaded {len(docs)} chunks from Notion export")

    dense_embeddings = embedder.encode(
        [d["text"] for d in docs],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    bm25_embeddings = list(bm25_embedding_model.embed(doc["text"] for doc in docs))

     # Recreate guarantees correct vector size
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config={
        "dense": VectorParams(size=dense_embeddings.shape[1], distance=Distance.COSINE),
        },
        sparse_vectors_config={
        "sparse":SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )

    points = []
    for i, (doc, dense_emb, bm25_emb) in enumerate(zip(docs, dense_embeddings, bm25_embeddings)):
        payload = {"text": doc["text"], "path": doc["path"], "chunk_id": doc["chunk_id"]}
        point = PointStruct(
            id=i,
            payload=payload,
            vector={
                "sparse": bm25_emb.as_object(),
                "dense": dense_emb
            }
        )
        points.append(point)

    # Upsert in batches
    BATCH_SIZE = 1000
    for start in range(0, len(points), BATCH_SIZE):
        batch = points[start:start + BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"Inserted {start + len(batch)}/{len(points)} chunks")

    print("✅ Indexed Notion documents into Qdrant")


def enrich_features(features: List, organization: str):
    feature_list = []
    feature_prompt = "You are a helpful assistant whos job is to find the most detailed information about concrete features and nothing else. The information you need to retrieve is: very detailed information about the feature AND a collection of tags which you think are keywords related to the feature. You need to search throughly in the documentation and return all of the possible information about the feature: {feature} with details: {details}. Do not invent information or return anything else other than the information about this concrete feature. If you don't find anything relevant in the docs, just return empty string. The format in which I want the information returned is plain text."
    for i, (feature) in enumerate(features):
        concrete_prompt = feature_prompt.format(feature = feature.name, details = feature.description)
        gelio_says = ask(concrete_prompt, organization)
        feature_list.append({"text": gelio_says})
    
    return feature_list
    

# ======================
# Query / Serve
# ======================
EXTRACTION_SCHEMA_HINT = (
    # NOTE: keep EXACT keys, including your requested 'feature_decription' spelling.
    'Return ONLY a JSON object of the form: '
    '{"organization_name":"", "organization_description":"", '
    '"main_features":[{"feature_name":"", "feature_decription":""}]}. '
    'Use only information present in the context. '
    'If any field is missing in the context, set it to an empty string (""), '
    'and if no features are found, use an empty array []. '
    'Do not invent content. Do not add extra keys, text, code fences, or comments.'
)


def ask(query: str, collection_name: str, want_json: bool = False) -> str:
    """Dense retrieve → rerank → answer with LLM."""
    # Encode query on GPU 1
    dense_vectors = embedder.encode([query], normalize_embeddings=True)[0]
    sparse_vectors = next(bm25_embedding_model.query_embed(query))

   
    initial = client.query_points(
    collection_name=COLLECTION,
    prefetch=[
        models.Prefetch(
            using="sparse",
            query=models.SparseVector(**sparse_vectors.as_object()),
            limit=TOP_K_INITIAL,
        ),
        models.Prefetch(
            using="dense",
            query=dense_vectors,
            limit=TOP_K_INITIAL,
        ),
    ],
    # Tell Qdrant to fuse the two prefetch result sets (RRF or DBSF)
    query=models.FusionQuery(fusion=models.Fusion.RRF),
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
    user_text = f"Context:\n{context}\n\nQuestion: {query}\n\nList all matching items."
    if want_json:
        user_text += "\n\n" + EXTRACTION_SCHEMA_HINT

    user = {"role": "user", "content": user_text}

    # Lower temp in JSON mode for determinism
    answer = llm_call([system, user], temperature=0.0 if want_json else 0.1, json_only=want_json)
    return answer


# ======================
# API
# ======================

app = FastAPI(title="RAG API")

cfg = JwtConfig(
    issuer="https://keycloak.1software.org/realms/1manager",
    audience="1manager_kolegai",
    jwks_uri="https://keycloak.1software.org/realms/1manager/protocol/openid-connect/certs"
)
verify_jwt = JwtVerifier(cfg)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class Feature(BaseModel):
    name: str
    description: str


class IndexRequest(BaseModel):
    organization: str
    features: List[Feature]


class AnswerResponse(BaseModel):
    answer: str


class FeatureResponse(BaseModel):
    feature: str


router = APIRouter(dependencies=[Depends(verify_jwt)])

@router.post("/ask", response_model=AnswerResponse)
def ask_api(req: QueryRequest):
    return {"answer": ask(req.question, COLLECTION)}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/index/features", response_model=AnswerResponse)
def index(req: IndexRequest):
    enreached_features = enrich_features(req.features, req.organization)
    index_collection(req.organization.lower()+ "_features", enreached_features)
    return {"status": "ok"}


@router.get("/ask/{organization}/feature", response_model=FeatureResponse)
def ask_api(organization: str, req: QueryRequest):
    return FeatureResponse(
        feature=ask(req.question, organization.lower() + "_features")
    )


app.include_router(router)

# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "serve"], default="serve")
    parser.add_argument("--force", action="store_true", help="Rebuild index even if it exists")
    parser.add_argument("--batch", type=int, default=256, help="Embedding batch size during indexing")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "index":
        docs = load_notion_export()
        index_collection(COLLECTION, docs, force=args.force, batch_size=args.batch)
    elif args.mode == "serve":
         uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
