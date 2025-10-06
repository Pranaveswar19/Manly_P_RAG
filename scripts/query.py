import argparse
import os
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_INDEX_DIR = DATA_DIR / "index"


def configure_embeddings():
    """Force the embedder to HuggingFace MiniLM (same as index build)."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print(f"[info] Using HuggingFace embeddings: sentence-transformers/all-MiniLM-L6-v2")


def main():
    parser = argparse.ArgumentParser(description="Query the FAISS index with reranking.")
    parser.add_argument("question", help="Your question (in quotes if multi-word)")
    parser.add_argument("--persist", default=str(DEFAULT_INDEX_DIR), help="Index directory")
    parser.add_argument("--k", type=int, default=CONFIG.default_k, help="Top-k retrieved candidates (default from .env)")
    parser.add_argument("--top-n", type=int, default=CONFIG.default_top_n, help="Top-n after rerank (default from .env)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or environment.")

    Settings.llm = OpenAI(model=CONFIG.default_llm_model)

    persist_dir = args.persist
    vector_store = FaissVectorStore.from_persist_dir(persist_dir=persist_dir)

    configure_embeddings()

    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

    retriever = index.as_retriever(similarity_top_k=args.k)

    reranker = SentenceTransformerRerank(
        top_n=args.top_n,
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    qe = RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
        response_mode="compact",
    )
    response = qe.query(args.question)

    print("\n=== ANSWER ===")
    print(str(response).strip())

    print("\n=== SOURCES ===")
    for i, sn in enumerate(response.source_nodes, start=1):
        meta = sn.node.metadata or {}
        source = meta.get("file_path") or meta.get("file_name") or "unknown"
        score = getattr(sn, "score", None)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        print(f"[{i}] {source} (score={score_str})")


if __name__ == "__main__":
    sys.exit(main())
