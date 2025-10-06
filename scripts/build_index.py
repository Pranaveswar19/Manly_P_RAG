import argparse
import os
import sys
from typing import List, Set
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

TXT_INPUT_DIR = DATA_DIR / "ocr_txt"
INDEX_DIR = DATA_DIR / "index"

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import faiss

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

def list_from_env(env_key: str) -> List[str]:
    """
    Read extra inputs from an env var, split by os.pathsep (semicolon on Windows).
    Example (.env on Windows):
      RAG_EXTRA_INPUTS=C:\\path\\to\\data_txt;C:\\path\\to\\all_docs.txt
    """
    val = os.getenv(env_key, "").strip()
    if not val:
        return []
    parts = [p.strip().strip('"').strip("'") for p in val.split(os.pathsep)]
    return [p for p in parts if p]

def discover_text_paths() -> List[str]:
    """Auto-discover .txt files from the OCR pipeline output."""
    paths: Set[str] = set()

    if TXT_INPUT_DIR.exists():
        for p in TXT_INPUT_DIR.rglob("*.txt"):
            if p.is_file():
                paths.add(str(p))

    return sorted(paths)

def expand_inputs(input_paths: List[str]) -> List[str]:
    """
    Expand a list of provided paths; if a directory, include all .txt files recursively.
    If a file, include it if it exists. If no extension and .txt exists, include that.
    """
    found: Set[str] = set()
    for p in input_paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(".txt"):
                        found.add(os.path.join(root, f))
        elif os.path.isfile(p):
            found.add(p)
        else:
            if not p.lower().endswith(".txt") and os.path.isfile(p + ".txt"):
                found.add(p + ".txt")
            else:
                print(f"[warn] Path not found or not a file/dir: {p}", file=sys.stderr)
    return sorted(found)


def get_embed_model(provider: str, model: str):
    provider = provider.lower()
    if provider == "openai":
        if not HAS_OPENAI:
            raise RuntimeError(
                "OpenAI embedding not available. Install llama-index-embeddings-openai "
                "and ensure the import works."
            )
        return OpenAIEmbedding(model=model)
    return HuggingFaceEmbedding(model_name=model)


def main():
    parser = argparse.ArgumentParser(description="Build a FAISS vector index from text files.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Optional files/dirs. If omitted, the script auto-discovers sources (including your Windows paths and RAG_EXTRA_INPUTS).",
    )
    parser.add_argument("--persist", default=str(INDEX_DIR),
        help="Directory to persist the index (default: data/index)")
    parser.add_argument("--embedding-provider", choices=["hf", "openai"], default="hf", help="Embedding provider")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model (HF default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument("--chunk-size", type=int, default=1600, help="Chunk size for splitting text")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    args = parser.parse_args()

    if args.inputs is None or len(args.inputs) == 0:
        txt_files = discover_text_paths()
        print(f"[info] Auto-discovered {len(txt_files)} text files.")
    else:
        txt_files = expand_inputs(args.inputs)
        print(f"[info] Expanded inputs into {len(txt_files)} text files.")

    if not txt_files:
        print(
            "[error] No .txt files found. Provide --inputs or ensure files exist in the configured folders/files.\n"
            "Tip: set RAG_EXTRA_INPUTS in your .env to list absolute paths (dirs and/or files).",
            file=sys.stderr,
        )
        sys.exit(1)

    Settings.node_parser = SentenceSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    embed_provider = "openai" if args.embedding_provider == "openai" else "hf"
    embed_model = get_embed_model(embed_provider, args.embedding_model)
    Settings.embed_model = embed_model

    probe_emb = embed_model.get_text_embedding("dimension probe")
    emb_dim = len(probe_emb)
    print(f"[info] Embedding dim: {emb_dim}")

    print("[info] Loading documents...")
    documents = SimpleDirectoryReader(input_files=txt_files).load_data()
    print(f"[info] Loaded {len(documents)} documents from {len(txt_files)} files.")

    faiss_index = faiss.IndexFlatIP(emb_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("[info] Building index...")
    _ = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    os.makedirs(args.persist, exist_ok=True)
    storage_context.persist(persist_dir=args.persist)
    print(f"[ok] Index persisted to: {args.persist}")

    try:
        print("[info] Persisted files:")
        for root, _, files in os.walk(args.persist):
            for f in files:
                print(" -", os.path.join(root, f))
    except Exception:
        pass


if __name__ == "__main__":
    main()