#!/usr/bin/env python3
"""
Run the offline RAG pipeline end-to-end:

1) Scrape PDFs into data/pdfs
2) OCR + text extraction into data/ocr_pdfs and data/ocr_txt
3) Build FAISS index into data/index
4) (Optional) Query the index from CLI
5) (Optional) Launch local Streamlit app

Place this script in: scripts/run_offline_pipeline.py  (next to build_index.py, ocr.py, scraper.py, query.py)
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# Load .env if present (you said you won't upload it; this will pick it up locally)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def run_step(name: str, cmd: list[str], cwd: Path | None = None, env: dict | None = None):
    """Run one step, print timing, and fail fast with a helpful message."""
    print(f"\n=== {name} ===")
    print("$ " + " ".join(str(c) for c in cmd))
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if proc.returncode != 0:
        print(f"[{name}] ❌ FAILED with exit code {proc.returncode}")
        sys.exit(proc.returncode)
    dt = time.time() - start
    print(f"[{name}] ✅ done in {dt:.1f}s")


def ensure_tree(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def main():
    script_dir = Path(__file__).resolve().parent              # .../scripts
    project_root = script_dir.parents[1]                      # repo root
    py = sys.executable                                      # current Python

    # Resolve script locations (they live alongside this runner)
    scraper_py = script_dir / "scraper.py"
    ocr_py     = script_dir / "ocr.py"
    build_py   = script_dir / "build_index.py"
    query_py   = script_dir / "query.py"
    app_py     = project_root / "app.py"                      # app.py sits at repo root

    # Data tree (matches your scripts' defaults)
    data_dir     = project_root / "data"
    pdfs_dir     = data_dir / "pdfs"
    ocr_pdfs_dir = data_dir / "ocr_pdfs"
    ocr_txt_dir  = data_dir / "ocr_txt"
    index_dir    = data_dir / "index"

    p = argparse.ArgumentParser(description="Offline RAG pipeline runner")
    p.add_argument(
        "--steps",
        nargs="+",
        choices=["scrape", "ocr", "index", "query", "app", "all"],
        default=["all"],
        help="Which steps to run. 'all' == scrape ocr index.",
    )
    # OCR knobs (your ocr.py ignores --input and reads data/pdfs by default; these are useful)
    p.add_argument("--ocr-optimize", type=int, default=-1, help="OCRmyPDF optimize level (ocr.py --optimize)")
    p.add_argument("--ocr-lang", default="eng", help="OCR language(s), e.g. 'eng', 'eng+deu'")
    p.add_argument("--ocr-jobs", type=int, default=max(1, (os.cpu_count() or 2)), help="Parallel OCR jobs")
    p.add_argument("--ocr-reprocess", action="store_true", help="Recreate OCR outputs even if they exist")

    # Index build knobs (build_index.py already defaults to HF MiniLM)
    p.add_argument("--embed-provider", choices=["hf", "openai"], default="hf", help="Embedding provider for index")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    p.add_argument("--chunk-size", type=int, default=1600, help="Chunk size")
    p.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")

    # Query knobs (optional)
    p.add_argument("--ask", metavar="QUESTION", help="Ask a question using query.py")
    p.add_argument("--k", type=int, default=None, help="Top-K retrieved (overrides .env)")
    p.add_argument("--top-n", type=int, default=None, help="Top-N after rerank (overrides .env)")

    # App launch (optional)
    p.add_argument("--launch-app", action="store_true", help="Launch Streamlit app at the end (app.py)")

    args = p.parse_args()

    # Expand 'all'
    chosen = args.steps[:]
    if "all" in chosen:
        chosen = ["scrape", "ocr", "index"]

    # Basic data tree (safe if already present)
    ensure_tree(data_dir, pdfs_dir, ocr_pdfs_dir, ocr_txt_dir, index_dir)

    # 1) SCRAPE → data/pdfs  (crawler is static-configured for site and output)
    if "scrape" in chosen:
        run_step(
            "scrape",
            [py, str(scraper_py)],
            cwd=script_dir,
        )

    # 2) OCR + TEXT → data/ocr_pdfs + data/ocr_txt
    if "ocr" in chosen:
        # NOTE: your ocr.py reads PDFs from data/pdfs and writes to data/ocr_pdfs & data/ocr_txt by default.
        # We'll pass useful flags only.
        ocr_cmd = [
            py, str(ocr_py),
            "--optimize", str(args.ocr_optimize),
            "-l", args.ocr_lang,
            "-j", str(args.ocr_jobs),
        ]
        if args.ocr_reprocess:
            ocr_cmd.append("--reprocess")
        run_step("ocr", ocr_cmd, cwd=script_dir)

    # 3) BUILD INDEX → data/index (consumes text in data/ocr_txt)
    if "index" in chosen:
        run_step(
            "index",
            [
                py, str(build_py),
                "--persist", str(index_dir),
                "--embedding-provider", args.embed_provider,
                "--embedding-model", args.embed_model,
                "--chunk-size", str(args.chunk_size),
                "--chunk-overlap", str(args.chunk_overlap),
                # no --inputs passed → build_index.py auto-discovers data/ocr_txt
            ],
            cwd=script_dir,
        )

    # 4) QUERY (optional) → requires OPENAI_API_KEY if you run it
    if "query" in chosen:
        if not args.ask:
            print("[query] No --ask provided; skipping.")
        else:
            if not os.getenv("OPENAI_API_KEY"):
                print("[query] OPENAI_API_KEY not set; add it to your environment or .env, then re-run.")
                sys.exit(2)

            q_cmd = [py, str(query_py), args.ask, "--persist", str(index_dir)]
            if args.k is not None:
                q_cmd += ["--k", str(args.k)]
            if args.top_n is not None:
                q_cmd += ["--top-n", str(args.top_n)]
            run_step("query", q_cmd, cwd=script_dir)

    # 5) APP (optional)
    if args.launch_app or "app" in chosen:
        # Use 'streamlit' if available, otherwise 'python -m streamlit'
        streamlit = shutil.which("streamlit")
        if streamlit:
            cmd = [streamlit, "run", str(app_py)]
        else:
            cmd = [py, "-m", "streamlit", "run", str(app_py)]
        run_step("app", cmd, cwd=project_root)


if __name__ == "__main__":
    main()
