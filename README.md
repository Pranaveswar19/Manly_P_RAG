# Manly P. Hall RAG System

A Retrieval-Augmented Generation (RAG) system for querying Manly P. Hall's works using FAISS vector search, OCR processing, and OpenAI's language models.

> **Note:** This is a placeholder README. A more comprehensive version with detailed documentation will be added in the future.

## Overview

This project scrapes, processes, and indexes documents from Manly P. Hall's works, enabling semantic search and Q&A capabilities through a Streamlit web interface.

## Prerequisites

### Required Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### External Tools (for OCR)

The OCR pipeline requires the following tools to be installed on your system:

- **Tesseract OCR** - For text extraction from images
- **Ghostscript** - For PDF processing
- **QPDF** - For PDF manipulation

Optional (recommended for better OCR quality):
- pngquant
- jbig2enc
- unpaper

## Setup

### 1. Configure Environment Variables

Rename `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit the `.env` file and replace the comment with your OpenAI API key:

```env
OPENAI_API_KEY=your-actual-openai-api-key-here
RAG_PERSIST_DIR=data/index
RAG_LLM_MODEL=gpt-4o-mini
RAG_TOP_K=12
RAG_TOP_N=6
RAG_RERANK=false
```

### 2. Run the Pipeline in Sequence

Execute the following scripts in order:

#### Step 1: Scrape PDFs

```bash
python scripts/scraper.py
```

This will:
- Crawl the Manly P. Hall website
- Download all PDF documents
- Save them to `data/pdfs/`
- Generate a manifest file

#### Step 2: OCR Processing

```bash
python scripts/ocr.py
```

This will:
- Process all downloaded PDFs
- Apply OCR to extract text
- Save processed PDFs to `data/ocr_pdfs/`
- Extract text files to `data/ocr_txt/`

#### Step 3: Build Vector Index

```bash
python scripts/build_index.py
```

This will:
- Read all extracted text files
- Create embeddings using HuggingFace models
- Build a FAISS vector index
- Persist the index to `data/index/`

#### Step 4: Query the System (CLI)

```bash
python scripts/query.py "What is the symbolic meaning of the number 33 in Freemasonry?"
```

Optional arguments:
- `--k`: Number of candidates to retrieve (default: 12)
- `--top-n`: Number of results after reranking (default: 6)

#### Step 5: Launch Streamlit Web Interface

```bash
streamlit run app.py
```

This will start the web interface at `http://localhost:8501` where you can:
- Ask questions interactively
- Adjust retrieval parameters (Top-K, Top-N)
- View source documents
- Get answers with reranking

## Project Structure

```
Manly_P_RAG/
├── scripts/
│   ├── scraper.py        # Step 1: Download PDFs
│   ├── ocr.py            # Step 2: OCR processing
│   ├── build_index.py    # Step 3: Build vector index
│   └── query.py          # Step 4: CLI query interface
├── app.py                # Step 5: Streamlit web interface
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
└── data/
    ├── pdfs/             # Downloaded PDFs
    ├── ocr_pdfs/         # OCR-processed PDFs
    ├── ocr_txt/          # Extracted text files
    └── index/            # FAISS vector index
```

## Technology Stack

- **LlamaIndex** - RAG orchestration framework
- **FAISS** - Vector similarity search
- **OpenAI** - Language model for responses
- **HuggingFace Transformers** - Embeddings and reranking
- **Streamlit** - Web interface
- **OCRmyPDF** - PDF OCR processing

## Notes

- Ensure you have a valid OpenAI API key before running queries
- The scraper and OCR steps may take considerable time depending on the number of documents
- The FAISS index uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings (384 dimensions)
- Reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for improved relevance

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
