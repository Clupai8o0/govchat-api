# GovChat API — Retrieval-friendly pipeline for Australian government datasets

Tooling to scrape gov.au data sources, normalize ABS metadata, build OpenAI embeddings, and serve a FastAPI endpoint for dataset discovery with trust signals. Currently hosted on Render.

## What it does
- Crawls *.gov.au pages to collect dataset file links and page context into CSV/JSONL.
- Normalizes ABS dataflow metadata into a consistent dataset catalog.
- Generates embeddings with OpenAI and loads them into a ChromaDB vector store.
- Serves a FastAPI API that answers natural-language queries with grounded dataset matches, similar-dataset lookups, and audit logs.

## Key features
- **Government data scraping**: `gov_data_scraper.py` discovers CSV/XLS/XLSX/PDF/ZIP links on *.gov.au with robots.txt respect; `rag_crawler.py` produces RAG-friendly HTML chunks (optional Playwright JS rendering).
- **ABS metadata normalization**: `Normalizer.py` flattens ABS dataflow JSON into a uniform CSV (id, title, description, agency, frequency, tags, API URL).
- **Embedding pipeline**: `statistics-server/make_embeddings_openai.py` builds OpenAI `text-embedding-3-small` vectors; `embeddings/` utilities to inspect/export embeddings.
- **Vector store setup**: `statistics-server/load_vectors.py` seeds a persistent Chroma collection (`statistics-server/vector_store`) from CSV + embeddings.
- **Retrieval API**: FastAPI app (`statistics-server/main.py`) exposes `/ping`, `/query?q=...` (top-4 matches + friendly answer + trust factors), `/similar/{dataset_id}`, and `/audit/{audit_id}`.
- **Trust & transparency**: Metadata-only grounding, similarity-based trust scoring, and persisted audit logs (`statistics-server/audit_logs`, gitignored).
- **CORS ready**: Preconfigured origins for localhost and `govchat-*-vercel.app` frontends.

## Tech stack
Python, FastAPI, Uvicorn, ChromaDB, OpenAI API, pandas/numpy, aiohttp/requests, optional Playwright for JS crawling.

## Architecture overview
Data flows from crawlers into a catalog, gets embedded, and is served via the API.

```
[gov_data_scraper.py | rag_crawler.py | Normalizer.py]
        -> datasets.csv (+ optional gov_data_results*.csv/jsonl)
        -> make_embeddings_openai.py -> embeddings.pkl
        -> load_vectors.py -> statistics-server/vector_store (ChromaDB)
        -> FastAPI app (statistics-server/main.py) -> /query, /similar, /audit
```

## Getting started (local)
### Prerequisites
- Python (compatible with FastAPI/ChromaDB stack) and pip.
- OpenAI API key for embedding generation and GPT-powered responses.
- Optional: Playwright (`pip install playwright` + `playwright install`) if using `--js` crawling.

### Install
```bash
cd statistics-server
python -m venv .venv && .\.venv\Scripts\activate  # or source .venv/bin/activate on Unix
pip install -r requirements.txt
```

### Environment variables
Set in `.env` or shell:
- `OPENAI_API_KEY` (required for embedding generation and API responses).

### Data & embeddings
The repo includes sample `statistics-server/datasets.csv` and `statistics-server/embeddings.pkl`. To regenerate:
```bash
# Normalize ABS dataflow JSON into a catalog
python ..\Normalizer.py --in <path-to-abs-dataflows-json-or-dir> --out statistics-server/datasets.csv

# Build embeddings with OpenAI
python make_embeddings_openai.py

# Create/load Chroma vector store
python load_vectors.py
```

### Run the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```
Health check: `curl http://localhost:8001/ping`.

## Usage
- Query datasets: `curl "http://localhost:8001/query?q=education statistics"`
- Find similar datasets: `curl "http://localhost:8001/similar/<dataset_id>"`
- Retrieve an audit record: `curl "http://localhost:8001/audit/<audit_id>"`

Crawling helpers:
```bash
# Gov data file discovery
python ..\gov_data_scraper.py --seeds https://www.abs.gov.au/ --max-pages 800 --max-files 200 --outfile results_abs

# RAG-friendly crawl (HTML chunks)
python ..\rag_crawler.py --seeds https://hackerspace.govhack.org/... --output out.csv --max-pages 300 --concurrency 6
```

## Testing / Quality
No automated tests are included. Recommended: run smoketests against `/ping`, `/query`, and `/similar` after setting up the vector store.

## Deployment
Currently hosted on Render. No deployment scripts are included; run the FastAPI app with Uvicorn (or your preferred ASGI host) after preparing `statistics-server/vector_store`.

## Credits / Contributors
Shalok Sharma, Ranveer Bhasin, contributors to this repository.
