# app.py
# Minimal core chat engine for your GovHack MVP (Streamlit + RAG + citations + audit JSON)
# - Drop CSVs/PDFs/text files (TXT, MD, JSON, XML, HTML, YAML, etc.) in ./data
# - Creates/loads a FAISS vector index in ./index
# - Answers strictly from retrieved chunks (safe fallback if not found)
# - Shows sources + a simple trust score + an expandable audit pane
#
# Quick start:
#   pip install streamlit langchain faiss-cpu pypdf python-dotenv tiktoken openai sentence-transformers
#   export OPENAI_API_KEY=...   # or set in a .env file
#   streamlit run app.py
#
# Notes:
# - Uses OpenAI for both LLM and embeddings by default.
# - You can toggle to SentenceTransformers (local) embeddings if you prefer.
# - Keep the dataset tiny for hackathon speed (2 datasets + 1 short policy doc).

from __future__ import annotations
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# LangChain + Embeddings + Vectorstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Loaders
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader

load_dotenv()
DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

# -------------- UI CONFIG --------------
st.set_page_config(page_title="GovHack Core Chat", page_icon="🤖", layout="wide")
st.title("🤖 Core Chat Engine — RAG with Citations & Audit")
st.caption("Drop files (PDFs, CSVs, TXT, MD, JSON, XML, HTML, YAML, etc.) into ./data then click \"(Re)build index\". Ask questions, get grounded answers with sources.")

# -------------- SIDEBAR: SETTINGS --------------
with st.sidebar:
    st.header("Settings")
    use_openai = st.toggle("Use OpenAI embeddings+LLM", value=True)
    top_k = st.slider("Retriever k", 2, 8, 4)
    chunk_size = st.slider("Chunk size", 300, 1500, 900, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 300, 120, step=10)
    model_name = st.text_input("LLM model (OpenAI)", value="gpt-4o-mini")
    embed_model = st.text_input("Embeddings (OpenAI/HF)", value=("text-embedding-3-small" if use_openai else "sentence-transformers/all-MiniLM-L6-v2"))

    st.divider()
    if st.button("(Re)build index", type="primary"):
        st.session_state.rebuild = True

# -------------- HELPERS --------------

def load_docs(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for p in sorted(data_dir.rglob("*")):
        if p.is_dir():
            continue
        meta = {"source": str(p), "filename": p.name}
        try:
            if p.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(p))
                docs += loader.load()
            elif p.suffix.lower() == ".csv":
                # CSVLoader creates one doc per row; we combine columns into text
                loader = CSVLoader(str(p), encoding="utf-8")
                docs += loader.load()
            elif p.suffix.lower() in {".txt", ".md", ".log", ".json", ".xml", ".html", ".rst", ".yaml", ".yml"}:
                # Try different encodings for better text file support
                for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                    try:
                        loader = TextLoader(str(p), encoding=encoding)
                        docs += loader.load()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, try with error handling
                    try:
                        loader = TextLoader(str(p), encoding="utf-8", autodetect_encoding=True)
                        docs += loader.load()
                    except Exception:
                        st.warning(f"Could not decode text file {p.name} with any encoding")
            else:
                continue
            # attach filename to each
            for d in docs[-5:]:
                d.metadata.update(meta)
        except Exception as e:
            st.warning(f"Failed to load {p.name}: {e}")
    return docs


def split_docs(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_or_load_index(split: List[Document], use_openai: bool, embed_model: str) -> FAISS:
    idx_path = INDEX_DIR / "faiss_index"
    meta_path = INDEX_DIR / "index_meta.json"

    if idx_path.exists() and meta_path.exists() and not st.session_state.get("rebuild", False):
        try:
            vs = FAISS.load_local(str(idx_path), OpenAIEmbeddings(model=embed_model) if use_openai else HuggingFaceEmbeddings(model_name=embed_model), allow_dangerous_deserialization=True)
            return vs
        except Exception:
            pass  # fall through to rebuild

    # Build new
    embeddings = OpenAIEmbeddings(model=embed_model) if use_openai else HuggingFaceEmbeddings(model_name=embed_model)
    vs = FAISS.from_documents(split, embeddings)
    vs.save_local(str(idx_path))
    with open(meta_path, "w") as f:
        json.dump({"embed_model": embed_model, "use_openai": use_openai, "ts": time.time()}, f)
    st.session_state.rebuild = False
    return vs


def compute_trust_score(similarities: List[float], distinct_sources: int, recency_flags: List[int]) -> int:
    # Simple heuristic: 60% from sim, 25% from source count, 15% from recency
    if not similarities:
        return 0
    sim = sum(similarities) / len(similarities)  # 0..1
    src = min(distinct_sources, 3) / 3  # cap at 3 sources
    rec = (sum(recency_flags) / max(1, len(recency_flags)))  # 0/1 flags
    score = 0.60 * sim + 0.25 * src + 0.15 * rec
    return int(round(score * 100))


def extract_recency_flag(path_str: str) -> int:
    # crude: file mtime within last 3 years → 1 else 0
    try:
        mtime = Path(path_str).stat().st_mtime
        three_years = 60 * 60 * 24 * 365 * 3
        return 1 if (time.time() - mtime) < three_years else 0
    except Exception:
        return 0


def grounded_answer(llm: ChatOpenAI, question: str, contexts: List[str]) -> str:
    system = (
        "You are a careful government data assistant. "
        "Answer ONLY using the provided sources. If the answer is not in the sources, say you don't have enough information and suggest what data to add."
    )
    content = "\n\n---\nSOURCES:\n" + "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = f"Question: {question}\n\nUse only the SOURCES. Cite as [1], [2]... inline."
    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt + content},
    ]
    resp = llm.invoke(msg)
    return resp.content

# -------------- INDEX BUILD (on first run) --------------
if "vectorstore" not in st.session_state or st.session_state.get("rebuild", False):
    with st.spinner("Loading documents from ./data ..."):
        raw_docs = load_docs(DATA_DIR)
    with st.spinner("Splitting into chunks ..."):
        split = split_docs(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    with st.spinner("Creating / loading index ..."):
        vectorstore = build_or_load_index(split, use_openai=use_openai, embed_model=embed_model)
    st.session_state["vectorstore"] = vectorstore

# -------------- CHAT AREA --------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {q,a,audit}

col_chat, col_audit = st.columns([0.62, 0.38])

with col_chat:
    user_q = st.text_input("Ask a question about the ingested data...", placeholder="e.g., What is the latest ABS employment figure?")
    ask = st.button("Ask", use_container_width=True)

    if ask and user_q.strip():
        vs: FAISS = st.session_state["vectorstore"]
        docs_and_scores = vs.similarity_search_with_score(user_q, k=top_k)
        docs: List[Document] = [d for d, _ in docs_and_scores]
        
        # Convert FAISS L2 distances to similarities (more robust approach)
        # For normalized embeddings, L2 distance relates to cosine similarity as: cos_sim = 1 - (L2_dist^2 / 2)
        sims: List[float] = []
        for _, distance in docs_and_scores:
            # Ensure distance is non-negative and convert to similarity
            distance = max(0.0, float(distance))
            # Convert L2 distance to approximate cosine similarity
            similarity = max(0.0, 1.0 - (distance * distance / 2.0))
            similarity = min(1.0, similarity)  # Clamp to [0, 1]
            sims.append(similarity)
        
        # Note: FAISS returns L2 distances; converted to cosine similarity approximation

        contexts = []
        sources = []
        recency_flags = []
        for d in docs:
            src = d.metadata.get("source", d.metadata.get("filename", "unknown"))
            snippet = d.page_content[:900]
            contexts.append(f"{snippet}\n(Source: {src})")
            sources.append(src)
            recency_flags.append(extract_recency_flag(src))

        distinct_sources = len(set(sources))
        trust = compute_trust_score(sims, distinct_sources, recency_flags)

        # LLM
        if use_openai:
            llm = ChatOpenAI(model=model_name, temperature=0)
        else:
            # Fallback: if you wire a local model, replace this with your client
            st.error("Local LLM not configured. Enable OpenAI in sidebar.")
            llm = ChatOpenAI(model=model_name, temperature=0)

        answer = grounded_answer(llm, user_q, contexts)

        audit = {
            "question": user_q,
            "trust_score": trust,
            "retrieved": [
                {
                    "source": s,
                    "similarity": sims[i] if i < len(sims) else None,
                    "recency_flag": recency_flags[i] if i < len(recency_flags) else None,
                    "preview": docs[i].page_content[:300] if i < len(docs) else None,
                }
                for i, s in enumerate(sources)
            ],
            "timestamp": time.time(),
        }

        st.session_state.history.append({"q": user_q, "a": answer, "audit": audit})

    # Render history
    for turn in st.session_state.history[::-1]:  # newest first
        st.markdown(f"**You:** {turn['q']}")
        st.write(turn["a"])  # model answer with inline [1],[2] citations
        with st.expander("Why this answer? (audit)"):
            st.json(turn["audit"], expanded=False)
        st.divider()

with col_audit:
    st.subheader("📊 Trust Meter (latest)")
    if st.session_state.history:
        latest = st.session_state.history[-1]["audit"]
        st.metric(label="Trust score", value=f"{latest['trust_score']} / 100")
        st.caption("Heuristic: embeddings similarity, number of distinct sources, recency flag")
        st.subheader("Sources")
        for r in latest["retrieved"]:
            st.markdown(f"- **{Path(r['source']).name}** — recency: {'✅' if r['recency_flag'] else '⚠️'}")
    else:
        st.info("Ask something to see trust scores and sources here.")

st.caption("© GovHack MVP — core chat engine. Strict grounding; graceful refusal when not in sources.")
