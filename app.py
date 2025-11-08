import os
import io
import re
import math
import tempfile
import base64
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PDF / OCR libs
from PyPDF2 import PdfReader
import pdfplumber
from PIL import Image
import pytesseract

# LangChain / Embeddings / FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

load_dotenv()

# ============ Config / Env ============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", None)  # set if using EURI proxy
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index_bajaj")
EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL_NAME = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in environment. Add it to .env and restart.")
    st.stop()

# ensure env keys are set for client libraries
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

# ============ Streamlit UI config ============
st.set_page_config(page_title="Bajaj Finserv Factsheet Chatbot", page_icon="📈", layout="wide")
st.title("📊 Bajaj Finserv AMC Factsheet Chatbot (RAG PoC)")
st.markdown(
    "Upload Bajaj AMC factsheets, index them, and ask questions. Answers are grounded in the uploaded documents only."
)

# ============ Session state ============
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "docs_meta" not in st.session_state:
    st.session_state["docs_meta"] = []  # metadata for docs ingested
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of (question, answer)

# ============ Helpers: PDF ingestion & parsing ============
def extract_text_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text per page using PyPDF2 + fallback to pdfplumber for tables."""
    pages = []
    try:
        reader = PdfReader(pdf_path)
        n = len(reader.pages)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text, "source_file": os.path.basename(pdf_path)})
    except Exception as e:
        # fallback: pdfplumber full extraction
        with pdfplumber.open(pdf_path) as pdf:
            for i, p in enumerate(pdf.pages):
                text = p.extract_text() or ""
                pages.append({"page": i + 1, "text": text, "source_file": os.path.basename(pdf_path)})
    return pages

def extract_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tables using pdfplumber and return list of dicts with DataFrames."""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for t in page_tables:
                    if not t:
                        continue
                    try:
                        df = pd.DataFrame(t[1:], columns=t[0])
                        tables.append({"page": i + 1, "table": df, "source_file": os.path.basename(pdf_path)})
                    except Exception:
                        # if columns mismatch, try naive conversion
                        try:
                            df = pd.DataFrame(t)
                            tables.append({"page": i + 1, "table": df, "source_file": os.path.basename(pdf_path)})
                        except Exception:
                            continue
    except Exception:
        pass
    return tables

def ocr_page_image(pdf_path: str, page_number: int) -> str:
    """
    Simple OCR: convert a single page to image via pdfplumber page.to_image and run pytesseract.
    Note: This is a fallback and can be slower.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            pil_image = page.to_image(resolution=200).original
            text = pytesseract.image_to_string(pil_image)
            return text
    except Exception:
        return ""

def chunk_texts(pages: List[Dict[str, Any]], chunk_size_chars: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Simple chunking: split page text into overlapping chunks.
    Returns list of dicts: {text,page,chunk_id,source_file,type}
    """
    chunks = []
    for p in pages:
        text = p.get("text", "") or ""
        if len(text) < 200:
            # try OCR fallback
            ocr_text = ocr_page_image(p["source_file"], p["page"]) if os.path.exists(p["source_file"]) else ""
            text = (text + "\n" + ocr_text).strip()
        if not text:
            continue
        start = 0
        page_num = p["page"]
        sid = p.get("source_file", "unknown.pdf")
        while start < len(text):
            end = start + chunk_size_chars
            chunk = text[start:end]
            chunk_id = f"{sid}_p{page_num}_c{start}"
            chunks.append({"text": chunk, "page": page_num, "chunk_id": chunk_id, "source_file": sid, "type": "text"})
            start = end - overlap
    return chunks

def table_chunks_from_tables(tables: List[Dict[str, Any]]) -> List[Dict]:
    table_chunks = []
    for t in tables:
        df = t["table"]
        # small cleaning of headers and numeric formatting
        try:
            csv_text = df.to_csv(index=False)
        except Exception:
            csv_text = str(df.head(20))
        chunk_id = f"{t['source_file']}_p{t['page']}_table"
        table_chunks.append({"text": csv_text, "page": t["page"], "chunk_id": chunk_id, "source_file": t["source_file"], "type": "table", "table": df})
    return table_chunks

# ============ Helpers: Computation layer ============
def cagr_from_start_end(start_value: float, end_value: float, years: float) -> float:
    if start_value is None or start_value == 0:
        return None
    return (end_value / start_value) ** (1.0 / years) - 1.0

def sharpe_ratio(returns: List[float], rf_annual: float = 0.0, periods_per_year: int = 12) -> float:
    import numpy as _np
    r = _np.array(returns, dtype=float)
    if r.size == 0:
        return None
    periodic_rf = (1 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - periodic_rf
    mean_excess = excess.mean() * periods_per_year
    std_annual = r.std(ddof=1) * (periods_per_year ** 0.5)
    if std_annual == 0:
        return None
    return mean_excess / std_annual

# quick numeric parser to pull numbers out of CSV-like table text
def parse_numeric_series_from_table_csv(csv_text: str, value_col_candidates=None):
    if value_col_candidates is None:
        value_col_candidates = ["NAV", "Close", "Value", "Return", "Returns"]
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None
    # find numeric column
    for c in df.columns:
        if any(vc.lower() in c.lower() for vc in value_col_candidates):
            try:
                series = pd.to_numeric(df[c].astype(str).str.replace("%", "").str.replace(",", ""), errors="coerce").dropna().values
                return series
            except Exception:
                continue
    # fallback: try any numeric column
    for c in df.columns:
        series = pd.to_numeric(df[c].astype(str).str.replace("%", "").str.replace(",", ""), errors="coerce").dropna().values
        if len(series) > 0:
            return series
    return None

# ============ Embeddings / FAISS helpers ============
def create_or_load_vectorstore(embedding_model, index_path=FAISS_INDEX_DIR):
    """Attempt to load existing FAISS index from disk, else return None."""
    try:
        if os.path.exists(index_path):
            db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
            return db
    except Exception:
        return None
    return None

def build_faiss_from_chunks(chunks: List[Dict], embedding_model, index_path=FAISS_INDEX_DIR):
    """
    Build and save FAISS index from chunk dicts.
    chunks: list of {text,page,chunk_id,source_file,type,...}
    Uses FAISS.from_texts which will call embedding_model to embed texts.
    """
    texts = [c["text"] if c.get("text") else "" for c in chunks]
    metadatas = [{"page": c["page"], "chunk_id": c["chunk_id"], "source_file": c["source_file"], "type": c["type"]} for c in chunks]
    # Using from_texts is simpler; it will embed internally using the embedding model
    db = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
    db.save_local(index_path)
    return db

# ============ LLM / Prompt utilities ============
PROMPT_TPL = """
You are an assistant whose knowledge is strictly limited to the following source chunks.
Use ONLY the information in the provided chunks to answer the user's question.
If the answer is not contained in the chunks, reply exactly: "I cannot find that information in the uploaded factsheets."

User question:
{question}

Retrieved chunks (each chunk includes: file, page, chunk_id, similarity): 
{retrieved_text}

Rules:
- Do not use any external knowledge other than the chunks shown above.
- When you state a fact, cite which chunk it came from by "file:..., page:..., chunk_id:...".
- For calculations, show steps and which table/values you used.

Answer:
"""

prompt_template = PromptTemplate(input_variables=["question", "retrieved_text"], template=PROMPT_TPL)

def answer_with_context(llm, retrieved_chunks: List[Dict], question: str, conversation_history: List[Dict]=None):
    # build retrieved_text block
    entries = []
    for doc, score in retrieved_chunks:
        meta = doc.metadata if hasattr(doc, "metadata") else getattr(doc, "metadata", {})
        snippet = doc.page_content[:800] if hasattr(doc, "page_content") else (doc.get("text","")[:800])
        entries.append(f"file: {meta.get('source_file','unknown')}, page: {meta.get('page','?')}, chunk_id: {meta.get('chunk_id','?')}, similarity: {score:.4f}\n---\n{snippet}\n---")
    retrieved_text = "\n\n".join(entries)
    # conversation_history can be included if needed (for follow-ups); here we only include prior Q/A as context optionally
    if conversation_history:
        hist_txt = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history])
        retrieved_text = hist_txt + "\n\n" + retrieved_text
    chain = LLMChain(llm=llm, prompt=prompt_template)
    out = chain.run(question=question, retrieved_text=retrieved_text)
    return out

# ============ Streamlit UI: sidebar for upload / index ============
st.sidebar.header("Indexing & Settings")

# Choose models (embedding + LLM) — uses langchain_openai wrappers
use_euri = False
if OPENAI_BASE_URL and OPENAI_BASE_URL.strip():
    use_euri = True

st.sidebar.markdown(f"**Embedding model:** `{EMBEDDING_MODEL_NAME}`")
st.sidebar.markdown(f"**LLM model:** `{LLM_MODEL_NAME}`")
st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader("Upload factsheet PDF(s) (Oct 2025, etc.)", type=["pdf"], accept_multiple_files=True)
index_button = st.sidebar.button("Ingest & Build Index")

# allow loading existing FAISS directory
if st.sidebar.button("Load existing index"):
    # create embedding model instance
    try:
        if use_euri:
            embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_BASE_URL)
        else:
            embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        db = create_or_load_vectorstore(embedding_model, FAISS_INDEX_DIR)
        if db:
            st.session_state["vectorstore"] = db
            st.sidebar.success("Loaded FAISS index from disk.")
        else:
            st.sidebar.warning("No FAISS index found on disk.")
    except Exception as e:
        st.sidebar.error(f"Error loading index: {e}")

if index_button:
    # ingest uploaded PDFs and build index
    if not uploaded_files:
        st.sidebar.error("Please upload one or more PDF factsheets first.")
    else:
        with st.spinner("Ingesting PDFs and building FAISS index..."):
            all_chunks = []
            docs_meta = []
            for up in uploaded_files:
                # save uploaded file to a temp path because our parsing functions expect filepath
                tmp_path = os.path.join(tempfile.gettempdir(), up.name)
                with open(tmp_path, "wb") as f:
                    f.write(up.getbuffer())
                # extract text pages
                pages = extract_text_pages(tmp_path)
                # but ensure source_file path so OCR fallback can reference
                for p in pages:
                    p["source_file"] = tmp_path
                tables = extract_tables(tmp_path)
                # chunk text
                text_chunks = chunk_texts(pages)
                table_chks = table_chunks_from_tables(tables)
                all_chunks.extend(text_chunks)
                all_chunks.extend(table_chks)
                docs_meta.append({"filename": up.name, "path": tmp_path, "pages": len(pages)})
            # create embeddings and FAISS
            try:
                if use_euri:
                    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=OPENAI_API_KEY, openai_api_base=OPENAI_BASE_URL)
                else:
                    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
                db = build_faiss_from_chunks(all_chunks, embedding_model, index_path=FAISS_INDEX_DIR)
                st.session_state["vectorstore"] = db
                st.session_state["docs_meta"] = docs_meta
                st.sidebar.success(f"Index built and saved to '{FAISS_INDEX_DIR}' with {len(all_chunks)} chunks.")
            except Exception as e:
                st.sidebar.error(f"Error building index: {e}")

# ============ Main Chat UI ============
st.markdown("---")
st.subheader("Ask questions from the uploaded factsheet(s)")

query = st.text_input("Your question (e.g., 'What is the 3-year return of Bajaj Flexi Cap Fund?')", key="query_input")
top_k = st.slider("Top-K retrieval", 1, 8, 4)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    elif st.session_state["vectorstore"] is None:
        st.error("No index loaded. Upload & build or load an existing FAISS index first (sidebar).")
    else:
        db: FAISS = st.session_state["vectorstore"]
        # retrieve top-k with score
        try:
            results = db.similarity_search_with_score(query, k=top_k)
        except Exception:
            # fallback to retriever.as_retriever
            retriever = db.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.get_relevant_documents(query)
            results = [(d, 0.0) for d in docs]
        # Build answer using LLM with strict prompt
        if use_euri:
            llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, openai_api_base=OPENAI_BASE_URL, openai_api_key=OPENAI_API_KEY)
        else:
            llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)
        # answer
        answer_text = answer_with_context(llm, results, query, conversation_history=st.session_state["chat_history"])
        st.session_state["chat_history"].append((query, answer_text))
        st.markdown("### 🧠 Answer (grounded to uploaded factsheets)")
        st.info(answer_text)

        # display sources & confidence
        st.markdown("### 📚 Sources & Confidence")
        rows = []
        for doc, score in results:
            meta = doc.metadata if hasattr(doc, "metadata") else doc
            snippet = doc.page_content[:400] if hasattr(doc, "page_content") else (doc.get("text","")[:400])
            rows.append({
                "file": meta.get("source_file", "Unknown"),
                "page": meta.get("page", "?"),
                "chunk_id": meta.get("chunk_id", "N/A"),
                "score": round(float(score), 6),
                "snippet": snippet
            })
        df_src = pd.DataFrame(rows)
        if not df_src.empty:
            st.dataframe(df_src[["file", "page", "chunk_id", "score"]])
            # show snippets
            for i, r in df_src.iterrows():
                st.markdown(f"**Source:** `{r['file']}` | Page: {r['page']} | Chunk: `{r['chunk_id']}` | Score: {r['score']}")
                st.write(r["snippet"] + "...")
        else:
            st.write("No sources retrieved.")

        # If answer mentions comparison table, try to parse and visualize
        def extract_comparison_table_from_answer(ans_text: str) -> pd.DataFrame:
            # naive parser: look for lines with fund name and 1/3/5 year numeric columns
            pattern = re.compile(r"([A-Za-z0-9 &\-\(\)\.]+)[\|\-\:]{1,}\s*([\d\.]+)[^\d\n]+([\d\.]+)[^\d\n]+([\d\.]+)")
            rows = []
            for ln in ans_text.splitlines():
                m = pattern.search(ln)
                if m:
                    fund = m.group(1).strip()
                    try:
                        a1 = float(m.group(2))
                        a3 = float(m.group(3))
                        a5 = float(m.group(4))
                        rows.append({"Fund": fund, "1Y": a1, "3Y": a3, "5Y": a5})
                    except Exception:
                        continue
            if rows:
                return pd.DataFrame(rows)
            return pd.DataFrame()

        df_comp = extract_comparison_table_from_answer(answer_text)
        if not df_comp.empty:
            st.markdown("### 📊 Comparison Visualization")
            st.dataframe(df_comp)
            fig, ax = plt.subplots(figsize=(8, 4))
            df_comp.set_index("Fund")[["1Y", "3Y", "5Y"]].plot(kind="bar", ax=ax)
            ax.set_ylabel("Return (%)")
            ax.set_title("Fund Performance Comparison")
            st.pyplot(fig)

# ============ Footer / Examples ============
st.markdown("---")
st.markdown("#### Example questions")
st.markdown(
    """
- What is the 3-year return of **Bajaj Flexi Cap Fund**?  
- List top 5 holdings of the **Consumption Fund** with weights.  
- Compare allocation between equity and debt for Bajaj Balanced Fund.  
- How has AUM changed compared to last month?  
- Which equity fund has the highest 3-year return?  
- State the YTM, Macaulay Duration and Average Maturity for the Money Market Fund.
"""
)
st.markdown(
    "<div style='text-align:center; color:grey;'>🚀 PoC by Omkar Bhosale — Answers come only from uploaded factsheets.</div>",
    unsafe_allow_html=True
)
