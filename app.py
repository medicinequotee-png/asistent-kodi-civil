import hashlib
import io
import re
from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np
import pdfplumber
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_TOP_K = 5
DEFAULT_SNIPPET_LEN = 700


def clean_text(text: str) -> str:
    text = (text or "").replace("\u00ad", "")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalize_article_number(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"\s+", "", raw)
    return raw


def split_articles(full_text: str) -> List[Dict[str, str]]:
    pattern = re.compile(r"\bNeni\s+(\d+(?:/\d+)?[a-zA-Z]?)\b", re.IGNORECASE)
    matches = list(pattern.finditer(full_text))
    articles: List[Dict[str, str]] = []

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        num = normalize_article_number(match.group(1))
        body = clean_text(full_text[start:end])
        if not body:
            continue

        lines = [line.strip() for line in body.split("\n") if line.strip()]
        title = lines[0] if lines else ""
        if re.match(r"(?i)^neni\\s+", title) and len(lines) > 1:
            title = lines[1]

        articles.append(
            {
                "num": num,
                "label": f"Neni {num}",
                "title": title,
                "text": body,
            }
        )

    return articles


def is_pdf_bytes(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@st.cache_data(show_spinner=False)
def load_articles_from_bytes(pdf_bytes: bytes, pdf_hash: str) -> List[Dict[str, str]]:
    if not is_pdf_bytes(pdf_bytes):
        raise ValueError("File nuk eshte PDF i vlefshem (mungon header %PDF).")

    pages_text: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages_text.append(clean_text(page.extract_text() or ""))

    full_text = "\n".join(pages_text)
    return split_articles(full_text)


def build_dense_from_tfidf(tfidf_matrix) -> Tuple[np.ndarray, Optional[TruncatedSVD]]:
    if min(tfidf_matrix.shape) <= 2:
        return tfidf_matrix.toarray(), None

    n_components = min(256, max(2, min(tfidf_matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense = svd.fit_transform(tfidf_matrix)
    return dense, svd


def build_ann_index(dense_vectors: np.ndarray) -> hnswlib.Index:
    dense_vectors = normalize(dense_vectors).astype(np.float32)
    dim = dense_vectors.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(dense_vectors), ef_construction=200, M=16)
    index.add_items(dense_vectors, np.arange(len(dense_vectors)))
    index.set_ef(50)
    return index


@st.cache_resource(show_spinner=False)
def build_indices(articles: List[Dict[str, str]], pdf_hash: str) -> Dict[str, object]:
    corpus = [a["text"] for a in articles]
    if not corpus:
        raise ValueError("No articles found in PDF.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(corpus)
    tfidf_norm = normalize(tfidf)

    embeddings, svd = build_dense_from_tfidf(tfidf)
    ann_index = build_ann_index(embeddings)

    return {
        "vectorizer": vectorizer,
        "tfidf_norm": tfidf_norm,
        "svd": svd,
        "ann_index": ann_index,
    }


def parse_article_query(q: str) -> Optional[str]:
    q = q.strip()
    if not q:
        return None
    match = re.search(r"\bNeni\s+(\d+(?:/\d+)?[a-zA-Z]?)\b", q, re.IGNORECASE)
    if match:
        return normalize_article_number(match.group(1))
    if q.isdigit():
        return q
    return None


def embed_query(query: str, vectorizer: TfidfVectorizer, svd: Optional[TruncatedSVD]) -> np.ndarray:
    tfidf_vec = vectorizer.transform([query])
    if svd is None:
        dense = tfidf_vec.toarray()
    else:
        dense = svd.transform(tfidf_vec)
    return normalize(dense)


def search_hybrid(
    query: str,
    top_k: int,
    vectorizer: TfidfVectorizer,
    tfidf_norm,
    ann_index: hnswlib.Index,
    svd: Optional[TruncatedSVD],
    alpha: float = 0.65,
) -> List[Tuple[int, float]]:
    tfidf_query = vectorizer.transform([query])
    tfidf_query = normalize(tfidf_query)
    lexical_scores = (tfidf_norm @ tfidf_query.T).toarray().ravel()

    q_embed = embed_query(query, vectorizer, svd)
    labels, distances = ann_index.knn_query(q_embed, k=min(top_k * 3, len(lexical_scores)))
    semantic_scores = {int(lbl): 1.0 - float(dist) for lbl, dist in zip(labels[0], distances[0])}

    candidate_ids = set(np.argsort(lexical_scores)[-top_k * 3 :])
    candidate_ids.update(semantic_scores.keys())

    results: List[Tuple[int, float]] = []
    for idx in candidate_ids:
        lex = float(lexical_scores[idx])
        sem = float(semantic_scores.get(idx, 0.0))
        score = alpha * sem + (1.0 - alpha) * lex
        results.append((int(idx), score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def format_snippet(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def fallback_answer(results: List[Tuple[int, float]], articles: List[Dict[str, str]]) -> str:
    if not results:
        return "Nuk gjeta nene te pershtatshme."
    top = results[0][0]
    art = articles[top]
    return f"Neni me relevant duket: {art['label']} - {art['title']}."


st.set_page_config(page_title="Asistent Kodi Civil", layout="wide")
st.title("Asistent Kodi Civil")
st.caption("Kerkim semantik dhe gjetje e shpejte e neneve nga PDF.")

with st.sidebar:
    st.subheader("Opsione")
    uploaded_pdf = st.file_uploader("Ngarko PDF (vetem PDF)", type=["pdf"])
    if uploaded_pdf is not None:
        size_mb = len(uploaded_pdf.getbuffer()) / (1024 * 1024)
        st.caption(f"PDF size: {size_mb:.2f} MB")
    top_k = st.slider("Rezultate", 1, 10, DEFAULT_TOP_K)
    snippet_len = st.slider("Gjatesia e fragmentit", 200, 1400, DEFAULT_SNIPPET_LEN)
    show_full = st.checkbox("Shfaq tekstin e plote", False)
    alpha = st.slider("Pesha semantike", 0.0, 1.0, 0.65)
    st.markdown("Keshille: shkruaj `Neni 5` ose vetem `5` per ta hapur direkt.")

if uploaded_pdf is None:
    st.info("Ngarko PDF per te filluar.")
    st.stop()

pdf_bytes = uploaded_pdf.getbuffer().tobytes()
if not is_pdf_bytes(pdf_bytes):
    st.error("Skedari nuk duket PDF i vlefshem. Provo perseri.")
    st.stop()

pdf_hash = hash_bytes(pdf_bytes)

with st.spinner("Duke lexuar PDF..."):
    try:
        articles = load_articles_from_bytes(pdf_bytes, pdf_hash)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

with st.spinner("Duke indeksuar..."):
    indices = build_indices(articles, pdf_hash)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input("Shkruaj pyetjen ne shqip...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    wanted_num = parse_article_query(user_query)
    if wanted_num:
        match = next((a for a in articles if a["num"] == wanted_num), None)
        with st.chat_message("assistant"):
            if not match:
                st.warning(f"Neni {wanted_num} nuk u gjet.")
            else:
                st.markdown(f"### {match['label']} - {match['title']}")
                st.write(match["text"])
        st.stop()

    results = search_hybrid(
        user_query,
        top_k,
        indices["vectorizer"],
        indices["tfidf_norm"],
        indices["ann_index"],
        indices["svd"],
        alpha=alpha,
    )

    answer = fallback_answer(results, articles)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("Nenet relevante"):
            for idx, score in results:
                art = articles[idx]
                st.markdown(f"**{art['label']} - {art['title']}**")
                st.caption(f"Relevance: {score:.3f}")
                if show_full:
                    st.write(art["text"])
                else:
                    st.write(format_snippet(art["text"], snippet_len))
