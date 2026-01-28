import os
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import hnswlib

PDF_PATH = "Kodi_Civill-2014_i_azhornuar-1.pdf"


def clean_text(t: str) -> str:
    t = (t or "").replace("\u00ad", "")
    t = re.sub(r"-\n", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


@st.cache_data(show_spinner=False)
def load_articles(pdf_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    pages_text: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(clean_text(page.extract_text() or ""))

    full_text = "\n".join(pages_text)

    parts = re.split(r"\bNeni\s+(\d+)\b", full_text, flags=re.IGNORECASE)
    articles: List[Dict[str, str]] = []
    for i in range(1, len(parts), 2):
        num = parts[i].strip()
        body = clean_text(parts[i + 1])
        if not body:
            continue
        title = body.split("\n", 1)[0].strip()
        articles.append(
            {
                "num": num,
                "label": f"Neni {num}",
                "title": title,
                "text": body,
            }
        )
    return articles


@st.cache_resource(show_spinner=False)
def build_index(articles: List[Dict[str, str]]):
    corpus = [a["text"] for a in articles]
    if not corpus:
        raise ValueError("No articles found in PDF.")

    min_df = 2 if len(corpus) >= 10 else 1
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df)
    tfidf = vectorizer.fit_transform(corpus)

    if min(tfidf.shape) <= 2:
        svd = None
        dense = tfidf.toarray()
    else:
        n_components = min(256, max(2, min(tfidf.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = svd.fit_transform(tfidf)

    dense = normalize(dense)
    dim = dense.shape[1]

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(corpus), ef_construction=200, M=16)
    index.add_items(dense, np.arange(len(corpus)))
    index.set_ef(50)

    return vectorizer, svd, index, dense


def parse_article_query(q: str) -> Optional[str]:
    q = q.strip()
    if not q:
        return None
    m = re.search(r"\bNeni\s+(\d+)\b", q, re.IGNORECASE)
    if m:
        return m.group(1)
    if q.isdigit():
        return q
    return None


def embed_query(
    q: str,
    vectorizer: TfidfVectorizer,
    svd: Optional[TruncatedSVD],
) -> np.ndarray:
    tfidf = vectorizer.transform([q])
    if svd is None:
        dense = tfidf.toarray()
    else:
        dense = svd.transform(tfidf)
    return normalize(dense)


def search_articles(
    q: str,
    k: int,
    vectorizer: TfidfVectorizer,
    svd: Optional[TruncatedSVD],
    index: hnswlib.Index,
) -> List[Tuple[int, float]]:
    q_vec = embed_query(q, vectorizer, svd)
    labels, distances = index.knn_query(q_vec, k=k)
    results = []
    for label, dist in zip(labels[0], distances[0]):
        score = 1.0 - float(dist)
        results.append((int(label), score))
    return results


def format_snippet(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


st.set_page_config(page_title="Asistent Kodi Civil", layout="wide")
st.title("Asistent Kodi Civil")
st.caption("Kerkim semantik dhe gjetje e shpejte e neneve nga PDF.")

with st.sidebar:
    st.subheader("Opsione")
    pdf_path = st.text_input("PDF path", PDF_PATH)
    top_k = st.slider("Rezultate", 1, 10, 5)
    snippet_len = st.slider("Gjatesia e fragmentit", 200, 1200, 600)
    show_full = st.checkbox("Shfaq tekstin e plote", False)
    st.markdown("Keshille: shkruaj `Neni 5` ose vetem `5` per te hapur nenin direkt.")

query = st.text_input("Pyetje ose kerkese", placeholder="p.sh. Pronesia, detyrimet, kontrata...")

if query:
    try:
        articles = load_articles(pdf_path)
        vectorizer, svd, index, _ = build_index(articles)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    wanted_num = parse_article_query(query)
    if wanted_num:
        match = next((a for a in articles if a["num"] == wanted_num), None)
        if not match:
            st.warning(f"Neni {wanted_num} nuk u gjet.")
        else:
            st.subheader(f'{match["label"]} - {match["title"]}')
            st.write(match["text"])
        st.stop()

    k = min(top_k, len(articles))
    results = search_articles(query, k, vectorizer, svd, index)

    for idx, score in results:
        art = articles[idx]
        st.markdown(f'### {art["label"]} - {art["title"]}')
        st.caption(f"Relevance: {score:.3f}")
        if show_full:
            st.write(art["text"])
        else:
            st.write(format_snippet(art["text"], snippet_len))
