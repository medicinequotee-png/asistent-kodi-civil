import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np
import pdfplumber
import requests
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DEFAULT_PDF_PATH = "Kodi_Civill-2014_i_azhornuar-1.pdf"
DEFAULT_TOP_K = 5
DEFAULT_SNIPPET_LEN = 700
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_EMBED_MODEL = os.getenv("LLM_EMBED_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "gpt-4o-mini")
MAX_CONTEXT_CHARS = 6000


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


@st.cache_data(show_spinner=False)
def load_articles(pdf_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Missing PDF: {pdf_path}")

    pages_text: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(clean_text(page.extract_text() or ""))

    full_text = "\n".join(pages_text)
    return split_articles(full_text)


def download_pdf(url: str, dest_path: str) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def get_pdf_signature(pdf_path: str) -> str:
    stat = os.stat(pdf_path)
    return f"{stat.st_size}_{int(stat.st_mtime)}"


def cache_dir() -> Path:
    path = Path(".cache")
    path.mkdir(exist_ok=True)
    return path


def safe_slug(value: str) -> str:
    value = value or "default"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value.strip("_").lower() or "default"


def normalize_base_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def get_api_key() -> Optional[str]:
    return os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")


def embed_texts_llm(
    texts: List[str],
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int = 64,
) -> np.ndarray:
    url = normalize_base_url(base_url) + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    embeddings: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = {"model": model, "input": batch}
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        data = sorted(data, key=lambda item: item.get("index", 0))
        embeddings.extend([item["embedding"] for item in data])

    return np.asarray(embeddings, dtype=np.float32)


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
def build_indices(
    articles: List[Dict[str, str]],
    use_llm_embeddings: bool,
    llm_key_present: bool,
    embed_model: str,
    embed_base_url: str,
    pdf_sig: str,
) -> Dict[str, object]:
    corpus = [a["text"] for a in articles]
    if not corpus:
        raise ValueError("No articles found in PDF.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf = vectorizer.fit_transform(corpus)
    tfidf_norm = normalize(tfidf)

    api_key = get_api_key() if use_llm_embeddings and llm_key_present else None
    embeddings = None
    svd = None

    if api_key:
        cache_key = f"{pdf_sig}_{safe_slug(embed_model)}"
        emb_path = cache_dir() / f"embeddings_{cache_key}.npy"
        if emb_path.exists():
            embeddings = np.load(emb_path)
        else:
            embeddings = embed_texts_llm(corpus, api_key, embed_base_url, embed_model)
            np.save(emb_path, embeddings)
    else:
        embeddings, svd = build_dense_from_tfidf(tfidf)

    ann_index = build_ann_index(embeddings)

    return {
        "vectorizer": vectorizer,
        "tfidf_norm": tfidf_norm,
        "svd": svd,
        "embeddings": embeddings,
        "ann_index": ann_index,
        "llm_embeddings": bool(api_key),
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


def embed_query(
    query: str,
    vectorizer: TfidfVectorizer,
    svd: Optional[TruncatedSVD],
    llm_enabled: bool,
    embed_model: str,
    embed_base_url: str,
) -> np.ndarray:
    if llm_enabled:
        api_key = get_api_key()
        if not api_key:
            raise ValueError("LLM API key missing.")
        vec = embed_texts_llm([query], api_key, embed_base_url, embed_model)
        return normalize(vec)

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
    llm_enabled: bool,
    embed_model: str,
    embed_base_url: str,
    alpha: float = 0.65,
) -> List[Tuple[int, float]]:
    tfidf_query = vectorizer.transform([query])
    tfidf_query = normalize(tfidf_query)
    lexical_scores = (tfidf_norm @ tfidf_query.T).toarray().ravel()

    q_embed = embed_query(
        query,
        vectorizer,
        svd,
        llm_enabled,
        embed_model,
        embed_base_url,
    )
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


def build_context(articles: List[Dict[str, str]], results: List[Tuple[int, float]]) -> Tuple[str, List[str]]:
    blocks: List[str] = []
    citations: List[str] = []
    total = 0
    for idx, _score in results:
        art = articles[idx]
        block = f"[{art['label']}] {art['text']}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        citations.append(art["label"])
        total += len(block)
    return "\n\n".join(blocks), citations


def call_chat_llm(messages: List[Dict[str, str]], api_key: str, base_url: str, model: str) -> str:
    url = normalize_base_url(base_url) + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No response from LLM.")
    return choices[0]["message"]["content"].strip()


def fallback_answer(results: List[Tuple[int, float]], articles: List[Dict[str, str]]) -> str:
    if not results:
        return "Nuk gjeta nene te pershtatshme."
    top = results[0][0]
    art = articles[top]
    return (
        f"Nuk ka LLM te konfiguruar. Neni me relevant duket: {art['label']} - {art['title']}."
    )


st.set_page_config(page_title="Asistent Kodi Civil", layout="wide")
st.title("Asistent Kodi Civil")
st.caption("Bashkebisedim ne shqip dhe kerkime mbi Kodin Civil.")

with st.sidebar:
    st.subheader("Opsione")
    pdf_path = st.text_input("PDF path", DEFAULT_PDF_PATH)
    pdf_url = st.text_input("PDF source URL (optional)", os.getenv("PDF_SOURCE_URL", ""))

    if st.button("Shkarko / Perditeso PDF") and pdf_url:
        try:
            download_pdf(pdf_url, pdf_path)
            st.success("PDF u perditesua.")
        except Exception as exc:
            st.error(f"Deshtoi shkarkimi: {exc}")

    top_k = st.slider("Rezultate", 1, 10, DEFAULT_TOP_K)
    snippet_len = st.slider("Gjatesia e fragmentit", 200, 1400, DEFAULT_SNIPPET_LEN)
    show_full = st.checkbox("Shfaq tekstin e plote", False)
    use_llm = st.checkbox("Perdor LLM per pergjigje", True)
    alpha = st.slider("Pesha semantike", 0.0, 1.0, 0.65)

    with st.expander("LLM settings"):
        api_key_input = st.text_input("LLM API key", value="", type="password")
        base_url = st.text_input("LLM base URL", DEFAULT_LLM_BASE_URL)
        embed_model = st.text_input("Embedding model", DEFAULT_EMBED_MODEL)
        chat_model = st.text_input("Chat model", DEFAULT_CHAT_MODEL)

    st.markdown("Keshille: shkruaj `Neni 5` ose vetem `5` per ta hapur direkt.")

if api_key_input:
    os.environ["LLM_API_KEY"] = api_key_input

try:
    articles = load_articles(pdf_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()

pdf_sig = get_pdf_signature(pdf_path)
indices = build_indices(
    articles,
    use_llm_embeddings=use_llm,
    llm_key_present=bool(get_api_key()),
    embed_model=embed_model,
    embed_base_url=base_url,
    pdf_sig=pdf_sig,
)

llm_ready = bool(get_api_key()) and use_llm
if use_llm and not llm_ready:
    st.warning("LLM nuk eshte aktiv. Vendos API key ose fik opsionin.")

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
        indices["llm_embeddings"],
        embed_model,
        base_url,
        alpha=alpha,
    )

    context_text, citations = build_context(articles, results)

    if llm_ready:
        system_prompt = (
            "Ti je asistent juridik per Kodin Civil te Republikes se Shqiperise. "
            "Pergjigju vetem me baze ne tekstin e dhene. Pergjigju ne shqip. "
            "Nese nuk ka baze te mjaftueshme, thuaj qe nuk ka informacion."
        )
        user_prompt = (
            f"Pyetja: {user_query}\n\n"
            f"Teksti i kodit civil (me citime):\n{context_text}\n\n"
            "Pergjigju shkurt dhe qartazi. Ne fund, jep citimet si [Neni X]."
        )
        try:
            answer = call_chat_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=get_api_key(),
                base_url=base_url,
                model=chat_model,
            )
        except Exception as exc:
            answer = f"LLM error: {exc}"
    else:
        answer = fallback_answer(results, articles)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
        if citations:
            st.caption("Citime: " + ", ".join(citations))

        with st.expander("Nenet relevante"):
            for idx, score in results:
                art = articles[idx]
                st.markdown(f"**{art['label']} - {art['title']}**")
                st.caption(f"Relevance: {score:.3f}")
                if show_full:
                    st.write(art["text"])
                else:
                    st.write(format_snippet(art["text"], snippet_len))
