import re
import numpy as np
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

PDF_PATH = "Kodi_Civill-2014_i_azhornuar-1.pdf"

def clean_text(t: str) -> str:
    t = (t or "").replace("\u00ad", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

@st.cache_data(show_spinner=False)
def load_nenet(pdf_path: str):
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(clean_text(page.extract_text() or ""))

    full_text = "\n".join(pages_text)
    pattern = re.compile(r"\bNeni\s+(\d+)\b", re.IGNORECASE)
    matches = list(pattern.finditer(full_text))

    nenet = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        nenet.append({
            "neni": int(m.group(1)),
            "text": full_text[start:end].strip()
        })
    return nenet

@st.cache_resource(show_spinner=False)
def build_retriever(nenet):
    texts = [n["text"] for n in nenet]
    vectorizer = TfidfVectorizer(lowercase=True, max_features=20000, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=300, random_state=42)
    X_svd = svd.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(X_svd)

    return vectorizer, svd, nn

def make_answer(query, nenet, vectorizer, svd, nn):
    q = vectorizer.transform([query])
    q_svd = svd.transform(q)

    distances, indices = nn.kneighbors(q_svd)

    best_idx = indices[0][0]
    best_txt = nenet[best_idx]["text"]

    sentences = re.split(r"(?<=[\.\?\!])\s+|\n+", best_txt)
    summary = " ".join(sentences[:3])

    cites = []
    for idx in indices[0]:
        txt = nenet[idx]["text"].replace("\n"," ")
        if len(txt) > 200:
            txt = txt[:200] + "..."
        cites.append(f"- (Neni {nenet[idx]['neni']}) {txt}")

    return summary + "\n\nNenet referente:\n" + "\n".join(cites)

st.title("Asistent i Kodit Civil (Shqip)")
st.caption("Kërkim inteligjent mbi nenet e Kodit Civil")

nenet = load_nenet(PDF_PATH)
vectorizer, svd, nn = build_retriever(nenet)

q = st.text_input("Shkruaj pyetjen:")
if q:
    answer = make_answer(q, nenet, vectorizer, svd, nn)
    st.write(answer)



