import json
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Embedding Lab", layout="wide")

# ========= MODELES =========
MODEL_NAMES = {
    "MiniLM (rapide)": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet (plus précis)": "sentence-transformers/all-mpnet-base-v2",
    "E5 Large Multilingual": "intfloat/multilingual-e5-large",
    "Solon Large (SOLON-embeddings-large-0.1)": "OrdalieTech/SOLON-embeddings-large-0.1",
    "Solon base (SOLON-embeddings-base-0.1)": "OrdalieTech/Solon-embeddings-base-0.1",
    "Camembert" : "dangvantuan/sentence-camembert-base"
}

# ========= CHARGEMENTS =========
@st.cache_data
def load_definitions(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cleaned = []
    for x in data:
        term = (x.get("term") or "").strip()
        definition = (x.get("definition") or "").strip()
        if term and definition:
            cleaned.append({"term": term, "definition": definition})
    return cleaned

@st.cache_resource
def load_model(model_id: str):
    # trust_remote_code=True utile pour certains modèles HF custom
    # device="cpu" évite des soucis GPU/FP16 (NaN) chez certains
    return SentenceTransformer(model_id, trust_remote_code=True, device="cpu")

# ========= EMBEDDINGS (robuste) =========
def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def embed_texts(model, texts):
    safe = []
    for t in texts:
        t = "" if t is None else str(t)
        t = t.strip()
        if not t:
            t = " "
        safe.append(t)

    vecs = model.encode(safe, normalize_embeddings=False)
    vecs = np.asarray(vecs, dtype=np.float32)

    # filtre lignes invalides (NaN/Inf) avant normalisation
    mask = np.isfinite(vecs).all(axis=1)
    vecs = vecs[mask]

    if vecs.shape[0] == 0:
        return vecs, mask

    vecs = l2_normalize(vecs)

    # recheck
    mask2 = np.isfinite(vecs).all(axis=1)
    vecs = vecs[mask2]

    # mask final (sur les textes initiaux)
    final_mask = np.zeros(len(safe), dtype=bool)
    valid_idx = np.where(mask)[0]
    if len(valid_idx) > 0:
        final_mask[valid_idx[mask2]] = True

    return vecs, final_mask

def embed_query(model, query: str, model_id: str):
    q = (query or "").strip()
    if not q:
        q = " "

    # Pour E5, les préfixes améliorent la qualité
    if "multilingual-e5" in model_id:
        q = "query: " + q

    vec = model.encode([q], normalize_embeddings=False)
    vec = np.asarray(vec, dtype=np.float32)

    if not np.isfinite(vec).all():
        return None

    vec = l2_normalize(vec)[0]
    if not np.isfinite(vec).all():
        return None
    return vec

def top_k_similar(query_vec, doc_vecs, k=5):
    sims = cosine_similarity([query_vec], doc_vecs)[0]
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]

# ========= UI =========
st.title("Recherche de définitions (Embeddings)")

defs = load_definitions("data/definitions.json")
all_terms = sorted({d["term"] for d in defs})

with st.sidebar:
    st.header("Filtre (optionnel)")
    selected_terms = st.multiselect(
        "Limiter la recherche à certains termes",
        options=all_terms,
        default=[],
    )
    show_terms = st.checkbox("Afficher la liste des termes", value=False)
    if show_terms:
        st.write(all_terms)

# on filtre les définitions si l’utilisateur choisit des termes
if selected_terms:
    defs_to_use = [d for d in defs if d["term"] in set(selected_terms)]
else:
    defs_to_use = defs

# IMPORTANT: tu veux indexer uniquement LA DEFINITION
docs = [d["definition"] for d in defs_to_use]

query = st.text_input("Requête", value="définition contrat commutatif")
k = st.slider("Top K", 1, 20, 5)

selected_models = st.multiselect(
    "Modèles à comparer",
    options=list(MODEL_NAMES.keys()),
    default=["MiniLM (rapide)", "E5 Large Multilingual"],
)

if st.button("Rechercher", type="primary"):
    if not query.strip():
        st.error("Entre une requête.")
        st.stop()
    if not selected_models:
        st.error("Sélectionne au moins un modèle.")
        st.stop()
    if len(docs) == 0:
        st.error("Aucune définition à chercher (filtre trop restrictif).")
        st.stop()

    cols = st.columns(len(selected_models))

    for col, label in zip(cols, selected_models):
        model_id = MODEL_NAMES[label]
        with col:
            st.subheader(label)
            st.caption(model_id)

            try:
                with st.spinner("Encodage..."):
                    model = load_model(model_id)

                    # E5: préfixe "passage:" conseillé
                    docs_for_model = docs
                    if "multilingual-e5" in model_id:
                        docs_for_model = ["passage: " + x for x in docs]

                    doc_vecs, mask = embed_texts(model, docs_for_model)
                    q_vec = embed_query(model, query, model_id)

                if q_vec is None:
                    st.error("Embedding requête invalide (NaN/Inf).")
                    continue

                if doc_vecs.shape[0] == 0:
                    st.error("Tous les embeddings documents sont invalides (NaN/Inf).")
                    continue

                # mapping des indices valides vers defs_to_use
                valid_doc_indices = np.where(mask)[0]

                results = top_k_similar(q_vec, doc_vecs, k=min(k, doc_vecs.shape[0]))

                st.write("Requête :")
                st.code(query)

                st.write("Résultats :")
                for rank, (i, score) in enumerate(results, start=1):
                    original_idx = int(valid_doc_indices[i])
                    item = defs_to_use[original_idx]

                    st.markdown(f"**{rank}. {item['term']}** — score={score:.4f}")
                    st.write(item["definition"])
                    st.divider()

            except Exception as e:
                st.error(f"Erreur modèle: {e}")
