import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Embedding Lab", layout="wide")

# 1) Un mini "corpus" local pour simuler une base de données (tu remplaceras par la tienne)
CORPUS = [
    "Streamlit permet de créer une interface web en Python.",
    "Les embeddings représentent du texte sous forme de vecteurs.",
    "FAISS est souvent utilisé pour faire de la recherche vectorielle.",
    "Le fine-tuning ajuste un modèle sur des données spécifiques.",
    "On peut comparer plusieurs modèles d'embeddings pour un même prompt.",
    "Une base vectorielle stocke des vecteurs et permet une recherche par similarité.",
    "Les modèles sentence-transformers servent à encoder des phrases.",
]

# 2) Liste de modèles publics à tester (tu peux en ajouter/enlever)
MODEL_NAMES = {
    "MiniLM (rapide, léger)": "sentence-transformers/all-MiniLM-L6-v2",
    "E5 Large Multilingual": "intfloat/multilingual-e5-large",
    "Solon Large (SOLON-embeddings-large-0.1)": "OrdalieTech/SOLON-embeddings-large-0.1",
    "Solon base (SOLON-embeddings-base-0.1)": "OrdalieTech/Solon-embeddings-base-0.1",
    "Camembert" : "dangvantuan/sentence-camembert-base"

}

@st.cache_resource
def load_model(model_id: str):
    return SentenceTransformer(model_id)

def embed_texts(model, texts):
    # normalize_embeddings=True aide pour la similarité cosinus
    return model.encode(texts, normalize_embeddings=True)

def top_k_similar(query_vec, corpus_vecs, k=5):
    sims = cosine_similarity([query_vec], corpus_vecs)[0]
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]

st.title("Embedding Lab — comparaison de modèles")
st.write("Pose une question, on calcule les embeddings avec plusieurs modèles, puis on affiche les résultats.")

# UI
col_left, col_right = st.columns([2, 1])

with col_left:
    query = st.text_input("Question / requête", value="Comment tester des embeddings sur plusieurs modèles ?")
    k = st.slider("Top K résultats", min_value=1, max_value=10, value=5)

with col_right:
    selected = st.multiselect(
        "Modèles à comparer",
        options=list(MODEL_NAMES.keys()),
        default=list(MODEL_NAMES.keys())[:2],
    )

st.divider()

if st.button("Lancer le test", type="primary"):
    if not selected:
        st.error("Sélectionne au moins un modèle.")
        st.stop()

    # Affichage en colonnes, un bloc par modèle
    cols = st.columns(len(selected))

    for col, label in zip(cols, selected):
        model_id = MODEL_NAMES[label]
        with col:
            st.subheader(label)
            st.caption(model_id)

            with st.spinner("Chargement + encodage..."):
                model = load_model(model_id)

                corpus_vecs = embed_texts(model, CORPUS)
                q_vec = embed_texts(model, [query])[0]
                st.write("q_vec finite:", bool(np.isfinite(q_vec).all()))
                st.write("corpus_vecs finite:", bool(np.isfinite(corpus_vecs).all()))
                st.write("corpus_vecs shape:", getattr(corpus_vecs, "shape", None))

                if not np.isfinite(q_vec).all():
                    st.error("NaN/Inf dans l'embedding de la requête.")
                    st.stop()

                bad = np.where(~np.isfinite(corpus_vecs).all(axis=1))[0]
                if len(bad) > 0:
                    st.error(f"NaN/Inf dans {len(bad)} embeddings du corpus. Ex: {bad[:10].tolist()}")
                    st.stop()
                results = top_k_similar(q_vec, corpus_vecs, k=k)

            st.write("Requête :")
            st.code(query)

            st.write("Top résultats (corpus local) :")
            for rank, (i, score) in enumerate(results, start=1):
                st.markdown(f"**{rank}.** score={score:.4f}")
                st.write(CORPUS[i])
                st.write("---")

    st.success("Test terminé.")
