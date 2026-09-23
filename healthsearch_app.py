import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


st.set_page_config(page_title="HealthSearch", page_icon="🩺", layout="wide")

CORPUS = [
    {"id": "Doc 1", "titulo": "Protocolo Emergência ECG", "texto": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."},
    {"id": "Doc 2", "titulo": "Guia de Farmacologia Cardíaca", "texto": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio."},
    {"id": "Doc 3", "titulo": "Diretriz de Hipertensão Arterial", "texto": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI."},
    {"id": "Doc 4", "titulo": "Manual de AVC Isquêmico", "texto": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas."},
    {"id": "Doc 5", "titulo": "Protocolo de Reanimação RCR", "texto": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul."},
    {"id": "Doc 6", "titulo": "Procedimentos de UTI Geral", "texto": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria."},
]

STOPWORDS = {"a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na", "para", "com", "por", "um", "uma", "que", "se", "ao", "aos", "às", "ou", "até", "dos", "das"}


def normalizar(texto):
    texto = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", texto)


def preprocessar(texto):
    return [token for token in normalizar(texto) if token not in STOPWORDS]


def similaridade_cosseno(vetor_a, vetor_b):
    denominador = np.linalg.norm(vetor_a) * np.linalg.norm(vetor_b)
    return float(np.dot(vetor_a, vetor_b) / denominador) if denominador else 0.0


@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_modelo():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_data(show_spinner=False)
def gerar_embeddings(textos):
    return carregar_modelo().encode(list(textos), normalize_embeddings=True)


def montar_ranking(query, k1, b, alpha):
    tokens_corpus = [preprocessar(item["texto"]) for item in CORPUS]
    bm25 = BM25Okapi(tokens_corpus, k1=k1, b=b)
    score_bm25 = bm25.get_scores(preprocessar(query))

    textos = tuple(item["texto"] for item in CORPUS)
    embeddings_docs = gerar_embeddings(textos)
    embedding_query = gerar_embeddings((query,))[0]
    score_semantico = np.array([similaridade_cosseno(embedding_query, vetor) for vetor in embeddings_docs])

    ordem_bm25 = np.argsort(-score_bm25)
    ordem_semantica = np.argsort(-score_semantico)
    rank_bm25 = {indice: posicao + 1 for posicao, indice in enumerate(ordem_bm25)}
    rank_semantico = {indice: posicao + 1 for posicao, indice in enumerate(ordem_semantica)}
    score_rrf = np.array([
        alpha * (1 / (60 + rank_bm25[indice])) + (1 - alpha) * (1 / (60 + rank_semantico[indice]))
        for indice in range(len(CORPUS))
    ])

    resultado = pd.DataFrame(CORPUS)
    resultado["Score BM25"] = score_bm25
    resultado["Rank BM25"] = [rank_bm25[indice] for indice in range(len(CORPUS))]
    resultado["Score Semântico"] = score_semantico
    resultado["Rank Semântico"] = [rank_semantico[indice] for indice in range(len(CORPUS))]
    resultado["Score RRF"] = score_rrf
    resultado["Rank RRF"] = resultado["Score RRF"].rank(ascending=False, method="min").astype(int)
    return resultado


@st.cache_resource(show_spinner="Carregando Cross-Encoder...")
def carregar_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


st.title("🩺 HealthSearch — Busca Híbrida BM25 e Semântica")
st.caption("Demonstração de precisão léxica para códigos médicos e cobertura semântica para sinônimos clínicos.")

with st.sidebar:
    st.header("⚙️ Calibração")
    k1 = st.slider("k1 — saturação de frequência", 0.0, 3.0, 1.2, 0.1)
    b = st.slider("b — normalização por tamanho", 0.0, 1.0, 0.75, 0.05)
    alpha = st.slider("α — peso do BM25 no RRF", 0.0, 1.0, 0.5, 0.05)
    st.caption("A constante do RRF é k = 60, conforme o enunciado.")

query = st.text_input("Consulta médica", placeholder="Ex.: ataque cardíaco, CÓD-ECG-12D, AAS 100mg")
usar_reranking = st.checkbox("Bônus: aplicar Cross-Encoder nos Top-3 do ranking híbrido")
st.dataframe(pd.DataFrame(CORPUS)[["id", "titulo", "texto"]], width="stretch", hide_index=True)

if query:
    try:
        ranking = montar_ranking(query, k1, b, alpha)
        tab_bm25, tab_semantico, tab_rrf, tab_comparativa = st.tabs(["🔤 Léxico BM25", "🧠 Semântico", "🔀 Híbrido RRF", "📊 Matriz Comparativa"])

        with tab_bm25:
            st.write("BM25 privilegia correspondências exatas, como códigos e dosagens.")
            st.dataframe(ranking.sort_values("Score BM25", ascending=False)[["id", "titulo", "Score BM25", "Rank BM25", "texto"]], width="stretch", hide_index=True)
        with tab_semantico:
            st.write("Embeddings aproximam expressões equivalentes, como “ataque cardíaco” e “infarto”.")
            st.dataframe(ranking.sort_values("Score Semântico", ascending=False)[["id", "titulo", "Score Semântico", "Rank Semântico", "texto"]], width="stretch", hide_index=True)
        with tab_rrf:
            final = ranking.sort_values("Score RRF", ascending=False).copy()
            if usar_reranking:
                candidatos = final.head(3).copy()
                cross_encoder = carregar_cross_encoder()
                candidatos["Score Cross-Encoder"] = cross_encoder.predict([[query, texto] for texto in candidatos["texto"]])
                candidatos = candidatos.sort_values("Score Cross-Encoder", ascending=False)
                final = pd.concat([candidatos, final.iloc[3:]], ignore_index=True)
                final["Rank Cross-Encoder"] = range(1, len(final) + 1)
                st.info("O Cross-Encoder reordenou os três melhores candidatos recuperados pelo RRF.")
            melhor = final.iloc[0]
            st.success(f"Melhor resultado híbrido: {melhor['id']} — {melhor['titulo']}")
            st.dataframe(final[["id", "titulo", "Score RRF", "Rank RRF", "texto"]], width="stretch", hide_index=True)
            st.latex(r"Score_{RRF}(D) = \alpha \frac{1}{60 + Rank_{BM25}} + (1-\alpha) \frac{1}{60 + Rank_{Semantico}}")
        with tab_comparativa:
            st.dataframe(ranking.sort_values("Rank RRF")[["id", "titulo", "Rank BM25", "Rank Semântico", "Rank RRF", "Score BM25", "Score Semântico", "Score RRF"]], width="stretch", hide_index=True)
    except Exception as erro:
        st.error(f"Não foi possível carregar o modelo semântico: {erro}")
        st.info("Na primeira execução, o sentence-transformers pode precisar baixar o modelo multilíngue.")
else:
    st.info("Digite uma consulta para comparar os rankings léxico, semântico e híbrido.")
