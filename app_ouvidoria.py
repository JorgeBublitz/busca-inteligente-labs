import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="📬", layout="wide")
ARQUIVO_DADOS = Path(__file__).with_name("manifestacoes.json")
MODELOS = {
    "Multilíngue MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
    "BGE Multilíngue": "BAAI/bge-m3",
}
# M004 x M036 (não M004 x M025): ambas tratam da mesma praça central com iluminação apagada
# no trecho entre o ponto de ônibus e a via, gerando insegurança para quem passa à noite.
# M025 fala de um ponto de ônibus em outra rua (Rua Verde) e não é o mesmo caso.
DUPLICATAS_REAIS = {
    frozenset(("M001", "M003")), frozenset(("M003", "M017")), frozenset(("M003", "M034")),
    frozenset(("M008", "M022")), frozenset(("M008", "M035")), frozenset(("M004", "M036")),
}


@st.cache_data
def carregar_dados():
    try:
        registros = json.loads(ARQUIVO_DADOS.read_text(encoding="utf-8"))
        dados = pd.DataFrame(registros)
        colunas_obrigatorias = {"id", "texto", "categoria_oficial"}
        faltantes = colunas_obrigatorias - set(dados.columns)
        if faltantes:
            raise ValueError(f"Campos obrigatórios ausentes no JSON: {', '.join(sorted(faltantes))}")
        return dados
    except (json.JSONDecodeError, ValueError, KeyError) as erro:
        st.error(f"Não foi possível carregar '{ARQUIVO_DADOS.name}': {erro}")
        st.stop()


@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_modelo(nome):
    return SentenceTransformer(MODELOS[nome])


@st.cache_data(show_spinner=False)
def gerar_embeddings(textos, nome_modelo):
    return carregar_modelo(nome_modelo).encode(list(textos), normalize_embeddings=True)


def reduzir_2d(vetores, metodo):
    if metodo == "PCA":
        return PCA(n_components=2, random_state=42).fit_transform(vetores)
    perplexidade = min(15, max(2, len(vetores) // 3))
    return TSNE(n_components=2, perplexity=perplexidade, random_state=42, init="pca").fit_transform(vetores)


def chunks_de_texto(texto, tamanho, sobreposicao):
    splitter = RecursiveCharacterTextSplitter(chunk_size=tamanho, chunk_overlap=sobreposicao)
    return splitter.split_text(texto)


def detectar_duplicatas(ids, vetores, limiar=0.85):
    """Retorna matriz e pares semanticamente duplicados acima do limiar."""
    matriz = cosine_similarity(vetores)
    pares = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if matriz[i, j] >= limiar:
                pares.append({"origem": ids[i], "destino": ids[j], "similaridade": matriz[i, j]})
    return matriz, pares


st.title("📬 Ouvidoria Inteligente")
st.caption("Triagem semântica de manifestações cidadãs com embeddings, detecção de duplicatas e chunking.")
dados = carregar_dados()

with st.sidebar:
    st.header("⚙️ Configurações")
    nome_modelo = st.selectbox("Modelo de embedding", list(MODELOS))
    top_k = st.slider("Top-k de resultados", 1, 10, 5)
    limiar = st.slider("Limiar de duplicidade", 0.50, 0.99, 0.85, 0.01)

try:
    embeddings = gerar_embeddings(tuple(dados["texto"]), nome_modelo)
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo de embeddings: {erro}")
    st.stop()

tab_busca, tab_base, tab_espaco, tab_chunking, tab_analise = st.tabs([
    "🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking", "📈 Análises",
])

with tab_busca:
    consulta = st.text_area("Descreva o problema", placeholder="Ex.: rua com buraco perto da escola")
    if consulta:
        vetor_consulta = gerar_embeddings((consulta,), nome_modelo)[0]
        scores = cosine_similarity([vetor_consulta], embeddings)[0]
        indices = np.argsort(-scores)[:top_k]
        resultados = dados.iloc[indices][["id", "categoria_oficial", "texto"]].copy()
        resultados["score"] = scores[indices]
        resultados["destaque"] = np.where(resultados["score"] > 0.7, "🟢 alto", np.where(resultados["score"] > 0.5, "🟡 médio", "🔴 baixo"))
        st.dataframe(resultados, width="stretch", hide_index=True, column_config={"score": st.column_config.NumberColumn(format="%.3f")})
    else:
        st.info("Digite uma manifestação para receber os resultados semanticamente mais próximos.")

with tab_base:
    st.dataframe(dados, width="stretch", hide_index=True)
    if st.button("Gerar matriz de similaridade", key="matriz"):
        matriz = cosine_similarity(embeddings)
        figura = px.imshow(matriz, color_continuous_scale="Blues", labels={"color": "similaridade", "x": "Manifestação", "y": "Manifestação"})
        st.plotly_chart(figura, width="stretch")

with tab_espaco:
    metodo = st.radio("Redução de dimensionalidade", ["PCA", "t-SNE"], horizontal=True)
    if metodo == "t-SNE" and len(dados) <= 3:
        st.warning("São necessários mais de 3 registros para calcular o t-SNE. Use PCA ou adicione mais dados.")
    else:
        coordenadas = reduzir_2d(embeddings, metodo)
        visual = dados[["id", "categoria_oficial", "texto"]].copy()
        visual["x"], visual["y"] = coordenadas[:, 0], coordenadas[:, 1]
        figura = px.scatter(visual, x="x", y="y", color="categoria_oficial", hover_data=["id", "texto"], title=f"Espaço semântico em 2D — {metodo}")
        st.plotly_chart(figura, width="stretch")
        st.caption("Compare visualmente os agrupamentos semânticos com as categorias oficiais do conjunto de dados.")

with tab_chunking:
    exemplo_longo = dados.assign(_tamanho=dados["texto"].str.len()).sort_values("_tamanho", ascending=False).iloc[0]["texto"]
    texto_longo = st.text_area("Manifestação longa", value=exemplo_longo, height=180)
    col_a, col_b = st.columns(2)
    tamanho = col_a.slider("chunk_size", 100, 1000, 350, 50)
    sobreposicao = col_b.slider("chunk_overlap", 0, 300, 60, 10)
    chunks = chunks_de_texto(texto_longo, tamanho, sobreposicao)
    st.write(f"Foram gerados **{len(chunks)} chunks**.")
    st.dataframe(pd.DataFrame({"chunk": range(1, len(chunks) + 1), "texto": chunks}), width="stretch", hide_index=True)
    if len(chunks) > 2:
        vetores_chunks = gerar_embeddings(tuple(chunks), nome_modelo)
        coords = reduzir_2d(vetores_chunks, "PCA")
        figura = px.scatter(x=coords[:, 0], y=coords[:, 1], text=[f"Chunk {i + 1}" for i in range(len(chunks))], title="Chunks no espaço semântico")
        st.plotly_chart(figura, width="stretch")

with tab_analise:
    # Este bloco espelha intencionalmente a análise feita em analise_comparativa.ipynb e
    # deteccao_duplicatas.ipynb: os notebooks reproduzem os mesmos cálculos de forma
    # standalone (sem depender do app) para fins didáticos, o que gera duplicação
    # aceitável neste repositório de aprendizado.
    st.subheader("Comparação BoW, TF-IDF e Embeddings")
    # M003 x M034 é um par genuinamente semelhante: M034 cita explicitamente "o buraco
    # da Avenida Brasil" (a manifestação M003), então ambos descrevem o mesmo problema.
    pares = [("M003", "M017"), ("M008", "M022"), ("M003", "M034")]
    textos = dados["texto"].tolist()
    bow = CountVectorizer().fit_transform(textos)
    tfidf = TfidfVectorizer().fit_transform(textos)
    linhas = []
    for origem, destino in pares:
        i = dados.index[dados["id"] == origem][0]
        j = dados.index[dados["id"] == destino][0]
        linhas.append({
            "par": f"{origem} × {destino}",
            "BoW": cosine_similarity(bow[i], bow[j])[0, 0],
            "TF-IDF": cosine_similarity(tfidf[i], tfidf[j])[0, 0],
            "Embeddings": cosine_similarity([embeddings[i]], [embeddings[j]])[0, 0],
        })
    st.dataframe(pd.DataFrame(linhas), width="stretch", hide_index=True, column_config={campo: st.column_config.NumberColumn(format="%.3f") for campo in ["BoW", "TF-IDF", "Embeddings"]})

    st.subheader("Detecção de duplicatas")
    matriz, pares_duplicados = detectar_duplicatas(dados["id"].tolist(), embeddings, limiar)
    st.write(f"Pares encontrados com similaridade ≥ {limiar:.2f}: **{len(pares_duplicados)}**")
    st.dataframe(pd.DataFrame(pares_duplicados), width="stretch", hide_index=True, column_config={"similaridade": st.column_config.NumberColumn(format="%.3f")})
    pares_detectados = {frozenset((item["origem"], item["destino"])) for item in pares_duplicados}
    falsos_positivos = pares_detectados - DUPLICATAS_REAIS
    falsos_negativos = DUPLICATAS_REAIS - pares_detectados
    st.write(f"Falsos positivos: **{len(falsos_positivos)}** | Falsos negativos: **{len(falsos_negativos)}**")
    st.caption("Ajuste o limiar para observar o equilíbrio entre recuperar duplicatas reais e evitar agrupamentos indevidos.")
