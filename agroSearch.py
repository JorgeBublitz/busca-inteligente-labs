import streamlit as st
import PyPDF2
import re
import unicodedata
from collections import defaultdict, Counter
import math
import pandas as pd

st.set_page_config(page_title="AgroSearch", page_icon="🚜", layout="wide")
st.title("🚜 AgroSearch: Motor de Busca Inteligente")

# 1. Componente para o usuário fazer upload dos PDFs direto na tela
st.subheader("Base de Manuais (Upload)")
arquivos_pdf = st.file_uploader(
    "Faça o upload dos manuais em PDF aqui", 
    type=["pdf"], 
    accept_multiple_files=True
)

st.divider()
st.header("Fase 1: 🧹 Pipeline de Pré-processamento de Texto")

# 2. Dicionário manual de Stopwords
STOPWORDS = {
    # Artigos, Preposições e Contrações
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas", "num", "numa",
    "por", "pelo", "pela", "pelos", "pelas", "para", "com", "sem", "sobre", "sob", 
    "até", "entre", "através", "durante",
    
    # Conjunções
    "e", "ou", "mas", "porém", "contudo", "todavia", "portanto", "pois", "porque", 
    "que", "se", "como", "logo",
    
    # Pronomes
    "seu", "sua", "seus", "suas", "este", "esta", "estes", "estas", "esse", "essa", 
    "esses", "essas", "isso", "aquilo", "ele", "ela", "eles", "elas", "qual", "quais", 
    "quem", "cada", "algum", "alguma", "qualquer", "aos", "às", "ao", "à",
    
    # Verbos Auxiliares e Modais
    "é", "são", "foi", "foram", "ser", "sendo", "sido", "está", "estão", "estar", 
    "esteve", "tem", "têm", "tinha", "ter", "tendo", "há", "houve", "pode", "podem", 
    "podendo", "deve", "devem",
    
    # Advérbios e Expressões de Intensidade/Tempo
    "não", "sim", "mais", "menos", "muito", "pouco", "apenas", "também", "já", 
    "ainda", "além", "mesmo", "mesma", "onde", "quando", "assim", "então"
}


def remover_acentos(texto):
    return "".join(
        caractere
        for caractere in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caractere) != "Mn"
    )


# O texto é normalizado antes da comparação; as stopwords precisam seguir a mesma regra.
STOPWORDS_NORMALIZADAS = {remover_acentos(palavra).lower() for palavra in STOPWORDS}

# 3. Funções do Pipeline
def extrair_texto_pdf(arquivo_upload):
    texto = ""
    # O PyPDF2 consegue ler diretamente o arquivo carregado no Streamlit
    leitor = PyPDF2.PdfReader(arquivo_upload)
    for pagina in leitor.pages:
        if pagina.extract_text():
            texto += pagina.extract_text() + " "
    return texto

def normalizar_e_tokenizar(texto):
    texto_sem_acento = remover_acentos(texto)
    texto_lower = texto_sem_acento.lower()
    tokens = re.findall(r'\b[a-z]+\b', texto_lower)
    return tokens

def aplicar_stemming_manual(token):
    sufixos = ['acao', 'coes', 'ando', 'endo', 'indo', 'mente', 'dade', 'iva', 'ivo']
    for sufixo in sufixos:
        if token.endswith(sufixo) and len(token) > len(sufixo) + 2:
            return token[:-len(sufixo)]
    if token.endswith('s') and len(token) > 3:
        return token[:-1]
    return token


def calcular_idf(N, documentos_com_termo):
    """IDF suavizado para evitar peso zero quando um termo aparece em todo o corpus."""
    return math.log10((N + 1) / (documentos_com_termo + 1)) + 1


def calcular_similaridade_cosseno(vetor_a, vetor_b):
    produto_escalar = sum(a * b for a, b in zip(vetor_a, vetor_b))
    norma_a = math.sqrt(sum(valor ** 2 for valor in vetor_a))
    norma_b = math.sqrt(sum(valor ** 2 for valor in vetor_b))
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return produto_escalar / (norma_a * norma_b)

# 4. Interface Interativa (Checkboxes)
col_a, col_b = st.columns(2)
usar_stopwords = col_a.checkbox("Remover Stopwords", value=False)
usar_stemming = col_b.checkbox("Aplicar Stemming (Redução ao radical)", value=False)
usar_cosseno = st.checkbox("Bônus: calcular Similaridade de Cosseno", value=False)

# 5. Processamento Dinâmico (Só roda se o usuário tiver feito upload)
if arquivos_pdf:
    documentos_processados = {}
    
    for arquivo in arquivos_pdf:
        # Extrai o texto do PDF
        texto_bruto = extrair_texto_pdf(arquivo)
        
        # Tokeniza o texto
        tokens = normalizar_e_tokenizar(texto_bruto)
        
        # Aplica Stopwords e Stemming conforme as regras da Fase 1
        if usar_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS_NORMALIZADAS]
            
        if usar_stemming:
            tokens = [aplicar_stemming_manual(t) for t in tokens]
            
        # Salva o resultado usando o nome do arquivo original como chave
        documentos_processados[arquivo.name] = tokens

    # Exibindo o resultado na tela para validação
    st.write("### Vocabulário Processado por Documento:")
    st.json(documentos_processados)

    st.divider()
    st.header("Fase 2: 🔗 Construtor de Índice Invertido")

    if st.button("Gerar Índice Invertido"):
        # O Índice Direto já é o próprio dicionário processado na Fase 1
        indice_direto = documentos_processados
        
        # Constrói o Índice Invertido (Termo -> Lista de Docs)
        indice_invertido = defaultdict(list)
        
        for nome_doc, termos in indice_direto.items():
            for termo in termos:
                # Adiciona o documento à lista do termo apenas se ele ainda não estiver lá
                if nome_doc not in indice_invertido[termo]:
                    indice_invertido[termo].append(nome_doc)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Índice Direto (Doc -> Termos)")
            st.json(indice_direto)
        
        with col2:
            st.subheader("Índice Invertido (Termo -> Docs)")
            st.json(dict(indice_invertido))

    st.divider()
    st.header("Fase 3: 📊 Busca e Ranqueamento (TF-IDF)")

    # 1. Barra de pesquisa para a Query
    query = st.text_input("Digite sua consulta (ex: controle de lagartas, agrotoxicos):")

    if st.button("Buscar"):
        if not query:
            st.warning("Por favor, digite uma consulta.")
        else:
            tokens_query = normalizar_e_tokenizar(query)
            if usar_stopwords:
                tokens_query = [t for t in tokens_query if t not in STOPWORDS_NORMALIZADAS]
            if usar_stemming:
                tokens_query = [aplicar_stemming_manual(t) for t in tokens_query]
            
            st.write(f"**Query processada:** {tokens_query}")
            
            N = len(documentos_processados) 
            scores = {doc: 0.0 for doc in documentos_processados.keys()}
            detalhes_calculo = []
            
            # 4. Cálculo do TF, IDF e TF-IDF "do zero"
            for termo in tokens_query:
                docs_com_termo = sum(1 for tokens in documentos_processados.values() if termo in tokens)
                
                if docs_com_termo > 0:
                    idf = calcular_idf(N, docs_com_termo)
                    
                    for doc, tokens_doc in documentos_processados.items():
                        frequencia_termo = tokens_doc.count(termo)
                        
                        if frequencia_termo > 0:
                            tf = frequencia_termo / len(tokens_doc)
                            
                            tf_idf = tf * idf
                            scores[doc] += tf_idf 
                            
                            detalhes_calculo.append({
                                "Documento": doc,
                                "Termo": termo,
                                "TF": round(tf, 4),
                                "IDF": round(idf, 4),
                                "TF-IDF": round(tf_idf, 4)
                            })
            
            # 5. Exibição dos Resultados Ranqueados
            if not detalhes_calculo:
                st.error("Nenhum documento relevante encontrado para esta busca.")
            else:
                st.subheader("🏆 Ranking de Relevância")
                # Ordena os documentos do maior para o menor score
                ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                # Filtra apenas os que pontuaram
                ranking = [r for r in ranking if r[1] > 0]
                
                df_ranking = pd.DataFrame(ranking, columns=["Documento", "Score TF-IDF Acumulado"])
                df_ranking.index = df_ranking.index + 1 # Começa o ranking do 1
                vencedor, score_vencedor = ranking[0]
                df_ranking["Destaque"] = ["🏆 Vencedor"] + ["" for _ in ranking[1:]]
                
                st.success(f"Documento mais relevante: **{vencedor}** (score TF-IDF: {score_vencedor:.4f})")
                st.dataframe(df_ranking, width="stretch")
                
                st.subheader("🧮 Memória de Cálculo")
                st.dataframe(pd.DataFrame(detalhes_calculo), width="stretch")

                if usar_cosseno:
                    vocabulario = sorted({termo for tokens in documentos_processados.values() for termo in tokens})
                    idf_por_termo = {
                        termo: calcular_idf(
                            N,
                            sum(1 for tokens in documentos_processados.values() if termo in tokens),
                        )
                        for termo in vocabulario
                    }

                    def vetor_tfidf(tokens):
                        contagem = Counter(tokens)
                        tamanho = len(tokens) or 1
                        return [
                            (contagem[termo] / tamanho) * idf_por_termo[termo]
                            for termo in vocabulario
                        ]

                    vetor_query = vetor_tfidf(tokens_query)
                    resultados_cosseno = [
                        {
                            "Documento": documento,
                            "Similaridade de Cosseno": round(
                                calcular_similaridade_cosseno(vetor_query, vetor_tfidf(tokens)), 4
                            ),
                        }
                        for documento, tokens in documentos_processados.items()
                    ]
                    df_cosseno = pd.DataFrame(resultados_cosseno).sort_values(
                        "Similaridade de Cosseno", ascending=False
                    )
                    st.subheader("⭐ Bônus: Ranking por Similaridade de Cosseno")
                    st.dataframe(df_cosseno, width="stretch")
else:
    st.info("Faça o upload de pelo menos um manual em PDF para iniciar o processamento.")
