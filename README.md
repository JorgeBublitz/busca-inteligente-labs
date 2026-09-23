# Busca Inteligente Labs

Projetos práticos de Recuperação de Informação e Processamento de Linguagem Natural desenvolvidos na disciplina **Tendências em Ciência da Computação**.

O repositório está organizado por branch para que cada desafio possa ser estudado, executado e versionado de maneira independente.

## Projetos

| Branch | Projeto | Principais conceitos |
| --- | --- | --- |
| `agrosearch` | Motor de busca para manuais agrícolas | Pré-processamento, índice invertido, TF-IDF e similaridade de cosseno |
| `healthsearch` | Motor de busca híbrido para protocolos médicos | BM25, embeddings, similaridade de cosseno, RRF e Cross-Encoder |
| `ouvidoria-inteligente` | Triagem semântica de manifestações cidadãs | Embeddings, duplicatas, PCA/t-SNE e chunking |

## Como usar

1. Troque para a branch do projeto desejado:

   ```bash
   git switch agrosearch
   ```

2. Crie e ative um ambiente virtual Python.
3. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

4. Execute a aplicação Streamlit indicada no README da branch.

## Estrutura

Cada branch contém apenas os arquivos do respectivo desafio: código-fonte, dependências, dados de apoio e, quando aplicável, notebooks de análise.

## Tecnologias

- Python e Streamlit
- TF-IDF e BM25
- Sentence Transformers e embeddings
- Scikit-learn
- LangChain Text Splitters
- Pandas, NumPy e Plotly

## Observação

Os modelos de embeddings podem ser baixados automaticamente na primeira execução. Por isso, a primeira inicialização pode levar alguns minutos e exige conexão com a internet.
