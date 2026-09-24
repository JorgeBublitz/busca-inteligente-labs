# Busca Inteligente Labs

Três motores de busca em Python, do mais simples ao mais sofisticado: busca léxica com TF-IDF implementado do zero, busca híbrida que combina BM25 com embeddings, e triagem semântica de textos. Projetos da disciplina **Tendências em Ciência da Computação**.

> **O código de cada projeto fica na branch dele.** Esta branch `main` tem só a visão geral. Use os links da tabela para abrir cada projeto.

## Projetos

| Branch | Projeto | Principais conceitos |
| --- | --- | --- |
| [`agrosearch`](https://github.com/JorgeBublitz/busca-inteligente-labs/tree/agrosearch) | Motor de busca para manuais agrícolas em PDF | Pré-processamento, stemming, índice invertido, TF-IDF e similaridade de cosseno implementados sem bibliotecas de busca |
| [`healthsearch`](https://github.com/JorgeBublitz/busca-inteligente-labs/tree/healthsearch) | Motor de busca híbrido para protocolos médicos | BM25, embeddings multilíngues, Reciprocal Rank Fusion (RRF) e reordenação com Cross-Encoder |
| [`ouvidoria-inteligente`](https://github.com/JorgeBublitz/busca-inteligente-labs/tree/ouvidoria-inteligente) | Triagem semântica de manifestações de cidadãos | Busca semântica, comparação BoW × TF-IDF × embeddings, detecção de duplicatas, PCA/t-SNE e chunking |

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
