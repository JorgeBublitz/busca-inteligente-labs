# Ouvidoria Inteligente

Protótipo de triagem semântica para manifestações cidadãs.

## Recursos

- Base com 40 manifestações anônimas e cinco textos longos
- Busca semântica com top-k configurável
- Comparação entre BoW, TF-IDF e embeddings
- Detecção de duplicatas com limiar ajustável
- Visualização PCA ou t-SNE por categoria
- Chunking com `RecursiveCharacterTextSplitter`

## Executar

```bash
pip install -r requirements.txt
streamlit run app_ouvidoria.py
```

## Notebooks

- `analise_comparativa.ipynb`
- `deteccao_duplicatas.ipynb`
- `chunking_manifestacoes.ipynb`
