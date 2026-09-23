# HealthSearch

Motor de busca híbrido para protocolos médicos, combinando recuperação léxica e semântica.

## Recursos

- Corpus médico pré-carregado com seis diretrizes
- BM25 com ajuste de `k1` e `b`
- Embeddings multilíngues e similaridade de cosseno
- Reciprocal Rank Fusion com peso `alpha`
- Abas para análise léxica, semântica, híbrida e comparativa
- Bônus: Cross-Encoder aplicado aos três melhores resultados

## Executar

```bash
pip install -r requirements.txt
streamlit run healthsearch_app.py
```

Na primeira execução, os modelos de embeddings podem ser baixados automaticamente.
