# AgroSearch

Motor de busca textual para manuais técnicos de agricultura.

## Recursos

- Upload dos cinco manuais PDF do corpus
- Tokenização, normalização, remoção de stopwords e stemming opcional
- Índice invertido implementado manualmente
- Ranking TF-IDF calculado do zero
- Bônus de similaridade de cosseno

## Executar

```bash
pip install -r requirements.txt
streamlit run agroSearch.py
```

Ao abrir o app, envie os PDFs presentes na pasta `docs/`.
