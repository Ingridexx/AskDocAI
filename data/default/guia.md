# Guia da Ferramenta

## Instalação
- Use Python 3.11+ e crie uma venv.
- Instale dependências com: pip install -r requirements.txt.


## Uso
- Coloque seus PDFs em data/.
- Execute o indexador para criar o índice FAISS.

## Conceitos
- Chunking: dividir o texto em pedaços com sobreposição para manter contexto.
- Embeddings: vetores numéricos que representam o significado do texto.

## FAQ
**Pergunta:** Onde o índice é salvo?
**Resposta:** Na pasta aiss_index/.

**Pergunta:** Preciso reconstruir o índice ao trocar os PDFs?
**Resposta:** Sim, sempre que alterar documentos ou chunking.
