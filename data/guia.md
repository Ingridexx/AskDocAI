# Guia da Ferramenta

## Instalação
- Use Python 3.11+ e crie uma venv.
- Instale dependências com: pip install -r requirements.txt.
- Você vai precisar de uma API KEY de alguma LLM, recomendo a do GOOGLE GEMINI, na qual estou usando nesse projeto, você pode encontra-la no seguinte link:
- [Google AI Studio](http://aistudio.google.com/app/apikey)
- Você também pode usar outro tipo de LLM, recomendo ler a documentação da LLM que você quer utilizar e alterar aqui no código para ser compatível com a qual você escolheu.

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
