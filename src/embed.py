# Gera embeddings a partir dos chuncks
from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import settings


def get_embedder() -> GoogleGenerativeAIEmbeddings:
    """
    Cria e retorna o objeto de embeddings configurado para o Gemini.
    - Usa o modelo definido no .env (EMBEDDING_MODEL, default: text-embedding-004)
    - Usa a chave da API do Google (GOOGLE_API_KEY ou GEMINI_API_KEY)
    """
    if not settings.google_api_key:
        raise RuntimeError(
            "Faltou GOOGLE_API_KEY/GEMINI_API_KEY no .env para gerar embeddings."
        )

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )

if __name__ == "__main__":
    
    e = get_embedder()
    vec = e.embed_query("teste rápido de embeddings")
    print("dimensão:", len(vec))