
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.config import settings
from src.utils import log, ensure_dir, timed
from src.ingest import load_documents_from_dir, chunk_documents
from src.embed import get_embedder 

@timed
def build_faiss_index(
    data_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    log(f"[indexer] lendo arquivos de: {data_dir}")
    docs: List[Document] = load_documents_from_dir(data_dir)
    log(f"[indexer] documentos carregados: {len(docs)}")

    log(f"[indexer] gerando chunks (size={chunk_size}, overlap={chunk_overlap})...")
    chunks: List[Document] = chunk_documents(docs, chunk_size, chunk_overlap)
    log(f"[indexer] chunks gerados: {len(chunks)}")

    if not chunks:
        log("[indexer] nenhum chunk encontrado. verifique a pasta data/ e o CHUNK_SIZE.")
        return

    if not settings.google_api_key:
        raise RuntimeError("Faltou GOOGLE_API_KEY/GEMINI_API_KEY no .env para gerar embeddings.")

    log("[indexer] criando embedder (Gemini)...")
    embedder = get_embedder()

    log("[indexer] construindo vetorstore FAISS (pode levar alguns minutos)...")
    vectordb = FAISS.from_documents(chunks, embedder)

    ensure_dir(index_dir)
    log(f"[indexer] salvando índice em: {index_dir}")
    vectordb.save_local(str(index_dir))

    log("[indexer] índice FAISS criado e salvo com sucesso!")

def main() -> None:
    log("==== CompareDocAI :: INDEXER START ====")
    build_faiss_index(
        data_dir=settings.data_dir,
        index_dir=settings.index_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    log("==== CompareDocAI :: INDEXER DONE ====")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[indexer][ERRO] {type(e).__name__}: {e}")
        raise
