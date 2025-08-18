
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Dict, List

from langchain_community.vectorstores import FAISS

from src.config import settings
from src.utils import ensure_dir, log
from src.ingest import load_documents_from_dir, chunk_documents

__all__ = ["IndexManager"]  # ajuda a deixar explícito o export


class IndexManager:
    """
    Gerencia índices por coleção:
        - resolve pastas de dados/índice,
        - carrega FAISS (com cache),
        - reconstrói índice a partir dos arquivos da coleção.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self._cache: Dict[str, FAISS] = {}
        self._lock = Lock()

    # --- caminhos por coleção ---
    def data_dir(self, name: str) -> Path:
        return ensure_dir(settings.data_base / name)

    def index_dir(self, name: str) -> Path:
        return ensure_dir(settings.index_base / name)

    # --- utilidades ---
    def list_collections(self) -> List[str]:
        base = settings.data_base
        if not base.exists():
            return []
        return sorted([p.name for p in base.iterdir() if p.is_dir()])

    # --- operações principais ---
    def load(self, name: str) -> FAISS:
        """
        Carrega o índice FAISS da coleção (usa cache em memória).
        Levanta erro se o índice ainda não existe.
        """
        with self._lock:
            if name in self._cache:
                return self._cache[name]

            idx_dir = self.index_dir(name)

            # proteção: índice ainda não existe / pasta vazia
            if not idx_dir.exists() or not any(idx_dir.glob("*")):
                raise FileNotFoundError(
                    f"Índice da coleção '{name}' não existe em {idx_dir}. "
                    f"Faça upload de documentos e reconstrua."
                )

            log(f"[manager] carregando FAISS de {idx_dir}")
            vectordb = FAISS.load_local(
                str(idx_dir),
                embeddings=self.embedder,
                allow_dangerous_deserialization=True,
            )
            self._cache[name] = vectordb
            return vectordb

    def rebuild(self, name: str, chunk_size: int, chunk_overlap: int) -> None:
        """
        Reconstrói o índice FAISS lendo todos os arquivos em data_dir(name).
        """
        ddir = self.data_dir(name)
        idir = self.index_dir(name)

        log(f"[manager] rebuild da coleção '{name}' lendo {ddir}")
        docs = load_documents_from_dir(ddir, include_subdirs=True)
        if not docs:
            log(f"[manager] nenhum documento encontrado em {ddir}; índice não atualizado.")
            return

        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        log(f"[manager] {len(chunks)} chunks; construindo FAISS...")
        vectordb = FAISS.from_documents(chunks, self.embedder)

        log(f"[manager] salvando índice em {idir}")
        vectordb.save_local(str(idir))

        with self._lock:
            self._cache[name] = vectordb
