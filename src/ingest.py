# Aqui os documentos são carregados e quebrados em "chunks"
from __future__ import annotations

from pathlib import Path
from typing import List, Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import settings
from utils import log

# extensões de arquivo que vai ser aceito na primeira versão
SUPPORTED_EXTS = {".pdf", ".txt", ".md"}

def _iter_files(base_dir: Path, include_subdirs: bool = True) -> Iterable[Path]:
    """
    Lista arquivos dentro de base_dir.
    - Se include_subdirs=True, busca recursivamente.
    - Filtra por extensões suportadas (SUPPORTED_EXTS).
    """
    pattern = "**/*" if include_subdirs else "*"
    for path in sorted(base_dir.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def load_documents_from_dir(base_dir: Path, include_subdirs: bool = True) -> List[Document]:
    """
    Carrega arquivos .pdf, .txt e .md do diretório informado e converte em uma lista de Document.
    - Para PDF: usa PyPDFLoader (quebra por páginas automaticamente).
    - Para TXT/MD: usa TextLoader (lê o arquivo inteiro).
    - Adiciona metadados úteis (source, file_type).
    """
    docs: List[Document] = []

    if not base_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {base_dir}")

    for file_path in _iter_files(base_dir, include_subdirs=include_subdirs):
        try:
            ext = file_path.suffix.lower()

            # cada loader retorna uma lista de Document
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
                file_docs = loader.load()
            else:  # ".txt" ou ".md" -> tratamos como texto puro
                loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
                file_docs = loader.load()

            # adiciona/garante metadados mínimos em cada Document
            for d in file_docs:
                d.metadata = d.metadata or {}
                d.metadata.setdefault("source", file_path.name)
                d.metadata.setdefault("file_type", ext)

                # limpa documentos vazios (por segurança)
                if not d.page_content or not d.page_content.strip():
                    continue

                docs.append(d)
        except Exception as e:
            log(f"[ingest] erro ao processar {file_path.name}: {e}")

    return docs


def chunk_documents(
    docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Divide os documentos em blocos menores mantendo overlap para contexto.
    Ex.: chunk_size=1000 e overlap=200 (valores vindo do .env).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # índice sequencial (útil para debug e rastreio)
    for i, c in enumerate(chunks):
        c.metadata = c.metadata or {}
        c.metadata.setdefault("chunk_index", i)

    return chunks


def _print_summary(docs: List[Document], chunks: List[Document]) -> None:
    """
    Mostra um resuminho no console para conferência rápida.
    """
    log(f"[ingest] documentos carregados: {len(docs)}")
    log(f"[ingest] chunks gerados: {len(chunks)}")

    if chunks:
        sample = chunks[0]
        preview = sample.page_content[:200].replace("\n", " ")
        log(
            f"[ingest] exemplo chunk[0] "
            f"({len(sample.page_content)} chars) | fonte={sample.metadata.get('source')}: {preview}..."
        )


def main() -> None:
    """
    Fluxo completo da ingestão:
    1) Lê arquivos de settings.data_dir
    2) Quebra em chunks (settings.chunk_size / settings.chunk_overlap)
    3) Exibe um resumo para validar o processo
    """
    log(f"[ingest] lendo documentos de: {settings.data_dir}")
    docs = load_documents_from_dir(settings.data_dir, include_subdirs=True)

    log(
        f"[ingest] fazendo chunking (size={settings.chunk_size}, "
        f"overlap={settings.chunk_overlap})..."
    )
    chunks = chunk_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    _print_summary(docs, chunks)


if __name__ == "__main__":
    main()