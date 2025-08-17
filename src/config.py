from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(".env"))

@dataclass(frozen=True)
class Settings:
    """ 
    Configs imutaveis do projeto
    """
    # IA API KEY
    google_api_key: Optional[str] 
    # Modelos
    embedding_model: str
    llm_model: str
    # Chunking
    chunk_size: int
    chunk_overlap: int
    # Caminhos
    data_dir: Path
    index_dir: Path

    @staticmethod
    def from_env() -> "Settings":
        """ 
        Lê variáveis do .env e aplica defaults sensatos.
        Aceita tanto GOOGLE_API_KEY quanto GEMINI_API_KEY.
        """
        
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            raise ValueError(
                "Faltou a chave da API. Defina GOOGLE_API_KEY ou GEMINI_API_KEY no .env."
            )
        
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-004").strip()
        llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash").strip()
        
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
        index_dir = Path(os.getenv("INDEX_DIR", "./faiss_index")).resolve()
        
        # garante pastas existentes
        data_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        return Settings(
            google_api_key=google_api_key,
            embedding_model=embedding_model,
            llm_model=llm_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            data_dir=data_dir,
            index_dir=index_dir,
        )
    
# Instância global pronta para uso
settings = Settings.from_env()