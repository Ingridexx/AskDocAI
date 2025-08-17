# Funções auxiliares genéricas
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

def log(msg: str) -> None:
    """
    Imprime mensagens no stdout e já faz flush (é útil para ver logs em tempo real)
    """
    print(msg, file=sys.stdout, flush=True)

def ensure_dir(path: Path) -> Path:
    """
    Garante que o diretório 'path' exista. Se não existir, cria.
    Retorna o próprio Path para encadear chamadas.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def timed(fn: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator simples pra medir tempo de execução de funções.
    """
    
    def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log(f"[timed] {fn.__name__} levou {elapsed:.2f}s")
        return result
    return wrapper