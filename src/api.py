"""
    Como esta sendo usado a API do FastAPI, ele vai receber as perguntas e retornar a resposta.
"""
from __future__ import annotations

from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from embed import get_embedder
from utils import log

app = FastAPI(title="AskDocAI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Trocar pelo domínio do front dps
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Modelos de entrada/saída ======

class AskRequest(BaseModel):
    query: str
    k: int = 4 # quantos chunks mais parecidos trazer do índice
    mmr: bool = False
    
class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    
# ====== Inicializações globais ======

# Carrega o índice FAISS do disco.

log(f"[api] carregando índice FAISS de: {settings.index_dir}")
embedder = get_embedder()
try:
    vectordb = FAISS.load_local(
        str(settings.index_dir),
        embeddings=embedder,
        allow_dangerous_deserialization=True, # -->  é necessário no FAISS do LangChain.
    )
except Exception as e: 
    raise RuntimeError(
        f"Não consegui carregar o índice em {settings.index_dir}. "
        f"Rode antes: `python .\\src\\indexer.py`. Erro: {e}"
    )
    
log(f"[api] usando LLM: {settings.llm_model}")
llm = ChatGoogleGenerativeAI(
    model= settings.llm_model,
    google_api_key=settings.google_api_key,
    temperature=0.2
)

# ====== Funções auxiliares ======
def make_prompt(context: str, question: str) -> str:
    """ 
    Monta um prompt simples que:
    - Força responder em português,
    - limita a resposta ao contexto recuperado do índice,
    - pede para admitir quando não há resposta no contexto.
    """
    return (
        "Você é um assistente que responde ESTRITAMENTE em português e SEM inventar.\n"
        "Responda usando SOMENTE o contexto abaixo. Se a resposta não estiver no contexto, diga que não encontrou.\n\n"
        f"=== CONTEXTO ===\n{context}\n\n"
        f"=== PERGUNTA ===\n{question}\n\n"
        "Responda de forma direta e cite brevemente a origem (arquivo/página/título) quando possível."
    )
    
@app.on_event("startup")
async def startup() -> None:
    log("[api] startup: iniciando carregamento de recursos...")
    # Embedder (verifica chave)
    embedder = get_embedder()
    log(f"[api] startup: carregando índice FAISS de {settings.index_dir} ...")
    try:
        vectordb = FAISS.load_local(
            str(settings.index_dir),
            embeddings=embedder,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        log(f"[api][ERRO] não consegui carregar o índice: {e}")
        raise

    log(f"[api] startup: criando LLM {settings.llm_model} ...")
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )

    # guarda no app.state (jeito padrão de manter dependências globais)
    app.state.vectordb = vectordb
    app.state.llm = llm
    log("[api] startup: pronto ✅")
    
@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    """
    Recebe uma pergunta (query) e retorna:
    - answer: texto do Gemini
    - sources: metadados dos chunks usados (arquivo, página etc.)
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query vazia.")
    
    vectordb = app.state.vectordb
    llm = app.state.llm
    
    if payload.mmr: 
        docs = vectordb.max_marginal_relevance_search(
        payload.query, k=payload.k, fetch_k=max(8, payload.k * 2)  
        )
    
    else:
        # Recupera os k documentos mais semelhantes do índice
        docs = vectordb.similarity_search(payload.query, k=payload.k)
    
    if not docs:
        return AskResponse(
            answer="Não encontrei nada relacionado no índice.",
            sources=[]
        )
        
    # Junta contexto para o modelo
    context = "\n\n".join(d.page_content for d in docs)
    
    # Monta o prompt e chama a IA
    prompt = make_prompt(context, payload.query)
    ai_msg = llm.invoke(prompt) # retorna um AIMessage com .content
    
    # Monta a resposta incluíndo fontes (metadata)
    sources = []
    for d in docs:
        m = d.metadata or {}
        item = {"source": m.get("source")}
        if m.get("page") is not None:
            item["page"] = int(m["page"]) + 1
        sources.append(item)
    return AskResponse(answer=ai_msg.content, sources=sources)
        
@app.get("/health")
async def health() -> dict:
    """ 
    Endpoint simples pra health-check
    """
    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "name": "CompareDocAI",
        "status": "ok",
        "docs": "/docs",
        "health": "/health"
    }