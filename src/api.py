from __future__ import annotations
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.embed import get_embedder
from src.manager import IndexManager
from src.utils import log

app = FastAPI(title="AskDocAI", version="0.2.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str
    k: int = 4
    mmr: bool = False
    collection: str = "default"    # coleção alvo (default para compatibilidade)
    history: List[Dict[str, str]] = []  

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

def make_prompt(context: str, question: str) -> str:
    return (
        "Você é um assistente que responde ESTRITAMENTE em português e SEM inventar.\n"
        "Responda usando SOMENTE o contexto abaixo. Se a resposta não estiver no contexto, diga que não encontrou.\n\n"
        f"=== CONTEXTO ===\n{context}\n\n"
        f"=== PERGUNTA ===\n{question}\n\n"
        "Responda de forma direta e cite brevemente a origem (arquivo/página/título) quando possível."
    )

@app.on_event("startup")
async def startup() -> None:
    log("[api] startup: preparando recursos...")
    embedder = get_embedder()
    app.state.manager = IndexManager(embedder=embedder)
    app.state.llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )
    # garanta que exista a coleção 'default'
    app.state.manager.data_dir("default")
    app.state.manager.index_dir("default")
    log("[api] startup: pronto ✅")

@app.get("/collections")
async def list_collections() -> Dict[str, Any]:
    """Lista coleções (subpastas em DATA_BASE)."""
    cols = app.state.manager.list_collections()
    return {"collections": cols}

@app.post("/collections")
async def create_collection(name: str = Form(...)) -> Dict[str, str]:
    """Cria uma nova coleção (pasta)."""
    name = name.strip()
    if not name:
        raise HTTPException(400, "Nome inválido.")
    app.state.manager.data_dir(name)
    app.state.manager.index_dir(name)
    return {"status": "created", "collection": name}

@app.post("/collections/{name}/upload")
async def upload_file(
    name: str,
    background: BackgroundTasks,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Faz upload de 1 arquivo para a coleção e dispara rebuild em background.
    """
    ddir = app.state.manager.data_dir(name)
    dest = ddir / file.filename
    content = await file.read()
    dest.write_bytes(content)
    log(f"[api] upload salvo em {dest}")

    # dispara rebuild em background
    background.add_task(
        app.state.manager.rebuild,
        name, settings.chunk_size, settings.chunk_overlap
    )
    return {"status": "accepted", "collection": name, "filename": file.filename}

@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    """
    Chat por coleção. Recebe:
        - collection: qual coleção consultar
        - query: pergunta
        - history: últimos turnos do chat para dar "memória leve"
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query vazia.")

    manager: IndexManager = app.state.manager
    llm: ChatGoogleGenerativeAI = app.state.llm

    # carrega índice da coleção
    try:
        vectordb = manager.load(payload.collection)
    except Exception as e:
        raise HTTPException(404, f"Índice da coleção '{payload.collection}' não encontrado: {e}")

    # busca
    if payload.mmr:
        docs = vectordb.max_marginal_relevance_search(
            payload.query, k=payload.k, fetch_k=max(8, payload.k * 2)
        )
    else:
        docs = vectordb.similarity_search(payload.query, k=payload.k)

    if not docs:
        return AskResponse(answer="Não encontrei nada relacionado no índice.", sources=[])

    # contexto + histórico leve concatenado
    history_text = ""
    if payload.history:
        for turn in payload.history[-6:]:  # pega só os 6 últimos pra ficar barato
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"\n[{role}] {content}"

    context = "\n\n".join(d.page_content for d in docs)
    prompt = (history_text + "\n\n" if history_text else "") + make_prompt(context, payload.query)
    ai_msg = llm.invoke(prompt)

    # fontes
    sources: List[Dict[str, Any]] = []
    for d in docs:
        m = d.metadata or {}
        item = {"source": m.get("source")}
        if m.get("page") is not None:
            item["page"] = int(m["page"]) + 1
        sources.append(item)

    return AskResponse(answer=ai_msg.content, sources=sources)

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

@app.get("/")
async def root() -> dict:
    return {"name": "CompareDocAI", "status": "ok", "docs": "/docs", "health": "/health"}
