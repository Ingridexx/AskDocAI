# CompareDocAI (Backend) — Q&A sobre documentação com RAG

API backend para perguntas e respostas sobre **documentações** (PDF/TXT/MD) usando **RAG**:
- **Ingestão** → chunking → **FAISS** (vetores)
- **Busca semântica** (com **MMR** opcional)
- **Geração** com **Gemini** (Google)
- **Múltiplas coleções** com upload e **reindex em background**

> Foco deste repo: **backend**. Um frontend em React pode consumir esta API (CORS habilitado).

---

## ✨ Features
- **Coleções**: separe bases por tema (`/collections`, `/{name}/upload`)
- **RAG** com fontes (arquivo e página)
- **MMR** para diversificar resultados
- **Reindexação em background** após upload
- **CORS** liberado p/ front
- **Swagger** em `/docs` e health em `/health`

---

## 🧰 Stack
- **Python 3.11+**
- **FastAPI** + Uvicorn
- **LangChain** + **FAISS**
- **Gemini** (Google) — `text-embedding-004` e `gemini-2.5-flash`

---

## 🏗️ Arquitetura (Visão Rápida)

```mermaid
flowchart LR
A[Cliente / Front] -->|POST /ask| B(FastAPI)
B -->|collection| C[IndexManager]
C -->|load()| D[(FAISS por coleção)]
B -->|retrieve docs| D
B -->|make prompt (contexto+history)| E[Gemini (LLM)]
E -->|answer+citations| B --> A

subgraph Ingestão
U[POST /collections/{name}/upload] --> V{Background}
V --> W[load docs + chunk]
W --> X[FAISS.from_documents]
X --> Y[save index (faiss_index)]
Y --> C
end
