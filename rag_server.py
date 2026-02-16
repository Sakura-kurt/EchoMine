import asyncio
import os

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from rag_pipeline import (
    load_vectorstore, create_vectorstore, load_documents,
    create_qa_chain, memory_gate, add_memory,
    KNOWLEDGE_DIR, CHROMA_DIR,
)

app = FastAPI(title="RAG Server (Sakura)")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

llm = None
vectorstore = None
qa_chain = None


class QueryRequest(BaseModel):
    text: str


class MemoryRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup():
    global llm, vectorstore, qa_chain

    print("[rag-server] Loading LLM and embeddings...")
    llm = ChatOllama(model="nemotron-3-nano", base_url=OLLAMA_HOST)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)

    print("[rag-server] Loading vector store...")
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
    else:
        documents = load_documents(KNOWLEDGE_DIR)
        if documents:
            vectorstore = create_vectorstore(documents, embeddings, CHROMA_DIR)
        else:
            print("[rag-server] WARNING: No knowledge documents found!")
            return

    qa_chain = create_qa_chain(llm, vectorstore)
    print("[rag-server] Ready.")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "llm_loaded": llm is not None,
        "vectorstore_loaded": vectorstore is not None,
    }


@app.post("/query")
async def query(req: QueryRequest):
    if not qa_chain:
        return {"response": "", "error": "RAG pipeline not initialized"}

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: qa_chain.invoke({"query": req.text})
    )
    response_text = result["result"]

    # Memory gate (fire and forget)
    loop.run_in_executor(
        None, _memory_gate_sync, req.text, response_text
    )

    return {"response": response_text}


@app.post("/memory")
async def add_memory_endpoint(req: MemoryRequest):
    if not vectorstore:
        return {"success": False, "error": "Vector store not initialized"}

    add_memory(vectorstore, req.text)
    return {"success": True}


def _memory_gate_sync(user_msg: str, assistant_msg: str):
    should_save, summary = memory_gate(llm, user_msg, assistant_msg)
    if should_save:
        add_memory(vectorstore, summary)
        print(f"[rag-server] Memory SAVED: {summary[:60]}")
    else:
        print(f"[rag-server] Memory SKIPPED: {summary[:60]}")
