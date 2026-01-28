"""Test script to verify RAG pipeline is working correctly."""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil

KNOWLEDGE_DIR = "./knowledge"
CHROMA_DIR = "./chroma_db"


def test_rag_pipeline():
    print("=" * 60)
    print("RAG PIPELINE TEST")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[1] Loading documents from knowledge/...")
    loader = DirectoryLoader(
        KNOWLEDGE_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"    Loaded {len(documents)} document(s)")
    for doc in documents:
        print(f"    - {doc.metadata.get('source', 'unknown')}")

    # Step 2: Create chunks
    print("\n[2] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"    Created {len(chunks)} chunk(s)")

    # Step 3: Create embeddings
    print("\n[3] Initializing embedding model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("    nomic-embed-text loaded")

    # Step 4: Create vector store (fresh)
    print("\n[4] Creating vector store...")
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"    Vector store created at {CHROMA_DIR}")

    # Step 5: Test retrieval
    print("\n[5] Testing retrieval...")
    test_queries = [
        "What magic can Sakura do?",
        "Who is Weibo?",
        "What do they do on the island?",
    ]

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    for query in test_queries:
        print(f"\n    Query: '{query}'")
        print("    Retrieved chunks:")
        results = retriever.invoke(query)
        for i, doc in enumerate(results):
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"      [{i+1}] {preview}...")

    # Step 6: Test full chain with LLM
    print("\n[6] Testing full chain with LLM...")
    llm = ChatOllama(model="nemotron-3-nano")

    from langchain_classic.chains import RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    test_query = "What kind of magic can you do?"
    print(f"\n    Query: '{test_query}'")
    result = qa_chain.invoke({"query": test_query})

    print(f"\n    LLM Response:\n    {result['result']}")

    print("\n    Source documents used:")
    for i, doc in enumerate(result['source_documents']):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"      [{i+1}] {preview}...")

    print("\n" + "=" * 60)
    print("TEST COMPLETE - RAG is working if chunks were retrieved!")
    print("=" * 60)


if __name__ == "__main__":
    test_rag_pipeline()