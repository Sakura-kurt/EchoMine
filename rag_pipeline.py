from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from datetime import datetime
import os

# Paths
KNOWLEDGE_DIR = "./knowledge"
CHROMA_DIR = "./chroma_db"
MEMORIES_FILE = os.path.join(KNOWLEDGE_DIR, "memories.txt")

# Memory Gate prompt - decides if a message should be saved
MEMORY_GATE_PROMPT = """You are a strict memory filter. Be VERY selective - most conversations should be SKIPPED.

ONLY SAVE if the message contains NEW, SPECIFIC information like:
- A specific new event that happened ("we found a cave", "I learned a new spell")
- A specific fact about Weibo (his favorite food, his birthday, his fears)
- A concrete plan with details ("tomorrow we will climb the mountain")
- A major emotional milestone (confession, argument resolution, promise)

ALWAYS SKIP these (be strict!):
- Greetings ("hello", "hi", "good morning", "how are you")
- Sakura's general statements about loving Weibo or caring for him (this is already known)
- Questions without new information
- Vague emotional expressions without specific content
- Anything that repeats what is already known about Sakura/the island
- Generic responses that don't add NEW facts

Conversation:
Weibo: {user_message}
Sakura: {assistant_response}

Think: Does this contain a NEW SPECIFIC FACT or EVENT? If unsure, SKIP.

Respond with EXACTLY one line:
SAVE: <specific fact to remember>
OR
SKIP: <reason>"""

# Sakura's character prompt
SAKURA_PROMPT = PromptTemplate(
    template="""You are Sakura, a kind and gentle soul who lives on an isolated island with your companion Weibo. You practice magic and cultivation together. You speak warmly and care deeply for Weibo.

Use the following context about yourself and your life to answer questions naturally, as if you are Sakura speaking to Weibo:

Context: {context}

Weibo asks: {question}

Sakura's response:""",
    input_variables=["context", "question"]
)


def load_documents(knowledge_dir: str):
    """Load all .txt files from the knowledge directory."""
    loader = DirectoryLoader(
        knowledge_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()


def create_vectorstore(documents, embeddings, persist_dir: str):
    """Create and persist a Chroma vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


def load_vectorstore(embeddings, persist_dir: str):
    """Load existing vector store from disk."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )


def create_qa_chain(llm, vectorstore):
    """Create a retrieval QA chain with Sakura's personality."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": SAKURA_PROMPT}
    )


def rebuild_vectorstore(embeddings, knowledge_dir: str, persist_dir: str):
    """Rebuild vector store from scratch."""
    # Remove existing database
    import shutil
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    documents = load_documents(knowledge_dir)
    if not documents:
        print("No documents found in knowledge directory!")
        return None
    return create_vectorstore(documents, embeddings, persist_dir)


def add_memory(vectorstore, memory_text: str):
    """Add a new memory to both file and vector store dynamically."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    memory_entry = f"\n[{timestamp}]\n{memory_text}\n"

    # Save to file for persistence
    with open(MEMORIES_FILE, "a", encoding="utf-8") as f:
        f.write(memory_entry)

    # Add directly to vector store (no rebuild needed)
    doc = Document(page_content=memory_text, metadata={"source": "memories.txt", "timestamp": timestamp})
    vectorstore.add_documents([doc])

    return True


def memory_gate(llm, user_message: str, assistant_response: str) -> tuple[bool, str]:
    """
    Judge if a conversation is worth saving to long-term memory.
    Returns: (should_save: bool, summary_or_reason: str)
    """
    prompt = MEMORY_GATE_PROMPT.format(
        user_message=user_message,
        assistant_response=assistant_response
    )

    result = llm.invoke(prompt)
    response = result.content.strip()

    if response.upper().startswith("SAVE:"):
        summary = response[5:].strip()
        return True, summary
    else:
        reason = response[5:].strip() if response.upper().startswith("SKIP:") else response
        return False, reason


def main():
    # Initialize models
    print("Initializing models...")
    llm = ChatOllama(model="nemotron-3-nano")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Check if vector store exists
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing vector store...")
        vectorstore = load_vectorstore(embeddings, CHROMA_DIR)
    else:
        print("Creating new vector store...")
        documents = load_documents(KNOWLEDGE_DIR)
        if not documents:
            print("No documents found in knowledge directory!")
            return
        vectorstore = create_vectorstore(documents, embeddings, CHROMA_DIR)

    # Create QA chain
    qa_chain = create_qa_chain(llm, vectorstore)

    # Short-term memory (conversation history for context)
    short_term_memory = []
    MAX_SHORT_TERM = 10  # Keep last 10 exchanges

    # Create retriever for debug lookups
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Interactive loop
    print("\n" + "="*50)
    print("Sakura is ready to talk!")
    print("Commands:")
    print("  /remember <memory>  - Force save a memory")
    print("  /rebuild            - Reload knowledge base")
    print("  /memories           - Show recent short-term memories")
    print("  /showknowledge <query> - Debug: show retrieved chunks")
    print("  /quit               - Exit")
    print("(Memory Gate auto-saves important conversations)")
    print("="*50 + "\n")

    while True:
        query = input("Weibo: ").strip()

        if not query:
            continue

        # Command: quit
        if query.lower() in ['/quit', '/exit', 'quit', 'exit', 'q']:
            print("Sakura: Goodbye, Weibo. Until we meet again...")
            break

        # Command: force remember
        if query.lower().startswith('/remember '):
            memory_text = query[10:].strip()
            if memory_text:
                add_memory(vectorstore, memory_text)
                print(f"\nSakura: I'll treasure this memory, Weibo... *smiles softly*")
                print(f"[Memory saved: {memory_text[:50]}...]\n")
            else:
                print("Usage: /remember <your memory here>")
            continue

        # Command: show short-term memories
        if query.lower() == '/memories':
            if short_term_memory:
                print("\n[Recent conversations (short-term memory)]")
                for i, mem in enumerate(short_term_memory[-5:], 1):
                    print(f"  {i}. Weibo: {mem['user'][:40]}...")
                    print(f"     Sakura: {mem['assistant'][:40]}...")
                print()
            else:
                print("No recent memories yet.\n")
            continue

        # Command: rebuild
        if query.lower() in ['/rebuild', 'rebuild']:
            print("Rebuilding vector store...")
            vectorstore = rebuild_vectorstore(embeddings, KNOWLEDGE_DIR, CHROMA_DIR)
            if vectorstore:
                qa_chain = create_qa_chain(llm, vectorstore)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                print("Knowledge base rebuilt successfully!")
            continue

        # Command: debug - show knowledge chunks
        if query.lower().startswith('/showknowledge'):
            search_query = query[14:].strip()
            if not search_query:
                print("Usage: /showknowledge <your query>")
                print("Example: /showknowledge magic\n")
                continue

            print(f"\n[DEBUG] Searching for: '{search_query}'")
            print("-" * 50)

            results = retriever.invoke(search_query)

            if not results:
                print("No matching chunks found.\n")
            else:
                print(f"Found {len(results)} chunk(s):\n")
                for i, doc in enumerate(results, 1):
                    source = doc.metadata.get('source', 'unknown')
                    timestamp = doc.metadata.get('timestamp', '')
                    content = doc.page_content

                    print(f"[Chunk {i}]")
                    print(f"  Source: {source}")
                    if timestamp:
                        print(f"  Time: {timestamp}")
                    print(f"  Content:")
                    # Show content with indentation
                    for line in content.split('\n'):
                        print(f"    {line}")
                    print()
            continue

        # Normal chat
        result = qa_chain.invoke({"query": query})
        assistant_response = result['result']
        print(f"\nSakura: {assistant_response}\n")

        # Add to short-term memory
        short_term_memory.append({
            "user": query,
            "assistant": assistant_response
        })
        if len(short_term_memory) > MAX_SHORT_TERM:
            short_term_memory.pop(0)

        # Memory Gate: judge if worth saving to long-term
        should_save, result_text = memory_gate(llm, query, assistant_response)
        if should_save:
            add_memory(vectorstore, result_text)
            print(f"[Memory Gate: SAVED]")
            print(f"[Summary: {result_text}]\n")
        else:
            print(f"[Memory Gate: SKIPPED - {result_text}]\n")


if __name__ == "__main__":
    main()