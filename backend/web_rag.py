# Hashing
import hashlib

# This libraries to be fast then usual RAG
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import torch # Use this in GPU


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_web"
COLLECTION_NAME = "web_documents"


class WebRAG:

    def __init__(self, llm):
        self.llm = llm

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )

        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings
        )

    def load_website(self, url):
        loader = WebBaseLoader(
            web_paths=(url,),
            requests_kwargs={
                "headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    )
                }
            }
        )

        try:
            documents = loader.load()
        except Exception as e:
            raise ValueError(f"Could not load website: {e}")

        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise ValueError("Could not load website (empty content).")

        return documents

    def create_document_id(self, url):
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"web_{url_hash}"

    def _document_exists(self, document_id):
        try:
            existing = self.vectorstore._collection.get(
                where={"document_id": document_id},
                limit=1
            )
            return len(existing.get("ids", [])) > 0
        except Exception:
            return False

    def create_vectorstore(self, url):
        document_id = self.create_document_id(url)

        if self._document_exists(document_id):
            return {"document_id": document_id, "url": url, "chunks": 0, "status": "already_indexed"}

        documents = self.load_website(url)
        chunks = self.text_splitter.split_documents(documents)

        for index, chunk in enumerate(chunks):
            chunk.metadata.update({
                "document_id": document_id,
                "source_type": "web",
                "source_url": url,
                "chunk_id": index
            })

        self.vectorstore.add_documents(chunks)

        return {"document_id": document_id, "url": url, "chunks": len(chunks), "status": "indexed"}

    def _call_llm(self, prompt):
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

    def ask(self, question, document_id=None, k=4):
        if self.vectorstore._collection.count() == 0:
            raise RuntimeError("Vector store is empty. Index a website first.")

        search_kwargs = {"k": k}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        documents = retriever.invoke(question)

        if not documents:
            return {"answer": "I don't know based on the website.", "sources": []}

        context = "\n\n".join(doc.page_content for doc in documents)

        prompt = f"""
You are answering a question about a website.

Use ONLY the information contained in the context below.

Rules:
- Do not use outside knowledge.
- Do not guess.
- Give a short and direct answer.
- If the answer is not available, say exactly:
I don't know based on the website.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self._call_llm(prompt)

        sources = [{
            "document_id": doc.metadata.get("document_id"),
            "source_url": doc.metadata.get("source_url"),
            "content": doc.page_content
        } for doc in documents]

        return {"answer": answer, "sources": sources}