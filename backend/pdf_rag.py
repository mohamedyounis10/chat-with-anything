# Hashing and Deal with paths 
import hashlib
from pathlib import Path

import torch # Use this in GPU

# This libraries to be fast then usual RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "./chroma_pdf"
COLLECTION_NAME = "pdf_documents"


class PDFRAG:

    def __init__(self, llm):
        self.llm = llm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model Embedding
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
        
    # Load PDF
    def load_pdf(self, pdf_path):
        pdf_path = str(pdf_path)
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        return PyPDFLoader(pdf_path).load()
    
    def create_document_id(self, pdf_path):
        path = Path(pdf_path)
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()
        return f"pdf_{file_hash}"

    def _document_exists(self, document_id):
        try:
            existing = self.vectorstore._collection.get(
                where={"document_id": document_id},
                limit=1
            )
            return len(existing.get("ids", [])) > 0
        except Exception:
            return False
        
    # Store in database
    def create_vectorstore(self, pdf_path):
        document_id = self.create_document_id(pdf_path)
        filename = Path(pdf_path).name

        if self._document_exists(document_id):
            return {
                "document_id": document_id,
                "document_name": filename,
                "chunks": 0,
                "status": "already_indexed"
            }

        documents = self.load_pdf(pdf_path)
        chunks = self.text_splitter.split_documents(documents)

        for index, chunk in enumerate(chunks):
            chunk.metadata.update({
                "document_id": document_id,
                "document_name": filename,
                "source_type": "pdf",
                "chunk_id": index
            })

        self.vectorstore.add_documents(chunks)

        return {
            "document_id": document_id,
            "document_name": filename,
            "chunks": len(chunks),
            "status": "indexed"
        }

    def _call_llm(self, prompt):
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

    def ask(self, question, document_id=None, k=4):
        if self.vectorstore._collection.count() == 0:
            raise RuntimeError("Vector store is empty. Index a document first.")

        search_kwargs = {"k": k}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        documents = retriever.invoke(question)

        if not documents:
            return {"answer": "I don't know based on the provided document.", "sources": []}

        context = "\n\n".join(doc.page_content for doc in documents)

        prompt = f"""
You are a precise document question-answering assistant.

Answer the question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not guess.
- If the answer is clearly available, answer directly.
- Keep the answer concise.
- If the answer is not available, say exactly:
I don't know based on the provided document.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self._call_llm(prompt)

        sources = [{
            "document_id": doc.metadata.get("document_id"),
            "document_name": doc.metadata.get("document_name"),
            "page": doc.metadata.get("page"),
            "content": doc.page_content
        } for doc in documents]

        return {"answer": answer, "sources": sources}