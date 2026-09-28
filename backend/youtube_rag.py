import hashlib
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH = "./chroma_youtube"
COLLECTION_NAME = "youtube_documents"


class YouTubeRAG:

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

    def extract_video_id(self, url):
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]

        if host == "youtu.be":
            video_id = parsed.path.lstrip("/").split("/")[0]
            if video_id:
                return video_id

        if host in ("youtube.com", "m.youtube.com", "music.youtube.com"):
            query = parse_qs(parsed.query)
            if query.get("v"):
                return query["v"][0]

            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                return parts[1]

        raise ValueError(f"Could not extract YouTube video ID from: {url}")

    def load_transcript(self, url):
        video_id = self.extract_video_id(url)
        api = YouTubeTranscriptApi()

        try:
            fetched = api.fetch(video_id, languages=["ar", "en"])
        except (TranscriptsDisabled, NoTranscriptFound):
            try:
                transcript_list = api.list(video_id)
                fetched = transcript_list.find_transcript(
                    [t.language_code for t in transcript_list]
                ).fetch()
            except Exception as e:
                raise ValueError(f"No transcript available for this video: {e}")
        except VideoUnavailable as e:
            raise ValueError(f"Video unavailable: {e}")

        text = "\n".join(snippet.text for snippet in fetched)

        if not text.strip():
            raise ValueError("YouTube transcript is empty.")

        return text, video_id

    def create_document_id(self, video_id):
        video_hash = hashlib.md5(video_id.encode()).hexdigest()
        return f"youtube_{video_hash}"

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
        video_id = self.extract_video_id(url)
        document_id = self.create_document_id(video_id)

        if self._document_exists(document_id):
            return {"document_id": document_id, "video_id": video_id, "url": url, "chunks": 0, "status": "already_indexed"}

        transcript, video_id = self.load_transcript(url)

        document = Document(
            page_content=transcript,
            metadata={
                "source": url,
                "source_type": "youtube",
                "video_id": video_id,
                "document_id": document_id
            }
        )

        chunks = self.text_splitter.split_documents([document])

        for index, chunk in enumerate(chunks):
            chunk.metadata.update({
                "document_id": document_id,
                "video_id": video_id,
                "source_type": "youtube",
                "chunk_id": index
            })

        self.vectorstore.add_documents(chunks)

        return {"document_id": document_id, "video_id": video_id, "url": url, "chunks": len(chunks), "status": "indexed"}

    def _call_llm(self, prompt):
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

    def ask(self, question, document_id=None, k=4):
        if self.vectorstore._collection.count() == 0:
            raise RuntimeError("Vector store is empty. Index a video first.")

        search_kwargs = {"k": k}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)
        documents = retriever.invoke(question)

        if not documents:
            return {"answer": "I don't know based on the video transcript.", "sources": []}

        context = "\n\n".join(doc.page_content for doc in documents)

        prompt = f"""
You are answering a question about a YouTube video.

Use ONLY the transcript context below.

Rules:
- Do not use outside knowledge.
- Do not guess.
- Give a short and direct answer.
- If the answer is not clearly available, say exactly:
I don't know based on the video transcript.

Transcript Context:
{context}

Question:
{question}

Answer:
"""

        answer = self._call_llm(prompt)

        sources = [{
            "document_id": doc.metadata.get("document_id"),
            "video_id": doc.metadata.get("video_id"),
            "source": doc.metadata.get("source"),
            "content": doc.page_content
        } for doc in documents]

        return {"answer": answer, "sources": sources}