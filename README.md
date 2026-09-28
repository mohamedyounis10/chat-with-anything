<div align="center">

<h1 align="center">
  💬 Chat With Anything
</h1>

> An AI-powered RAG assistant that lets you chat with **PDF files**, **websites**, and **YouTube videos**. Every answer comes straight from your source, together with the passages used to produce it.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Mistral-orange)
![RAG](https://img.shields.io/badge/Architecture-RAG-8b5cf6)

</div>

---

## 📌 Overview

**Chat With Anything** is a Retrieval-Augmented Generation (RAG) application with a clean Streamlit interface. You add a source (a PDF, a web page, or a YouTube video), the app indexes it, and you can then ask questions about its content in a chat window while previewing the source side by side.

To keep the app lightweight on local machines, the **LLM (Mistral) runs on a Kaggle GPU notebook** and is exposed through an **ngrok** tunnel. Everything else (document loading, chunking, embeddings, vector store, and the Streamlit app) runs locally.

## ✨ Features

- 📄 **Chat with PDFs**: upload a PDF, index it, and ask questions about it.
- 🌐 **Chat with websites**: paste any article or page URL.
- ▶️ **Chat with YouTube videos**: the video transcript is fetched and indexed.
- 📚 **Source-grounded answers**: each answer comes with an expandable "Sources" section showing the retrieved passages.
- 🖼️ **Live source preview**: the PDF, web page, or video is displayed next to the chat.
- 🔌 **Remote GPU LLM**: run the model on free Kaggle GPUs and connect from the sidebar with a single URL.
- 🔄 **Switch sources anytime** with the "New Source" button.

## 🏗️ Architecture

```
┌──────────────────────────── Local machine ────────────────────────────┐
│                                                                        │
│   Streamlit UI (app.py)                                                │
│        │                                                               │
│        ▼                                                               │
│   backend/                                                             │
│     ├── pdf_rag.py      (PDFRAG)                                       │
│     ├── web_rag.py      (WebRAG)                                       │
│     └── youtube_rag.py  (YouTubeRAG)                                   │
│        │   load → chunk → embed → vector store → retrieve              │
│        ▼                                                               │
│   wraaper.py  (KaggleMistralLLM)  ── HTTP POST /generate ──┐           │
└────────────────────────────────────────────────────────────┼───────────┘
                                                             │  ngrok tunnel
                                            ┌────────────────▼───────────┐
                                            │  Kaggle GPU notebook       │
                                            │  Mistral LLM + /generate   │
                                            └────────────────────────────┘
```

**Flow:**

1. The user adds a source and clicks **Index**. The matching engine (`PDFRAG`, `WebRAG`, or `YouTubeRAG`) builds a vector store and returns a `document_id`.
2. The user asks a question. The engine retrieves the most relevant passages for that `document_id`.
3. The passages and the question are sent as a prompt to the Mistral model on Kaggle through `KaggleMistralLLM.invoke()`.
4. The answer is displayed in the chat, along with the passages used.

## 📁 Project Structure

```
chat-with-anything/
├── app.py                      # Streamlit application (UI + session logic)
├── wraaper.py                  # KaggleMistralLLM: client for the remote LLM API
├── backend/
│   ├── pdf_rag.py              # PDF RAG engine
│   ├── web_rag.py              # Website RAG engine
│   └── youtube_rag.py          # YouTube RAG engine
├── kaggel-notebook-gpu.ipynb   # Kaggle notebook that hosts the LLM API
├── test.py                     # Test script
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Kaggle](https://www.kaggle.com/) account (GPU enabled)
- A free [ngrok](https://ngrok.com/) account and auth token

### 1. Clone the repository

```bash
git clone https://github.com/mohamedyounis10/chat-with-anything.git
cd chat-with-anything
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the LLM on Kaggle

1. Upload `kaggel-notebook-gpu.ipynb` to Kaggle.
2. Enable a **GPU** accelerator in the notebook settings.
3. Add your **ngrok auth token** (for example as a Kaggle secret).
4. Run all cells. When the API server starts, the notebook prints a public **ngrok URL** such as `https://xxxx.ngrok-free.app`. Copy it.

> The notebook exposes a `/generate` endpoint that accepts `prompt`, `max_new_tokens`, and `temperature`, and returns the generated text in a `response` field.

### 4. Run the app

```bash
streamlit run app.py
```
<img width="1917" height="892" alt="Screenshot 2026-09-28 184007" src="https://github.com/user-attachments/assets/cbb8dce4-927f-441b-b72b-f8fcecadbd20" />
<img width="1912" height="897" alt="Screenshot 2026-09-28 184305" src="https://github.com/user-attachments/assets/1ae5b7a3-2548-44f9-b95e-e4cc752b28ce" />
<img width="1917" height="913" alt="Screenshot 2026-09-28 184119" src="https://github.com/user-attachments/assets/ce92798f-63fc-4504-9176-8175f6ef405b" />


### 5. Connect and chat

1. In the sidebar, paste the **Kaggle ngrok URL** and click **Connect to Model**.
2. Choose a tab: **PDF**, **Website**, or **YouTube**.
3. Add your source and click **Index**.
4. Start asking questions. 🎉

## 🔌 LLM API Contract

The app talks to the Kaggle-hosted model through a simple HTTP API (see `wraaper.py`):

**Request**

```http
POST {ngrok_url}/generate
Content-Type: application/json

{
  "prompt": "Your prompt here",
  "max_new_tokens": 512,
  "temperature": 0.3
}
```

**Response**

```json
{ "response": "Generated text..." }
```

Default generation settings: `max_new_tokens=512`, `temperature=0.3`, request timeout of 120 seconds.

## 🧩 RAG Engine Interface

Each engine in `backend/` follows the same interface, which keeps `app.py` simple and makes new sources easy to add:

| Method | Description |
| --- | --- |
| `create_vectorstore(source)` | Loads and indexes the source, and returns a dict containing `document_id`. |
| `ask(question, document_id)` | Retrieves relevant passages and returns a dict with `answer` and `sources`. |

## 🛠️ Tech Stack

- **Frontend:** Streamlit (with custom CSS)
- **LLM:** Mistral, served from a Kaggle GPU notebook
- **Tunnel:** ngrok
- **Architecture:** Retrieval-Augmented Generation (RAG)
- **Sources:** PDF, web pages (web scraping), YouTube transcripts

## ⚠️ Notes & Limitations

- The ngrok URL changes every time the Kaggle notebook restarts, so reconnect from the sidebar when that happens.
- Kaggle sessions have runtime limits, and the LLM stops when the session ends.
- YouTube chat requires the video to have an available transcript.
- Some websites block being embedded in an iframe, so the live preview may not render for every page. Chat still works from the indexed content.

## 🔮 Future Improvements

- Chat history persistence across sessions
- Support for more formats (DOCX, TXT, CSV)
- Streaming responses
- Multi-document chat
- Docker support for easier deployment

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 👤 Author

**Mohamed Younis**

This project was developed as part of my **AI & LLM Internship at Tips Hindawi**.

* GitHub: [@mohamedyounis10](https://github.com/mohamedyounis10)
* LinkedIn: [Mohamed Younis](https://www.linkedin.com/in/mohamedyounis15/)
* Internship Program: [Tips Hindawi](https://www.linkedin.com/company/tipshindawi/)

---

⭐ If you found this project useful, consider giving it a star!


---

⭐ If you found this project useful, consider giving it a star!
