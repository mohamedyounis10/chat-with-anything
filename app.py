import streamlit as st
from pathlib import Path
import base64

# Load Small Projects
from backend.pdf_rag import PDFRAG
from backend.web_rag import WebRAG
from backend.youtube_rag import YouTubeRAG
from wraaper import KaggleMistralLLM

# Page Configrations
st.set_page_config(
    page_title="Chat With Anything",
    page_icon="💬",
    layout="wide" 
)

# 1. Ngrok Slider
st.sidebar.header("⚙️ LLM Connection")

api_url = st.sidebar.text_input(
    "Kaggle ngrok URL",
    value=st.session_state.get("api_url", ""),
    placeholder="https://..."
)

if st.sidebar.button("Connect to Model"):
    if not api_url.strip():
        st.sidebar.error("Please enter the ngrok URL first.")
    else:
        try:
            llm = KaggleMistralLLM(api_url=api_url.strip())
            _ = llm.invoke("Say only the word: OK")
            st.session_state.llm = llm
            st.session_state.api_url = api_url.strip()
            st.sidebar.success("Connected successfully! ✅")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}")

llm_ready = "llm" in st.session_state

if not llm_ready:
    st.sidebar.warning("Model is not connected yet.")


# 2. CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    .main .block-container {
        padding-bottom: 5rem !important;
    }

    .hero {
        text-align: center;
        padding: 10px 20px 40px;
    }
    .hero h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        line-height: 1.2;
        margin-bottom: 10px;
    }
    .hero h1 span { color: #8b5cf6 !important; }
    .hero p { font-size: 1.1rem !important; opacity: 0.8; }

    .stTabs {
        padding: 40px 30px; 
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background-color: rgba(255, 255, 255, 0.03); 
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        max-width: 1100px !important; 
        width: 100% !important;
        margin: 0 auto;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
        border-bottom: none;
        margin-bottom: 30px;
        background-color: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 9999px;
        width: fit-content;
        margin: 0 auto 40px auto;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9999px !important;
        padding: 12px 32px !important; 
        background-color: transparent;
        border: none !important;
        font-size: 1.1rem !important; 
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-bottom: 2px solid #8b5cf6 !important; 
    }

    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        background-color: rgba(0, 0, 0, 0.2) !important;
        padding: 60px 20px !important; 
        min-height: 200px;
    }

    .stButton>button {
        background-color: #8b5cf6 !important;
        color: white !important;
        border-radius: 24px !important;
        padding: 12px 28px !important;
        border: none !important;
        font-weight: 500 !important;
        font-size: 1.05rem !important;
        display: flex;
        margin: 0 auto;
    }
    .stButton>button:hover { background-color: #7c3aed !important; }

    .stTextInput>div>div>input {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        padding: 16px 20px !important;
        font-size: 1.1rem !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139,92,246,0.3) !important;
    }

    [data-testid="stChatMessageContent"] {
        font-size: 1.15rem !important; 
        line-height: 1.6 !important;
    }
    [data-testid="stChatInput"] {
        padding-bottom: 15px !important; 
    }
    [data-testid="stChatInput"] textarea {
        font-size: 1.15rem !important; 
        padding: 18px 20px !important; 
        border-radius: 16px !important;
        line-height: 1.5 !important;
    }
    
    .pdf-container {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# 3. Session State
def get_engine(source_type):
    key = f"{source_type}_engine"
    if key not in st.session_state:
        if source_type == "pdf":
            st.session_state[key] = PDFRAG(llm=st.session_state.llm)
        elif source_type == "web":
            st.session_state[key] = WebRAG(llm=st.session_state.llm)
        elif source_type == "youtube":
            st.session_state[key] = YouTubeRAG(llm=st.session_state.llm)
    return st.session_state[key]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None
if "active_source_type" not in st.session_state:
    st.session_state.active_source_type = None
if "source_content_path" not in st.session_state:
    st.session_state.source_content_path = None 


# 4.PDF 
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'''
    <div class="pdf-container">
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700px" type="application/pdf" style="border:none;"></iframe>
    </div>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)


# 5. UI Logic

if st.session_state.active_source_type is None:
    
    st.markdown("""
    <div class="hero">
        <h1>Chat with any <span>file</span>, <span>video</span> or <span>website</span></h1>
        <p>Add a source, and every answer comes straight from it —<br>with the passages used.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_pdf, tab_web, tab_youtube = st.tabs(["📄 PDF", "🌐 Website", "▶️ YouTube"])

    with tab_pdf:
        st.write("")
        uploaded_file = st.file_uploader("Drop a PDF here", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None and llm_ready:
            st.write("")
            if st.button("Index PDF File", key="index_pdf"):
                with st.spinner("Processing PDF..."):
                    save_path = Path("./uploaded_pdfs")
                    save_path.mkdir(exist_ok=True)
                    file_path = save_path / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        engine = get_engine("pdf")
                        result = engine.create_vectorstore(file_path)
                        st.session_state.active_document_id = result["document_id"]
                        st.session_state.active_source_type = "pdf"
                        st.session_state.source_content_path = file_path
                        st.rerun() 
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab_web:
        st.write("")
        url_input = st.text_input("Website URL", placeholder="https://example.com/article", label_visibility="collapsed")
        if url_input and llm_ready:
            st.write("")
            if st.button("Index Website", key="index_web"):
                with st.spinner("Fetching content..."):
                    try:
                        engine = get_engine("web")
                        result = engine.create_vectorstore(url_input.strip())
                        st.session_state.active_document_id = result["document_id"]
                        st.session_state.active_source_type = "web"
                        st.session_state.source_content_path = url_input.strip()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab_youtube:
        st.write("")
        yt_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", label_visibility="collapsed")
        if yt_url and llm_ready:
            st.write("")
            if st.button("Index Video", key="index_youtube"):
                with st.spinner("Fetching transcript..."):
                    try:
                        engine = get_engine("youtube")
                        result = engine.create_vectorstore(yt_url.strip())
                        st.session_state.active_document_id = result["document_id"]
                        st.session_state.active_source_type = "youtube"
                        st.session_state.source_content_path = yt_url.strip()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

else:
    col_chat, col_source = st.columns([5, 5], gap="large")

    with col_source:
        st.markdown('<h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 0;"><span>📄</span> Source Preview</h3>', unsafe_allow_html=True)
        st.write("")
        
        if st.session_state.active_source_type == "pdf":
            display_pdf(st.session_state.source_content_path)
            
        elif st.session_state.active_source_type == "youtube":
            st.video(st.session_state.source_content_path)
            
        elif st.session_state.active_source_type == "web":
            st.info(f"🌐 **Website URL:** {st.session_state.source_content_path}")
            st.markdown(f'<iframe src="{st.session_state.source_content_path}" width="100%" height="700px" style="border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px;"></iframe>', unsafe_allow_html=True)

    with col_chat:
        col_title, col_btn = st.columns([3, 1])
        with col_title:
            st.markdown('<h3 style="display: flex; align-items: center; gap: 10px; margin-bottom: 0;"><span>💬</span> Chat</h3>', unsafe_allow_html=True)
        with col_btn:
            if st.button("🔄 New Source"):
                st.session_state.active_source_type = None
                st.session_state.active_document_id = None
                st.session_state.source_content_path = None
                st.session_state.chat_history = []
                st.rerun()
        
        st.caption(f"Currently chatting with: **{st.session_state.active_source_type.upper()}**")
        st.divider()

        chat_container = st.container(height=480, border=False)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        question = st.chat_input("Ask your question here...")
        
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            engine = get_engine(st.session_state.active_source_type)
                            result = engine.ask(question, document_id=st.session_state.active_document_id)
                            answer = result["answer"]
                            st.markdown(answer)

                            with st.expander("📚 Sources"):
                                for src in result["sources"]:
                                    st.markdown(f"- {src.get('content', '')[:200]}...")

                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        except Exception as e:
                            st.error(f"Error: {e}")