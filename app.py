import streamlit as st
from rag_system import RAGSystem
from mcq_generator import generate_mcq

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DL RAG Learning Assistant",
    page_icon="🧠",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ System Status")
    st.success("✅ Vector DB Loaded")
    st.success("✅ Groq OSS-120B Connected")

    st.markdown("---")
    st.markdown("### 📚 Features")
    st.markdown("""
    - Citation-based answers  
    - Course-restricted responses  
    - Semantic search (FAISS)  
    - MCQ generation  
    - Zero hallucination  
    """)

    st.markdown("---")
    st.markdown("### 🎓 Project Info")
    st.markdown("""
    **Course:** Deep Learning  
    **Architecture:** RAG  
    **LLM:** Groq OSS-120B  
    """)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;'>🧠 Deep Learning RAG Assistant</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Answers strictly from your course PDFs</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- LOAD SYSTEM ----------------
@st.cache_resource
def load_rag():
    rag = RAGSystem()
    rag.load_database()
    return rag

rag = load_rag()

# ---------------- QUESTION INPUT ----------------
st.markdown("### ❓ Ask a Question")

question = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="e.g. What is Deep Learning?"
)

col1, col2 = st.columns([1, 1])

with col1:
    ask_btn = st.button("🔍 Get Answer")

with col2:
    mcq_btn = st.button("📝 Generate MCQs")

# ---------------- ANSWER ----------------
if ask_btn:
    if question.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🔎 Searching course materials..."):
            result = rag.query(question)

        st.markdown("### 💡 Answer")
        st.success(result["answer"])

        st.markdown("### 📚 Citations")
        for i, src in enumerate(result["sources"], 1):
            with st.expander(f"📄 Source {i}: {src['source']} (Page {src['page']})"):
                st.write(src["content"])

# ---------------- MCQ GENERATION ----------------
if mcq_btn:
    with st.spinner("🧠 Generating MCQs..."):
        base = rag.query(question if question else "Summarize key concepts")
        mcqs = generate_mcq(base["answer"])

    st.markdown("### 📝 Generated MCQs")
    st.write(mcqs)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "DL-RAG Assistant | FAISS + Groq OSS-120B | Streamlit"
    "</div>",
    unsafe_allow_html=True
)

