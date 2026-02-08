"""
Core RAG System - OpenRouter.ai Version (FREE!)
"""
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


class RAGSystem:
    def __init__(self):
        load_dotenv()

        self.db_path = "vectordb"

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_db = None
        self.qa_chain = None

        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-120b",
            temperature=0
        )

    def load_database(self):
        print("📚 Loading vector database...")

        self.vector_db = FAISS.load_local(
            self.db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

        print("✅ QA system ready!")

    def query(self, question):
        if not self.qa_chain:
            raise ValueError("Database not loaded!")

        result = self.qa_chain.invoke({"query": question})

        # Extract sources
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:300] + "..."
            })

        return {
            "answer": result["result"],
            "sources": sources
        }
