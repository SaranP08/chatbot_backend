# rag_chatbot.py

import os
from huggingface_hub import hf_hub_download
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
# <<< CHANGED: Import from the new, correct package
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from flashrank import Ranker
from typing import Dict, Any, List

# This is a one-time operation for flashrank, it's fine to keep it here.
# FlashrankRerank.model_rebuild()

# --- Configuration ---
PDF_PATH = "data/sat2farm_doc.pdf"
# <<< REMOVED: This constant is no longer needed here as the model is passed in.
# EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5" 
MODEL_NAME = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

class RAGChatBot:
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        """
        Initializes the RAG ChatBot by accepting a pre-loaded embedding model.
        This is an efficient approach to ensure a single model instance is used.
        """
        print("Initializing RAG ChatBot...")
        
        print(f"Downloading GGUF model: {MODEL_NAME}/{MODEL_FILE}")
        model_path = hf_hub_download(repo_id=MODEL_NAME, filename=MODEL_FILE)
        
        # Use the provided embedding model to build the chain.
        self.chain = self._create_rag_chain(model_path, embeddings)
        print("\n✅ RAG ChatBot initialized successfully!")

    def _create_rag_chain(self, model_path: str, embeddings: HuggingFaceEmbeddings):
        """Builds and returns the full RAG chain."""
        print("--- Building new retrievers from PDF ---")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""], length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} document chunks.")

        # STAGE 1: RECALL (Ensemble Retriever)
        print("Initializing Stage 1 Retriever (Ensemble)...")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 10

        # This line is key: it uses the centrally-managed `embeddings` object.
        vectorstore = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5], search_type="rrf"
        )

        # STAGE 2: RE-RANK (Flashrank)
        print("Initializing Stage 2 Re-ranker (Flashrank)...")
        compressor = FlashrankRerank(top_n=3)
        final_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        print(f"Initializing GGUF LLM from: {model_path}")
        llm = LlamaCpp(
            model_path=model_path, n_ctx=2048, n_gpu_layers=-1, # Use -1 for all layers on GPU
            temperature=0.0, top_k=1, verbose=False, max_tokens=512
        )

        prompt_template = """
You are a factual question-answering assistant. Your task is to answer the user's query based ONLY on the provided text snippets.
Follow these rules strictly:
1. Provide only the direct answer to the question and nothing else. DO NOT add any summary, conclusion, or other extra information.
2. If the answer is not in the context, state that you do not have that information.

Context: {context}
Question: {question}
Helpful Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        print("Creating RAG chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=final_retriever,
            return_source_documents=True, chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Asks a question to the RAG chain and returns the answer and source documents.
        """
        if not self.chain:
            return {"answer": "RAG chain is not initialized.", "sources": []}
            
        print(f"Invoking RAG chain with query: '{query}'")
        result = self.chain.invoke(query)
        
        # Clean up the answer
        answer = result.get("result", "").strip()
        if "Helpful Answer:" in answer:
            answer = answer.split("Helpful Answer:")[1].strip()

        # Format sources
        sources = []
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content,
                    "page": doc.metadata.get('page', 'N/A')
                })
        
        return {"answer": answer, "sources": sources}