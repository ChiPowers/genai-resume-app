

import chromadb
from app.services.openai_service import get_embedding
import os
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_docs(doc_path):
	loader = PyPDFDirectoryLoader(doc_path)
	documents = loader.load()
	return documents


def embed_chunks_and_upload_to_chroma(chunks, db_path):
	client = OpenAI(
		api_key=os.environ.get("OPENAI_API_KEY"),
	)
	# Set Up Embeddings
	embeddings = OpenAIEmbeddings()

	# Store Text in ChromaDB
	Chroma.from_documents(chunks, embeddings, persist_directory=db_path)


def get_most_similar_chunks_for_query(query, rag_pdf_path):
	client = OpenAI(
		api_key=os.environ.get("OPENAI_API_KEY"),
	)
	# Set Up Embeddings
	embeddings = OpenAIEmbeddings()
	vectordb = Chroma(persist_directory=rag_pdf_path, embedding_function=embeddings)
	context_chunks = vectordb.similarity_search(query)
	return context_chunks
