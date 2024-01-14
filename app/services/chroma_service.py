

import chromadb
from app.utils.helper_functions import format_docs
import os
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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
	return


def get_most_similar_chunks_for_query(query, rag_pdf_path):
	client = OpenAI(
		api_key=os.environ.get("OPENAI_API_KEY"),
	)
	# Set Up Embeddings
	embeddings = OpenAIEmbeddings()
	vectordb = Chroma(persist_directory=rag_pdf_path, embedding_function=embeddings)
	retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
	return retriever
