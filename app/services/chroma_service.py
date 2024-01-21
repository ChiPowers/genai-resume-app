
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_docs(doc_path):
	loader = PyPDFDirectoryLoader(doc_path)
	documents = loader.load()
	return documents


def embed_chunks_and_upload_to_chroma(chunks, db_path):
	# Set Up Embeddings
	embeddings = OpenAIEmbeddings()

	# Store Text in ChromaDB
	Chroma.from_documents(chunks, embeddings)
	return


def get_most_similar_chunks_for_query(db_path):
	# Set Up Embeddings
	embeddings = OpenAIEmbeddings()
	vectordb = Chroma(embedding_function=embeddings)
	retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 6})
	return retriever

