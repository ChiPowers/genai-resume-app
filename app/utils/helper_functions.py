# YourApp/utils/helper_functions.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_docs(path_to_pdfs):
    loader = PyPDFDirectoryLoader(path_to_pdfs)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=660, chunk_overlap=90):
    # Split Text into Manageable Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt(query, context_chunks):
    from langchain_core.prompts import ChatPromptTemplate
    context = format_docs(context_chunks)
    template = f'You are a potential employer reviewing the work history \
    and accomplishments of Dr. Chivon E. Powers. \
    Use these documents to answer questions asked by an interviewer who is interviewing Chivon \
    for an ai engineer role in the tech industry. Context: {context} \
    Interviewer Question: \
    {query} \
    Answer:'

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

