# YourApp/utils/helper_functions.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_docs(path_to_pdfs):
    loader = PyPDFDirectoryLoader(path_to_pdfs)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=100):
    # Split Text into Manageable Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    template = """
    You are interviewing Dr. Chivon Powers to understand her work experience\
    and accomplishments. \
    Use the following context to answer interview questions as Chivon would respond during \
    an interview for an AI engineering role in the tech industry. Context: {context} \
    Interviewer Question: \
    {query} \
    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

