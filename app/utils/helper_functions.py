# YourApp/utils/helper_functions.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_docs(path_to_pdfs):
    loader = PyPDFDirectoryLoader(path_to_pdfs)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=66, chunk_overlap=20):
    # Split Text into Manageable Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    return texts


def build_prompt(query, context_chunks):
    prompt_limit = 3750
    prompt_start = ("You are a potential employer reviewing the work history /"
        "and accomplishments of Dr. Chivon E. Powers. Use the context below /"
        "to answer questions asked by an interviewer who is interviewing /"
        "Chivon for an AI engineering role in the tech industry.\n Context:\n ")
    prompt_end = f"\n\nQuestion: {query}\nAnswer:"

    # append context chunks until we hit the limit of tokens we want to send to the prompt.
    prompt = ""
    for i in range(1, len(context_chunks)):
        if len("\n\n---\n\n".join(context_chunks[:i])) >= prompt_limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(context_chunks[:i-1]) +
                prompt_end
            )
            break
        elif i == len(context_chunks)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(context_chunks) +
                prompt_end
            )
    return prompt

