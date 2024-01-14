# YourApp/utils/helper_functions.py

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_docs(path_to_pdfs):
    loader = PyPDFDirectoryLoader(path_to_pdfs)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    # Split Text into Manageable Chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False)
    texts = text_splitter.split_documents(documents)
    return texts


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt():
    from langchain_core.prompts import ChatPromptTemplate
    template = """
    You are interviewing for an Applied AI scientist position at a tech company.\
        Use the following context to answer interview questions in a way that describes how your experience \
        and skills relate to the job requirements. Use no more than 2 short sentences.\

        Context: {context} \
        Interview Question: \
        {query} \
        Answer:
        """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

