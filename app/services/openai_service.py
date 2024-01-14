# YourApp/services/openai_service.py

import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.utils.helper_functions import format_docs


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_EMBEDDING_MODEL = "gpt-3.5-turbo"


def get_llm_answer(prompt, retriever, question):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.2)
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = rag_chain.invoke(question)
    return response
