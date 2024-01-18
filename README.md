# genai-resume-app

This application uses LLM with retrieval augmented generation (RAG) to answer interview questions about the career history and experience listed on the resume. 

LLM used is GPT3.5-Turbo from Open AI. An API key from OpenAI is required.

To run this app locally, you will need an .env file that provides the following environment variables:

OPENAI_API_KEY="<your-api-key>"
rag_pdf_path = "<path-to-pdf-of-resume-on-local-machine"
db_path = "<path-to-chroma-data-folder-where-embeddings-db-is-stored>"


Put any resume into the pdfs folder. 

First use the 'embed-and-store' call to do the document retrieval and store as embeddings in the Chroma db. 
Example: 
    
    curl -X POST http://localhost:5000/embed-and-store \
    -H "Content-Type: application/json"

Then use the 'handle-query' call to ask questions about the career history represented in the resume. 
Example:

    curl -X POST http://localhost:5000/handle-query \
     -H "Content-Type: application/json" \
     -d '{"question":"What is your most recent job?"}'

Tip: A good way to check that it is properly capturing the resume is to ask "What is your name?"
