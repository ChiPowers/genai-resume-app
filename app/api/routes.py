# fullstack-resume/app/api/routes.py
from . import api_blueprint
from flask import request, jsonify
from app.services import openai_service, chroma_service, scraping_service
from app.utils.helper_functions import split_docs, build_prompt, load_docs

db_path = "/Users/chivonpowers/chroma_data/"
rag_pdf_path = "/Users/chivonpowers/Downloads/Resume_Docs/pdfs/"

@api_blueprint.route('/embed-and-store', methods=['POST'])
def embed_and_store():
	texts = load_docs(rag_pdf_path)
	chunks = split_docs(texts)
	chroma_service.embed_chunks_and_upload_to_chroma(chunks, db_path)
	response_json = {
		"message": "Document chunks embedded and stored successfully."
	}
	return jsonify(response_json)


@api_blueprint.route('/handle-query', methods=['POST'])
def handle_query():
	# handles embedding the user's question
	question = request.json['question']
	context_chunks = chroma_service.get_most_similar_chunks_for_query(question, rag_pdf_path)
	prompt = build_prompt(question, context_chunks)
	# answer = openai_service.get_llm_answer(prompt)
	return jsonify({"question": question, "prompt": str(prompt), "context": context_chunks})

