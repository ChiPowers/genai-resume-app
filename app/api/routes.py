# fullstack-resume/app/api/routes.py
import os
from . import api_blueprint
from flask import request, jsonify
from app.services import openai_service, chroma_service
from app.utils.helper_functions import split_docs, build_prompt, load_docs, session_first_embed_and_store



@api_blueprint.route('/embed-and-store', methods=['POST'])
def embed_and_store():
	texts = load_docs(os.environ.get("rag_pdf_path"))
	chunks = split_docs(texts)
	chroma_service.embed_chunks_and_upload_to_chroma(chunks, os.environ.get("db_path"))
	response_json = {
		"message": "Document chunks embedded successfully"
	}
	return jsonify(response_json)


@api_blueprint.route('/handle-query', methods=['POST'])
def handle_query():
	session_first_embed_and_store()
	# handles embedding the user's question
	question = request.json['question']
	retriever = chroma_service.get_most_similar_chunks_for_query(os.environ.get("db_path"))
	prompt = build_prompt()
	answer = openai_service.get_llm_answer(prompt, retriever, question)
	return jsonify({"question:": question, "answer:": answer})
