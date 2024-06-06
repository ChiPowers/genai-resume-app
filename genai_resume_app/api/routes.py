# fullstack-resume/genai_resume_app/api/routes.py
import os
from . import api_blueprint
from flask import request, jsonify, render_template
from genai_resume_app.services import openai_service, chroma_service
from genai_resume_app.utils.helper_functions import split_docs, build_prompt, load_docs, session_first_embed_and_store

@api_blueprint.route('/embed-and-store', methods=['GET'])
def embed_and_store():
	texts = load_docs(os.environ.get("rag_pdf_path"))
	chunks = split_docs(texts)
	chroma_service.embed_chunks_and_upload_to_chroma(chunks, os.environ.get("db_path"))
	return "Document chunks embedded successfully"


@api_blueprint.route('/handle-query', methods=['POST'])
def handle_query() -> 'html':
	# handles embedding the user's question
	question = request.form['the_question']
	retriever = chroma_service.get_most_similar_chunks_for_query(os.environ.get("db_path"))
	prompt = build_prompt()
	answer = openai_service.get_llm_answer(prompt, retriever, question)
	print(answer)
	return render_template('handle_query.html',
						 the_title='Interview Me',
						 the_question=question,
						 the_answer=answer,
						 )


@api_blueprint.route('/')
@api_blueprint.route('/get-query', methods=['GET','POST'])
def get_query():
	return render_template('get_query.html',
						the_title='Ask an Interview Question')
