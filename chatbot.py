import streamlit as st
from streamlit_option_menu import option_menu
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from firebase_admin import credentials
from firebase_admin import auth
from dotenv import load_dotenv
from urllib.parse import urlparse, unquote
import os
import json
import requests
import tempfile
from functools import partial
import datetime
import pytz
import mimetypes
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

### Functions: Start ###

def fetch_token_data():
    """Fetch the token data from Firestore."""
    try:
        token_ref = st.session_state.db.collection('Token').limit(1)
        token_docs = token_ref.get()
        
        if not token_docs:
            st.error("No token document found in Firestore.")
            return None, None
        
        token_doc = None
        doc_id = None
        for doc in token_docs:
            token_doc = doc.to_dict()
            doc_id = doc.id
            break

        if not token_doc:
            st.error("No token document found in Firestore.")
            return None, None

        return token_doc, doc_id
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None, None

def load_creds():
    token_doc, doc_id = fetch_token_data()

    if token_doc:
        account = token_doc.get("account")
        client_id = token_doc.get("client_id")
        client_secret = token_doc.get("client_secret")
        expiry = token_doc.get("expiry")
        refresh_token = token_doc.get("refresh_token")
        scopes = token_doc.get("scopes")
        token = token_doc.get("token")
        token_uri = token_doc.get("token_uri")
        universe_domain = token_doc.get("universe_domain")

        st.session_state['account'] = account
        st.session_state['client_id'] = client_id
        st.session_state['client_secret'] = client_secret
        st.session_state['expiry'] = expiry
        st.session_state['refresh_token'] = refresh_token
        st.session_state['scopes'] = scopes
        st.session_state['token'] = token
        st.session_state['token_uri'] = token_uri
        st.session_state['universe_domain'] = universe_domain

        temp_dir = tempfile.mkdtemp()
        token_file_path = os.path.join(temp_dir, 'token.json')
        with open(token_file_path, 'w') as token_file:
            json.dump(token_doc, token_file)

        creds = Credentials.from_authorized_user_file(token_file_path, scopes)

        if creds.expired:
            token_doc, _ = fetch_token_data()
            if token_doc:
                new_refresh_token = token_doc.get("refresh_token")
                if creds.refresh_token and creds.refresh_token == new_refresh_token:
                    st.toast("Refreshing token...")
                    creds.refresh(Request())
                    new_token_data = {
                        "account": account,
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "expiry": creds.expiry.isoformat() if creds.expiry else expiry,
                        "refresh_token": creds.refresh_token,
                        "scopes": scopes,
                        "token": creds.token,
                        "token_uri": token_uri,
                        "universe_domain": universe_domain
                    }
                    
                    st.session_state.db.collection('Token').document(doc_id).set(new_token_data)
                    st.session_state.update(new_token_data)
                    with open(token_file_path, 'w') as token_file:
                        json.dump(new_token_data, token_file)
                else:
                    st.error("Refresh token mismatch or missing. Please log in again.")
                    return None
            else:
                st.error("Failed to re-fetch token data from Firestore.")
                return None
    else:
        return None

    return creds

def extract_and_parse_json(text):
    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None, False

    json_str = text[start_index:end_index + 1]

    try:
        parsed_json = json.loads(json_str)
        return parsed_json, True
    except json.JSONDecodeError:
        return None, False

def is_expected_json_content(json_data):
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
    except json.JSONDecodeError:
        return False
    
    required_keys = ["Is_Answer_In_Context", "Answer"]

    if not all(key in data for key in required_keys):
        return False
    
    return True

def get_generative_model(response_mime_type="text/plain"):
    generation_config = {
        "temperature": 0.4,
        "top_p": 1,
        "max_output_tokens": 8192,
        "response_mime_type": response_mime_type
    }

    if st.session_state["oauth_creds"] is not None:
        genai.configure(credentials=st.session_state["oauth_creds"])
    else:
        st.session_state["oauth_creds"] = load_creds()
        genai.configure(credentials=st.session_state["oauth_creds"])

    model = genai.GenerativeModel('tunedModels/connext-wide-chatbot-ddal5ox9d38h', generation_config=generation_config) if response_mime_type == "text/plain" else genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    return model

def generate_response(question, context, fine_tuned_knowledge=False):
    prompt_using_fine_tune_knowledge = f"""
    Based on your base or fine-tuned knowledge, can you answer the the following question?

    --------------------

    Question:
    {question}

    --------------------

    Answer:

    """
    prompt_with_context = f"""

    Answer the question below as detailed as possible from the provided context below, make sure to provide all the details but if the answer is not in
    provided context. Try not to make up an answer just for the sake of answering a question.

    --------------------
    Context:
    {context}

    --------------------

    Question:
    {question}
    
    Provide your answer in a json format following the structure below:
    {{
        "Is_Answer_In_Context": <boolean>,
        "Answer": <answer (string)>,
    }}
    """

    prompt = prompt_using_fine_tune_knowledge if fine_tuned_knowledge else prompt_with_context
    model = get_generative_model("text/plain" if fine_tuned_knowledge else "application/json")
    
    return model.generate_content(prompt).text

def try_get_answer(user_question, context="", fine_tuned_knowledge=False):
    parsed_result = {}
    if not fine_tuned_knowledge:
        response_json_valid = False
        is_expected_json = False
        max_attempts = 3
        while not response_json_valid and max_attempts > 0:
            response = ""

            try:
                response = generate_response(user_question, context, fine_tuned_knowledge)
            except Exception as e:
                max_attempts -= 1
                st.toast(f"Failed to create a response for your query.\n Error Code: {str(e)} \nTrying again... Retries left: {max_attempts} attempt/s")
                continue

            parsed_result, response_json_valid = extract_and_parse_json(response)
            if not response_json_valid:
                max_attempts -= 1
                st.toast(f"Failed to validate and parse json for your query.\n Trying again... Retries left: {max_attempts} attempt/s")
                continue

            is_expected_json = is_expected_json_content(parsed_result)
            if not is_expected_json:
                max_attempts -= 1
                st.toast(f"Successfully validated and parse json for your query.\n Trying again... Retries left: {max_attempts} attempt/s")
                continue

            break
    else:
        try:
            parsed_result = generate_response(user_question, context, fine_tuned_knowledge)
        except Exception as e:
            st.toast(f"Failed to create a response for your query.")

    return parsed_result

def user_input(user_question, api_key):
    with st.spinner("Processing..."):
        st.session_state.show_fine_tuned_expander = True
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        context = "\n\n--------------------------\n\n".join([doc.page_content for doc in docs])

        parsed_result = try_get_answer(user_question, context)
    
    return parsed_result

def app():

    google_ai_api_key = st.session_state["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"]
    firestore_db = firestore.client()
    st.session_state.db = firestore_db

    col1, col2, col3 = st.columns([3, 4, 3])

    with col1:
        st.write(' ')

    with col2:
        st.image("Connext_Logo.png", width=250)

    with col3:
        st.write(' ')

    st.markdown('## Welcome to :blue[Connext Chatbot] :robot_face:')

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'parsed_result' not in st.session_state:
        st.session_state.parsed_result = {}

    chat_history_placeholder = st.empty()

    def display_chat_history():
        with chat_history_placeholder.container():
            for chat in st.session_state.chat_history:
                st.markdown(f"🧑 **You:** {chat['question']}")
                st.markdown(f"🤖 **Bot:** {chat['answer']['Answer']}")

    display_chat_history()

    user_question = st.text_input("Ask a Question", key="user_question")
    submit_button = st.button("Submit", key="submit_button")
    clear_history_button = st.button("Clear Chat History")

    if clear_history_button:
        st.session_state.chat_history = []
        display_chat_history()

    if "answer" not in st.session_state:
        st.session_state["answer"] = ""

    if "request_fine_tuned_answer" not in st.session_state:
        st.session_state["request_fine_tuned_answer"] = False

    if 'fine_tuned_answer_expander_state' not in st.session_state:
        st.session_state.fine_tuned_answer_expander_state = False

    if 'show_fine_tuned_expander' not in st.session_state:
        st.session_state.show_fine_tuned_expander = False

    if submit_button:
        if user_question and google_ai_api_key:
            parsed_result = user_input(user_question, google_ai_api_key)
            st.session_state.parsed_result = parsed_result
            if "Answer" in parsed_result:
                st.session_state.chat_history.append({"question": user_question, "answer": parsed_result})
                display_chat_history()
                if "Is_Answer_In_Context" in parsed_result and not parsed_result["Is_Answer_In_Context"]:
                    st.session_state.show_fine_tuned_expander = True
            else:
                st.toast("Failed to get a valid response from the model.")

    display_chat_history()

    if st.session_state.show_fine_tuned_expander:
        with st.expander("Get fine-tuned answer?", expanded=True):
            st.write("Would you like me to generate the answer based on my fine-tuned knowledge?")
            col1, col2, _ = st.columns([1, 1, 1])
            with col1:
                if st.button("Yes", key=f"yes_button"):
                    st.session_state.request_fine_tuned_answer = True
                    st.session_state.show_fine_tuned_expander = False
                    st.rerun()
            with col2:
                if st.button("No", key=f"no_button"):
                    st.session_state.show_fine_tuned_expander = False
                    st.rerun()

    if st.session_state["request_fine_tuned_answer"]:
        if st.session_state.chat_history:
            with st.spinner("Generating fine-tuned answer..."):
                fine_tuned_result = try_get_answer(st.session_state.chat_history[-1]['question'], context="", fine_tuned_knowledge=True)
            if fine_tuned_result:
                st.session_state.chat_history[-1]['answer'] = {"Answer": fine_tuned_result.strip()}
                display_chat_history()
            else:
                st.toast("Failed to generate a fine-tuned answer.")
        st.session_state["request_fine_tuned_answer"] = False

if __name__ == "__main__":
    app()
