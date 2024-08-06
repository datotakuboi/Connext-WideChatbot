import streamlit as st
from streamlit_option_menu import option_menu
import firebase_admin
from firebase_admin import firestore, credentials
from google.cloud import storage
from urllib.parse import urlparse, unquote
import os
import tempfile
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import datetime
import requests
import json

### Initialize Session State ###
if "oauth_creds" not in st.session_state:
    st.session_state["oauth_creds"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "conversation_context" not in st.session_state:
    st.session_state["conversation_context"] = ""

### Initialize Firebase SDK ###
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["service_account"]))
    firebase_admin.initialize_app(cred)

SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

### Helper Functions ###
def google_oauth_link(flow):
    auth_url, _ = flow.authorization_url(prompt='consent')
    st.write("Please go to this URL and authorize access:")
    st.markdown(f"[Sign in with Google]({auth_url})", unsafe_allow_html=True)
    return st.text_input("Enter the authorization code:")

def load_creds():
    token_info = st.secrets["token"]
    temp_dir = tempfile.mkdtemp()
    token_file_path = os.path.join(temp_dir, 'token.json')
    with open(token_file_path, 'w') as token_file:
        json.dump(token_info, token_file)
    creds = Credentials.from_authorized_user_file(token_file_path, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(token_file_path, 'w') as token_file:
            token_file.write(creds.to_json())
    return creds

def download_file_to_temp(url):
    storage_client = storage.Client.from_service_account_info(st.session_state["connext_chatbot_admin_credentials"])
    bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')
    temp_file_path = os.path.join(tempfile.mkdtemp(), os.path.basename(unquote(urlparse(url).path)))
    bucket.blob(os.path.basename(urlparse(url).path)).download_to_filename(temp_file_path)
    return temp_file_path, os.path.basename(urlparse(url).path)

def extract_and_parse_json(text):
    start_index, end_index = text.find('{'), text.rfind('}')
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None, False
    try:
        return json.loads(text[start_index:end_index + 1]), True
    except json.JSONDecodeError:
        return None, False

def is_expected_json_content(json_data):
    try:
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
    except json.JSONDecodeError:
        return False
    return all(key in data for key in ["Is_Answer_In_Context", "Answer"])

def get_pdf_text(pdf_files):
    return "".join(extract_text(pdf) for pdf in pdf_files)

def get_text_chunks(text):
    return RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(text)

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_generative_model(response_mime_type="text/plain"):
    if not st.session_state.get("oauth_creds"):
        st.session_state["oauth_creds"] = load_creds()
    genai.configure(credentials=st.session_state["oauth_creds"])
    generation_config = {"temperature": 0.4, "top_p": 1, "max_output_tokens": 8192, "response_mime_type": response_mime_type}
    model_name = 'tunedModels/connext-wide-chatbot-ddal5ox9d38h' if response_mime_type == "text/plain" else "gemini-1.5-flash"
    return genai.GenerativeModel(model_name, generation_config=generation_config)

def generate_response(question, context, fine_tuned_knowledge=False):
    prompt = (
        f"Based on your base or fine-tuned knowledge, can you answer the following question?\n\nQuestion:\n{question}\n\nAnswer:\n"
        if fine_tuned_knowledge else
        f"Answer the question below as detailed as possible from the provided context below, make sure to provide all the details but if the answer is not in provided context. Try not to make up an answer just for the sake of answering a question.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nProvide your answer in a json format following the structure below:\n{{\n\"Is_Answer_In_Context\": <boolean>,\n\"Answer\": <answer (string)>,\n}}"
    )
    return get_generative_model("text/plain" if fine_tuned_knowledge else "application/json").generate_content(prompt).text

def try_get_answer(user_question, context="", fine_tuned_knowledge=False):
    max_attempts, parsed_result = 3, {}
    for _ in range(max_attempts):
        try:
            response = generate_response(user_question, context, fine_tuned_knowledge)
            parsed_result, response_json_valid = extract_and_parse_json(response)
            if response_json_valid and is_expected_json_content(parsed_result):
                return parsed_result
        except Exception as e:
            st.toast(f"Error: {str(e)}. Retrying...")
    return parsed_result

def user_input(user_question, api_key):
    with st.spinner("Processing..."):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        context = "\n\n--------------------------\n\n".join([doc.page_content for doc in docs])
        return try_get_answer(user_question, context)

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.conversation_context = ""

def app():
    st.set_page_config(page_title="Connext Chatbot", layout="centered")

    google_ai_api_key = st.secrets["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"]
    if not google_ai_api_key:
        st.error("Google API key is missing. Please provide it in the secrets configuration.")
        return

    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate(st.secrets["connext_chatbot_admin_credentials"]))
    st.session_state.db = firestore.client()

    # UI Setup
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
    
    def display_chat_history():
        with st.empty().container():
            for chat in st.session_state.chat_history:
                st.markdown(f"🧑 **You:** {chat['user_question']}")
                st.markdown(f"🤖 **Bot:** {chat['response']}")

    display_chat_history()

    user_question = st.text_input("Ask a Question", key="user_question")
    if st.button("Submit", key="submit_button"):
        if user_question and google_ai_api_key:
            parsed_result = user_input(user_question, google_ai_api_key)
            st.session_state.parsed_result = parsed_result
            if "Answer" in parsed_result:
                st.session_state.chat_history.append({"user_question": user_question, "response": parsed_result["Answer"]})
                display_chat_history()
                if "Is_Answer_In_Context" in parsed_result and not parsed_result["Is_Answer_In_Context"]:
                    st.session_state.show_fine_tuned_expander = True
            else:
                st.toast("Failed to get a valid response from the model.")

    if st.button("Clear Chat History"):
        clear_chat()
        display_chat_history()

    if st.session_state.get("show_fine_tuned_expander"):
        with st.expander("Get fine-tuned answer?", expanded=True):
            st.write("Would you like me to generate the answer based on my fine-tuned knowledge?")
            col1, col2, _ = st.columns([1, 1, 1])
            with col1:
                if st.button("Yes", key="yes_button"):
                    st.session_state.request_fine_tuned_answer = True
                    st.session_state.show_fine_tuned_expander = False
                    st.rerun()
            with col2:
                if st.button("No", key="no_button"):
                    st.session_state.show_fine_tuned_expander = False
                    st.rerun()

    if st.session_state.get("request_fine_tuned_answer"):
        fine_tuned_result = try_get_answer(st.session_state.chat_history[-1]['user_question'], fine_tuned_knowledge=True)
        if fine_tuned_result:
            st.session_state.chat_history[-1]['response'] = fine_tuned_result.strip()
            st.session_state.show_fine_tuned_expander = False
            st.session_state.parsed_result['Answer'] = fine_tuned_result.strip()
        else:
            st.toast("Failed to generate a fine-tuned answer.")
        st.session_state["request_fine_tuned_answer"] = False

    with st.sidebar:
        st.title("PDF Documents:")
        retrievers_ref = st.session_state.db.collection('Retrievers')
        docs = retrievers_ref.stream()
        for idx, doc in enumerate(docs, start=1):
            retriever = doc.to_dict()
            retriever['id'] = doc.id
            retriever_name, retriever_description = retriever['retriever_name'], retriever['retriever_description']
            with st.expander(retriever_name):
                st.markdown(f"**Description:** {retriever_description}")
                file_path, file_name = download_file_to_temp(retriever['document'])
                st.markdown(f"_**File Name**_: {file_name}")
                retriever["file_path"] = file_path
                st.session_state["retrievers"][retriever_name] = retriever
        st.title("PDF Document Selection:")
        st.session_state["selected_retrievers"] = st.multiselect("Select Documents", list(st.session_state["retrievers"].keys()))
        if st.button("Submit & Process", key="process_button"):
            if google_ai_api_key:
                with st.spinner("Processing..."):
                    selected_files = [st.session_state["retrievers"][name]["file_path"] for name in st.session_state["selected_retrievers"]]
                    raw_text = get_pdf_text(selected_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, google_ai_api_key)
                    st.success("Done")
            else:
                st.toast("Failed to process the documents", icon="💥")

if __name__ == "__main__":
    app()
