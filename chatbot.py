import streamlit as st
from streamlit_option_menu import option_menu
import firebase_admin
from firebase_admin import firestore
from google.cloud import storage
from firebase_admin import credentials
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

# Initialize session_state values
if "oauth_creds" not in st.session_state:
    st.session_state["oauth_creds"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "conversation_context" not in st.session_state:
    st.session_state["conversation_context"] = ""

# Initialize Firebase SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["service_account"]))
    firebase_admin.initialize_app(cred)

SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

@st.experimental_dialog("Google Consent Authentication Link")
def google_oauth_link(flow):
    auth_url, _ = flow.authorization_url(prompt='consent')
    st.write("Please go to this URL and authorize access:")
    st.markdown(f"[Sign in with Google]({auth_url})", unsafe_allow_html=True)
    code = st.text_input("Enter the authorization code:")
    return code

def load_creds():
    """Load credentials from Streamlit secrets and handle them using a temporary file."""
    token_info = {
        'token': st.secrets["token"]["value"],
        'refresh_token': st.secrets["token"]["refresh_token"],
        'token_uri': st.secrets["token"]["token_uri"],
        'client_id': st.secrets["token"]["client_id"],
        'client_secret': st.secrets["token"]["client_secret"],
        'scopes': st.secrets["token"]["scopes"],
        'expiry': st.secrets["token"]["expiry"]
    }

    temp_dir = tempfile.mkdtemp()
    token_file_path = os.path.join(temp_dir, 'token.json')
    with open(token_file_path, 'w') as token_file:
        json.dump(token_info, token_file)

    creds = Credentials.from_authorized_user_file(token_file_path, SCOPES)

    if creds and creds.expired and creds.refresh_token:
        st.toast("Currently refreshing token...")
        creds.refresh(Request())

        with open(token_file_path, 'w') as token_file:
            token_file.write(creds.to_json())

    return creds

def generate_signed_url(bucket_name, blob_name, service_account_info, expiration=3600):
    storage_client = storage.Client.from_service_account_info(service_account_info)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(expiration=datetime.timedelta(seconds=expiration))
    return url

def download_file_from_url(url):
    try:
        temp_dir = tempfile.mkdtemp()
        file_name = os.path.basename(urlparse(url).path)
        temp_file_path = os.path.join(temp_dir, file_name)

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(temp_file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            return temp_file_path, file_name
        else:
            st.error(f"Failed to download file: {response.status_code}")
            return None, None
    except Exception as e:
        st.error(f"Failed to download file: {e}")
        return None, None

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        extracted_text = extract_text(pdf)
        text += extracted_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

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

def get_generative_model(response_mime_type = "text/plain"):
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

    model = genai.GenerativeModel('tunedModels/connext-wide-chatbot-ddal5ox9d38h' ,generation_config=generation_config) if response_mime_type == "text/plain" else genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
    return model

def generate_response(question, context, fine_tuned_knowledge = False):
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

def try_get_answer(user_question, context="", fine_tuned_knowledge = False):
    parsed_result = {}
    if not fine_tuned_knowledge:
        response_json_valid = False
        is_expected_json = False
        max_attempts = 3
        while not response_json_valid and max_attempts > 0:
            response = ""
            try:
                response = generate_response(user_question, context , fine_tuned_knowledge)
            except Exception as e:
                max_attempts -= 1
                st.toast(f"Failed to create a response for your query.\n Error Code: {str(e)} \nTrying again... Retries left: {max_attempts} attempt/s")
                continue

            parsed_result, response_json_valid = extract_and_parse_json(response)
            if response_json_valid == False:
                max_attempts -= 1
                st.toast(f"Failed to validate and parse json for your query.\n Trying again... Retries left: {max_attempts} attempt/s")
                continue

            is_expected_json = is_expected_json_content(parsed_result)  
            if is_expected_json == False:
                max_attempts -= 1
                st.toast(f"Successfully validated and parse json for your query.\n Trying again... Retries left: {max_attempts} attempt/s")
                continue
            
            break
    else:
        try:
            parsed_result = generate_response(user_question, context , fine_tuned_knowledge)
        except Exception as e:
            parsed_result = ""
            st.toast(f"Failed to create a response for your query.")

    return parsed_result

def user_input(user_question, api_key):
    with st.spinner("Processing..."):
        st.session_state.show_fine_tuned_expander = True
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        context = "\n\n--------------------------\n\n".join([doc.page_content for doc in docs])

        # Combine previous conversation context with the current context
        full_context = f"{st.session_state.conversation_context}\n\n{context}"

        parsed_result = try_get_answer(user_question, full_context)
        if parsed_result:
            st.session_state.chat_history.append({
                "user_question": user_question,
                "response": parsed_result["Answer"] if "Answer" in parsed_result else "No response generated."
            })
            # Update the conversation context
            st.session_state.conversation_context += f"\n\nUser: {user_question}\nBot: {parsed_result['Answer']}"

    return parsed_result

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.conversation_context = ""

def clear_input():
    st.session_state["user_question"] = ""

def app():
    st.set_page_config(page_title="Connext Chatbot", layout="centered")

    google_ai_api_key = st.secrets["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"]

    if not google_ai_api_key:
        st.error("Google API key is missing. Please provide it in the secrets configuration.")
        return

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

    retrievers_ref = st.session_state.db.collection('Retrievers')
    docs = retrievers_ref.stream()

    chat_placeholder = st.empty()
    with chat_placeholder.container():
        for chat in st.session_state.chat_history:
            st.write(f"**You:** {chat['user_question']}")
            st.write(f"**Bot:** {chat['response']}")

    user_question = st.text_input("Ask a Question", key="user_question")
    submit_button = st.button("Submit", key="submit_button", on_click=clear_input)
    clear_button = st.button("Clear Chat History", on_click=clear_chat)

    if "retrievers" not in st.session_state:
        st.session_state["retrievers"] = {}
    
    if "selected_retrievers" not in st.session_state:
        st.session_state["selected_retrievers"] = []

    if "answer" not in st.session_state:
        st.session_state["answer"] = ""

    if "request_fine_tuned_answer" not in st.session_state:
        st.session_state["request_fine_tuned_answer"] = False

    if 'fine_tuned_answer_expander_state' not in st.session_state:
        st.session_state.fine_tuned_answer_expander_state = False

    if 'show_fine_tuned_expander' not in st.session_state:
        st.session_state.show_fine_tuned_expander = True

    if 'parsed_result' not in st.session_state:
        st.session_state.parsed_result = {}

    with st.sidebar:
        st.title("PDF Documents:")
        for idx, doc in enumerate(docs, start=1):
            retriever = doc.to_dict()
            retriever['id'] = doc.id
            retriever_name = retriever['retriever_name']
            retriever_description = retriever['retriever_description']
            with st.expander(retriever_name):
                st.markdown(f"**Description:** {retriever_description}")

                parsed_url = urlparse(retriever['document'])
                file_name = os.path.basename(unquote(parsed_url.path))
                signed_url = generate_signed_url('connext-chatbot-admin.appspot.com', file_name, st.secrets["service_account"])

                st.markdown(f"_**File Name**_: {file_name}")
                st.markdown(f"[Download PDF]({signed_url})", unsafe_allow_html=True)

                retriever["signed_url"] = signed_url
                st.session_state["retrievers"][retriever_name] = retriever

        st.title("PDF Document Selection:")
        st.session_state["selected_retrievers"] = st.multiselect("Select Retrievers", list(st.session_state["retrievers"].keys()))

        if st.button("Submit & Process", key="process_button"):
            if google_ai_api_key:
                with st.spinner("Processing..."):
                    selected_retrievers = st.session_state["selected_retrievers"]
                    downloaded_files = []
                    for name in selected_retrievers:
                        signed_url = st.session_state["retrievers"][name]["signed_url"]
                        file_path, _ = download_file_from_url(signed_url)
                        if file_path:
                            downloaded_files.append(file_path)
                    
                    raw_text = get_pdf_text(downloaded_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, google_ai_api_key)
                    st.success("Processing complete.")
            else:
                st.error("Google API key is missing. Please provide it in the secrets configuration.")

    if submit_button:
        if user_question and google_ai_api_key:
            st.session_state.parsed_result = user_input(user_question, google_ai_api_key)
            with chat_placeholder.container():
                for idx, chat in enumerate(st.session_state.chat_history):
                    st.write(f"**You:** {chat['user_question']}")
                    st.write(f"**Bot:** {chat['response']}")
                    if idx == len(st.session_state.chat_history) - 1:  # Check the last question
                        if "Is_Answer_In_Context" in st.session_state.parsed_result and not st.session_state.parsed_result["Is_Answer_In_Context"]:
                            if st.session_state.show_fine_tuned_expander:
                                with st.expander("Get fine-tuned answer?", expanded=False):
                                    st.write("Would you like me to generate the answer based on my fine-tuned knowledge?")
                                    col1, col2, _ = st.columns([3, 3, 6])
                                    with col1:
                                        if st.button("Yes", key=f"yes_button_{idx}"):
                                            st.session_state["request_fine_tuned_answer"] = True
                                            st.session_state.show_fine_tuned_expander = False
                                            st.rerun()
                                    with col2:
                                        if st.button("No", key=f"no_button_{idx}"):
                                            st.session_state.show_fine_tuned_expander = False
                                            st.rerun()

    if st.session_state["request_fine_tuned_answer"]:
        fine_tuned_result = try_get_answer(user_question, context="", fine_tuned_knowledge=True)
        if fine_tuned_result:
            st.session_state.chat_history[-1]["response"] = fine_tuned_result.strip()
            st.session_state.show_fine_tuned_expander = False
            st.session_state.parsed_result['Answer'] = fine_tuned_result.strip()
        else:
            st.error("Failed to generate a fine-tuned answer.")
        st.session_state["request_fine_tuned_answer"] = False

if __name__ == "__main__":
    app()
