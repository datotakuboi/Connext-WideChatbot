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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Initialize Firebase SDK
if not firebase_admin._apps:
    try:
        cred = credentials.Certificate({
            "type": st.secrets["service_account"]["type"],
            "project_id": st.secrets["service_account"]["project_id"],
            "private_key_id": st.secrets["service_account"]["private_key_id"],
            "private_key": st.secrets["service_account"]["private_key"].replace("\\n", "\n"),
            "client_email": st.secrets["service_account"]["client_email"],
            "client_id": st.secrets["service_account"]["client_id"],
            "auth_uri": st.secrets["service_account"]["auth_uri"],
            "token_uri": st.secrets["service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["service_account"]["client_x509_cert_url"]
        })
        firebase_admin.initialize_app(cred)
    except ValueError as e:
        st.error(f"Failed to initialize Firebase: {e}")
        st.stop()

# Function to download file to a temporary directory
def download_file_to_temp(url):
    try:
        storage_client = storage.Client.from_service_account_info({
            "type": st.secrets["service_account"]["type"],
            "project_id": st.secrets["service_account"]["project_id"],
            "private_key_id": st.secrets["service_account"]["private_key_id"],
            "private_key": st.secrets["service_account"]["private_key"].replace("\\n", "\n"),
            "client_email": st.secrets["service_account"]["client_email"],
            "client_id": st.secrets["service_account"]["client_id"],
            "auth_uri": st.secrets["service_account"]["auth_uri"],
            "token_uri": st.secrets["service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["service_account"]["client_x509_cert_url"]
        })
        bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')
        temp_dir = tempfile.mkdtemp()

        parsed_url = urlparse(url)
        file_name = os.path.basename(unquote(parsed_url.path))
        blob = bucket.blob(file_name)
        temp_file_path = os.path.join(temp_dir, file_name)
        blob.download_to_filename(temp_file_path)

        return temp_file_path, file_name
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return None, None

# Other functions remain unchanged

# Main app function
def app():
    st.set_page_config(page_title="Connext Chatbot", layout="wide")

    # Retrieve API key from secrets
    google_ai_api_key = st.secrets["api_keys"]["GOOGLE_AI_STUDIO_API_KEY"]

    # Check if the API key is provided
    if not google_ai_api_key:
        st.error("Google API key is missing. Please provide it in the secrets configuration.")
        return

    # Get firestore client
    try:
        firestore_db = firestore.client()
        st.session_state.db = firestore_db
    except Exception as e:
        st.error(f"Error initializing Firestore: {e}")
        return

    # Center the logo image
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

    user_question = st.text_input("Ask a Question", key="user_question")

    if "retrievers" not in st.session_state:
        st.session_state["retrievers"] = {}
    
    if "selected_retrievers" not in st.session_state:
        st.session_state["selected_retrievers"] = []

    with st.sidebar:
        st.title("PDF Documents:")
        for idx, doc in enumerate(docs, start=1):
            retriever = doc.to_dict()
            retriever['id'] = doc.id  # Add document ID to the retriever dictionary
            retriever_name = retriever['retriever_name']
            retriever_description = retriever['retriever_description']
            with st.expander(retriever_name):
                st.markdown(f"**Description:** {retriever_description}")
                file_path, file_name = download_file_to_temp(retriever['document'])  # Get the document file path and file name
                if file_path and file_name:
                    st.markdown(f"_**File Name**_: {file_name}")
                    st.markdown(f"[Download PDF](https://{retriever['document']})", unsafe_allow_html=True)
                    retriever["file_path"] = file_path
                    st.session_state["retrievers"][retriever_name] = retriever  # Populate the retriever dictionary
        st.title("PDF Retriever Selection:")
        st.session_state["selected_retrievers"] = st.multiselect("Select Retrievers", list(st.session_state["retrievers"].keys()))

        if st.button("Submit & Process", key="process_button"):
            if google_ai_api_key:
                with st.spinner("Processing..."):
                    selected_files = [st.session_state["retrievers"][name]["file_path"] for name in st.session_state["selected_retrievers"]]
                    raw_text = get_pdf_text(selected_files)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, google_ai_api_key)
                    st.success("Processing complete.")
            else:
                st.error("Google API key is missing. Please provide it in the secrets configuration.")

    if user_question and google_ai_api_key:
        user_input(user_question, google_ai_api_key)

if __name__ == "__main__":
    app()
