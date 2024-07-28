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
import datetime
import requests
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

### Functions: Start ###

SCOPES = ['https://www.googleapis.com/auth/generative-language.retriever']

def load_creds():
    """Load credentials from Streamlit secrets and handle them using a temporary file."""
    # Parse the token data from Streamlit's secrets
    token_info = {
        'token': st.secrets["token"]["value"],
        'refresh_token': st.secrets["token"]["refresh_token"],
        'token_uri': st.secrets["token"]["token_uri"],
        'client_id': st.secrets["token"]["client_id"],
        'client_secret': st.secrets["token"]["client_secret"],
        'scopes': st.secrets["token"]["scopes"],
        'expiry': st.secrets["token"]["expiry"]  # Assuming expiry is directly usable
    }

    # Create a temporary file to store the token data
    temp_dir = tempfile.mkdtemp()
    token_file_path = os.path.join(temp_dir, 'token.json')
    with open(token_file_path, 'w') as token_file:
        json.dump(token_info, token_file)

    # Load the credentials from the temporary file
    creds = Credentials.from_authorized_user_file(token_file_path, SCOPES)

    # Refresh the token if necessary
    if creds and creds.expired and creds.refresh_token:
        st.toast("Currently refreshing token...")
        creds.refresh(Request())

        # Optionally update the temporary file with the refreshed token data
        with open(token_file_path, 'w') as token_file:
            token_file.write(creds.to_json())

    return creds

# Function to generate a signed URL for a file
def generate_signed_url(bucket_name, blob_name, service_account_info, expiration=3600):
    storage_client = storage.Client.from_service_account_info(service_account_info)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(expiration=datetime.timedelta(seconds=expiration))
    return url

# Function to download file from URL to a temporary directory
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

# Function to extract text from PDFs
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        extracted_text = extract_text(pdf)
        text += extracted_text
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create conversational chain
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input
def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to process user input with fine-tuned Gemini model
def user_input_fine_tuned(user_question, api_key):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": user_question,
            "temperature": 0.5,
            "max_tokens": 150
        }
        response = requests.post("YOUR_GEMINI_MODEL_ENDPOINT", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["text"]
        else:
            st.error(f"Failed to get response from fine-tuned model: {response.status_code}")
            return "An error occurred while processing your request."
    except Exception as e:
        st.error(f"Error processing user input with fine-tuned model: {e}")
        return "An error occurred while processing your request."

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
    firestore_db = firestore.client()
    st.session_state.db = firestore_db

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
    submit_button = st.button("Submit", key="submit_button")

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
            retriever['id'] = doc.id  # Add document ID to the retriever dictionary
            retriever_name = retriever['retriever_name']
            retriever_description = retriever['retriever_description']
            with st.expander(retriever_name):
                st.markdown(f"**Description:** {retriever_description}")

                # Generate signed URL for the document
                parsed_url = urlparse(retriever['document'])
                file_name = os.path.basename(unquote(parsed_url.path))
                signed_url = generate_signed_url('connext-chatbot-admin.appspot.com', file_name, st.secrets["service_account"])

                st.markdown(f"_**File Name**_: {file_name}")
                st.markdown(f"[Download PDF]({signed_url})", unsafe_allow_html=True)

                retriever["signed_url"] = signed_url
                st.session_state["retrievers"][retriever_name] = retriever  # Populate the retriever dictionary

        st.title("PDF Retriever Selection:")
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

    # Setup placeholders for answers
    answer_placeholder = st.empty()

    if st.session_state.parsed_result is not None and "Answer" in st.session_state.parsed_result:
        answer_placeholder.write(f"Reply:\n\n {st.session_state.parsed_result['Answer']}")
        
        # Check if the answer is not directly in the context
        if "Is_Answer_In_Context" in st.session_state.parsed_result and not st.session_state.parsed_result["Is_Answer_In_Context"]:
            if st.session_state.show_fine_tuned_expander:
                with st.expander("Get fine-tuned answer?", expanded=False):
                    st.write("Would you like me to generate the answer based on my fine-tuned knowledge?")
                    col1, col2, _ = st.columns([3, 3, 6])
                    with col1:
                        if st.button("Yes", key="yes_button"):
                            # Use session state to handle the rerun after button press
                            print("Requesting fine_tuned_answer...")
                            st.session_state["request_fine_tuned_answer"] = True
                            st.session_state.show_fine_tuned_expander = False
                            st.rerun()
                    with col2:
                        if st.button("No", key="no_button"):
                            st.session_state.show_fine_tuned_expander = False
                            st.rerun()

    # Handle the generation of fine-tuned answer if the flag is set
    if st.session_state["request_fine_tuned_answer"]:
        print("Generating fine-tuned answer...")
        fine_tuned_result = user_input_fine_tuned(user_question, google_ai_api_key)
        if fine_tuned_result:
            print(fine_tuned_result.strip())
            answer_placeholder.write(f"Fine-tuned Reply:\n\n {fine_tuned_result.strip()}")
            st.session_state.show_fine_tuned_expander = False
        else:
            answer_placeholder.write("Failed to generate a fine-tuned answer.")
        st.session_state["request_fine_tuned_answer"] = False  # Reset the flag after handling

if __name__ == "__main__":
    app()
