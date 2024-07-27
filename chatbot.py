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
from functools import partial
import mimetypes
import datetime

# Initialize Firebase SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(st.secrets["service_account"])
    firebase_admin.initialize_app(cred)

### Functions: Start ###

def download_file_to_temp(url):
    storage_client = storage.Client.from_service_account_info(st.secrets["service_account"])
    bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')
    temp_dir = tempfile.mkdtemp()

    parsed_url = urlparse(url)
    file_name = os.path.basename(unquote(parsed_url.path))

    blob = bucket.blob(file_name)
    temp_file_path = os.path.join(temp_dir, file_name)
    blob.download_to_filename(temp_file_path)

    return temp_file_path, file_name

def update_file(file, retriever):
    if file is not None:
        storage_client = storage.Client.from_service_account_info(st.secrets["service_account"])
        bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')

        old_file_url = retriever['document']
        old_file_path = urlparse(old_file_url).path
        old_file_name = os.path.basename(unquote(old_file_path))
        old_blob = bucket.blob(old_file_name)

        if old_blob.exists():
            old_blob.delete()

        file_path = file.name
        new_blob = bucket.blob(file_path)
        
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            mime_type = 'application/pdf'
        
        new_blob.upload_from_file(file, content_type=mime_type)
        download_url = new_blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')

        retriever['document'] = download_url
        st.session_state.db.collection('Retrievers').document(retriever['id']).update({'document': download_url})
        st.session_state["retrievers"][retriever['retriever_name']]['file_name'] = file_path
        st.session_state["retrievers"][retriever['retriever_name']]['document'] = download_url

    else:
        st.error("No file was selected to update.")

@st.experimental_memo
def delete_retriever(retriever):
    retriever_name = retriever['retriever_name']
    storage_client = storage.Client.from_service_account_info(st.secrets["service_account"])
    bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')

    doc_id = retriever['id']
    document_url = retriever['document']

    file_path = urlparse(document_url).path
    file_name = os.path.basename(unquote(file_path))
    blob = bucket.blob(file_name)

    if blob.exists():
        blob.delete()

    st.session_state.db.collection('Retrievers').document(doc_id).delete()
    del st.session_state["retrievers"][retriever_name]

@st.experimental_memo
def add_retriever(name, description, file):
    storage_client = storage.Client.from_service_account_info(st.secrets["service_account"])
    bucket = storage_client.bucket('connext-chatbot-admin.appspot.com')

    file_path = file.name
    new_blob = bucket.blob(file_path)

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = 'application/pdf'

    new_blob.upload_from_file(file, content_type=mime_type)
    download_url = new_blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')

    retriever_data = {
        'retriever_name': name,
        'retriever_description': description,
        'document': download_url
    }
    doc_ref = st.session_state.db.collection('Retrievers').add(retriever_data)
    retriever_data['id'] = doc_ref[1].id
    st.session_state["retrievers"][name] = retriever_data

@st.experimental_memo
def update_description(retriever, new_description):
    st.session_state.db.collection('Retrievers').document(retriever['id']).update({'retriever_description': new_description})
    st.session_state["retrievers"][retriever['retriever_name']]['retriever_description'] = new_description

@st.experimental_memo
def update_name(retriever, new_name):
    st.session_state.db.collection('Retrievers').document(retriever['id']).update({'retriever_name': new_name})
    st.session_state["retrievers"][new_name] = st.session_state["retrievers"].pop(retriever['retriever_name'])
    st.session_state["retrievers"][new_name]['retriever_name'] = new_name

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
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
    st.write("Reply:\n\n", response["output_text"])

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
                st.markdown(f"_**File Name**_: {file_name}")
                st.markdown(f"[Download PDF](https://{retriever['document']})", unsafe_allow_html=True)
                retriever["file_path"] = file_path
                st.session_state["retrievers"][retriever_name] = retriever  # Populate the retriever dictionary
                if st.button("Edit Document Name", key=f"{retriever_name}_retriever_name_editor"):
                    new_name = st.text_input("Retriever Name", value=retriever["retriever_name"], key=f"name_{retriever['retriever_name']}")
                    if st.button("Update", key=f"update_name_button_{retriever['retriever_name']}"):
                        update_name(retriever, new_name)
                if st.button("Delete Document", key=f"{retriever_name}_retriever_delete"):
                    delete_retriever(retriever)
                if st.button("Edit Description", key=f"{retriever_name}_retriever_description_editor"):
                    new_description = st.text_area("Document Description", height=300, value=retriever["retriever_description"], key=f"description_{retriever['retriever_name']}")
                    if st.button("Update", key=f"update_desc_button_{retriever['retriever_name']}"):
                        update_description(retriever, new_description)
                updated_doc = st.file_uploader("Upload Chatbot Document", accept_multiple_files=False, type=["pdf", "doc", "docx"], key=f"{retriever_name}_file_uploader")
                if st.button("Update File", key=f"{retriever_name}_file_update_button"):
                    update_file(updated_doc, retriever)
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

    if st.button("Add New Document"):
        name = st.text_input("Document Name", key="new_retriever")
        description = st.text_area("Document Description", height=300)
        file = st.file_uploader("Upload Chatbot Document", accept_multiple_files=False, type=["pdf", "doc", "docx"])

        if st.button("Submit", key="submit_new_retriever"):
            if name and description and file:
                add_retriever(name, description, file)
                st.success("New Document Added Successfully")
                st.rerun()
            else:
                st.error("Please fill all fields to add a new document.")

if __name__ == "__main__":
    app()
