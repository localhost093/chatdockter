import streamlit as st
import os
import subprocess
from langchain.document_loaders import PyPDFLoader  # Changed to PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# Try to import Ollama integration from LangChain
try:
    from langchain.llms import Ollama
    ollama_integration_available = True
except ImportError:
    ollama_integration_available = False

st.title("Custom PDF Chatbot")

# Helper function to get available Ollama models using the Ollama CLI
def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            # Assume the output is a list of models, one per line
            models = result.stdout.splitlines()
            models = [model.strip() for model in models if model.strip()]
            return models
        else:
            return []
    except Exception as e:
        return []

# Sidebar configuration
st.sidebar.header("Settings")

if ollama_integration_available:
    model_type = st.sidebar.radio("Select Model Type", options=["HuggingFace", "Ollama"])
else:
    model_type = "HuggingFace"
    st.sidebar.info("Ollama integration not available; defaulting to HuggingFace models.")

if model_type == "HuggingFace":
    hf_model_options = ["MBZUAI/LaMini-T5-738M", "google/flan-t5-base", "google/flan-t5-small"]
    selected_model = st.sidebar.radio("Choose HuggingFace Model", hf_model_options)
elif model_type == "Ollama":
    ollama_models = get_ollama_models()
    if ollama_models:
        selected_model = st.sidebar.radio("Choose Ollama Model", ollama_models)
    else:
        st.sidebar.error("No Ollama models detected on this machine. Please ensure Ollama is installed and models are available.")
        selected_model = None

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

st.sidebar.markdown("---")
st.sidebar.markdown("**About Me**")
st.sidebar.write("Name: Deepak Yadav")
st.sidebar.write("Bio: Passionate about AI and machine learning. Enjoys working on innovative projects and sharing knowledge with the community.")
st.sidebar.markdown("[GitHub](https://github.com/deepak7376)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/dky7376/)")

@st.cache_resource
def initialize_qa_chain(filepath, model_type, checkpoint):
    # Use PyPDFLoader instead of PDFMinerLoader to avoid metadata issues.
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Create embeddings 
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(splits, embeddings)

    # Initialize LLM based on the chosen model type
    if model_type == "HuggingFace":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, device_map=torch.device('cpu'), torch_dtype=torch.float32
        )
        pipe = pipeline(
            'text2text-generation',
            model=base_model,
            tokenizer=tokenizer,
            max_length=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    elif model_type == "Ollama":
        # Use the Ollama LLM integration from LangChain.
        # The 'checkpoint' here is the Ollama model name.
        llm = Ollama(model=checkpoint)
    else:
        raise ValueError("Unsupported model type selected.")

    # Build the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )
    return qa_chain

def process_answer(query, qa_chain):
    generated_text = qa_chain.run(query)
    return generated_text

# Initialize or update the QA chain if a file is uploaded and a model is selected
if uploaded_file is not None and selected_model is not None:
    os.makedirs("docs", exist_ok=True)
    filepath = os.path.join("docs", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())
    with st.spinner("Processing document..."):
        qa_chain = initialize_qa_chain(filepath, model_type, selected_model)
else:
    qa_chain = None

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history using chat_message containers
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Chat input area
if prompt := st.chat_input("Type your question here..."):
    # Append and display the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Process and generate a response
    if qa_chain:
        response = process_answer({'query': prompt}, qa_chain)
    else:
        response = "Please upload a PDF file to enable the chatbot."

    # Append and display the assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
