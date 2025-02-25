import asyncio
import logging
import os
import subprocess
import traceback

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# Import Ollama from langchain_community (the new import)
try:
    from langchain_community.llms import Ollama
    from langchain_community.llms.ollama import OllamaEndpointNotFoundError
    ollama_integration_available = True
except ImportError:
    ollama_integration_available = False

# Set up logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

st.title("Custom PDF Chatbot (Offline Mode)")

def get_ollama_models():
    """
    Uses the Ollama CLI to get a list of available models.
    Parses the output to extract only the model names (first column).
    Returns an empty list if any error occurs.
    """
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            models = []
            # Skip header and extract the first token (model name) from each line.
            for line in lines:
                if line.startswith("NAME"):
                    continue
                tokens = line.split()
                if tokens:
                    models.append(tokens[0])
            return models
        else:
            logger.error("Ollama CLI error: %s", result.stderr)
            return []
    except Exception as e:
        logger.exception("Error detecting Ollama models: %s", str(e))
        return []

# Sidebar configuration.
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
        st.sidebar.error("No Ollama models detected. Please ensure Ollama is installed and models are available.")
        selected_model = None

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

st.sidebar.markdown("---")
st.sidebar.markdown("**About Me**")
st.sidebar.write("Name: Deepak Yadav")
st.sidebar.write("Bio: Passionate about AI and machine learning. Enjoys working on innovative projects and sharing knowledge with the community.")
st.sidebar.markdown("[GitHub](https://github.com/deepak7376)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/dky7376/)")

@st.cache_resource(show_spinner=False)
def initialize_qa_chain(filepath, model_type, checkpoint):
    """
    Loads the PDF document, splits it, creates embeddings, initializes the selected LLM,
    and returns the QA chain.
    """
    try:
        loader = PyPDFLoader(filepath)
        documents = loader.load()
    except Exception as e:
        logger.exception("Error loading PDF: %s", str(e))
        raise ValueError("Failed to load the PDF document. Please ensure the file is valid.")

    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
    except Exception as e:
        logger.exception("Error splitting document: %s", str(e))
        raise ValueError("Failed to split the PDF document for processing.")

    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(splits, embeddings)
    except Exception as e:
        logger.exception("Error creating embeddings/vector store: %s", str(e))
        raise ValueError("Failed to create embeddings or vector store.")

    try:
        if model_type == "HuggingFace":
            # Load local copies of the model and tokenizer.
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint, device_map=torch.device("cpu"), torch_dtype=torch.float32, local_files_only=True
            )
            pipe = pipeline(
                "text2text-generation",
                model=base_model,
                tokenizer=tokenizer,
                max_length=256,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        elif model_type == "Ollama":
            # For Ollama, the model name (checkpoint) is used directly.
            llm = Ollama(model=checkpoint)
        else:
            raise ValueError("Unsupported model type selected.")
    except Exception as e:
        logger.exception("Error initializing the LLM: %s", str(e))
        raise ValueError("Failed to initialize the language model. Check your model configuration.")

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
        )
        return qa_chain
    except Exception as e:
        logger.exception("Error creating QA chain: %s", str(e))
        raise ValueError("Failed to initialize the QA chain.")

def process_answer(query, qa_chain):
    """
    Processes the query through the QA chain and returns the generated response.
    Specific handling for Ollama endpoint errors is included.
    """
    try:
        generated_text = qa_chain.run(query)
        return generated_text
    except OllamaEndpointNotFoundError as e:
        logger.exception("Ollama model endpoint not found: %s", str(e))
        return ("Ollama model endpoint not found. Please ensure that the specified model is pulled locally. "
                "You can try running the command suggested in the error message (e.g., `ollama pull <model>`).")
    except Exception as e:
        logger.exception("Error during query processing: %s", str(e))
        return "An error occurred while processing your query. Please try again later."

# Initialize or update the QA chain if a file is uploaded and a model is selected.
if uploaded_file is not None and selected_model is not None:
    try:
        os.makedirs("docs", exist_ok=True)
        filepath = os.path.join("docs", uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Processing document..."):
            qa_chain = initialize_qa_chain(filepath, model_type, selected_model)
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        qa_chain = None
else:
    qa_chain = None

# Initialize session state for chat messages.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages.
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Chat input.
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if qa_chain:
        response = process_answer({'query': prompt}, qa_chain)
    else:
        response = "Please upload a valid PDF file and ensure the QA chain is properly initialized."

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

