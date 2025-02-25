import logging
import os
import subprocess
import traceback

import streamlit as st

# Import document loader, embeddings, text splitter, and vectorstore from langchain-community.
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Import the MapReduceDocumentsChain and LLMChain to manually construct the map-reduce chain.
from langchain.chains.mapreduce import MapReduceDocumentsChain
from langchain.chains.llm import LLMChain

# Import the new Ollama LLM from the langchain-ollama package.
from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import OllamaEndpointNotFoundError

from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Custom CSS for Modern Minimalist UI with Light/Dark Mode and Animations
# =============================================================================
custom_css = """
<style>
    /* Base styling and font */
    html, body {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        margin: 0;
        padding: 0;
        transition: background 0.3s, color 0.3s;
    }
    /* Light mode */
    @media (prefers-color-scheme: light) {
        body {
            background-color: #f5f5f5;
            color: #222;
        }
        .chat-message {
            background-color: #fff;
            border: 1px solid #e0e0e0;
        }
    }
    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        body {
            background-color: #121212;
            color: #e0e0e0;
        }
        .chat-message {
            background-color: #1e1e1e;
            border: 1px solid #333;
        }
    }
    /* Chat container styling */
    .chat-container {
        margin: 10px 0;
        padding: 10px;
        border-radius: 8px;
        max-width: 75%;
        animation: fadeIn 0.5s ease-out;
    }
    .chat-container.user {
        margin-left: auto;
        text-align: right;
    }
    .chat-container.assistant {
        margin-right: auto;
        text-align: left;
    }
    /* Fade in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Chat input styling */
    .stChatInput {
        font-size: 1.1em;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("Custom PDF Chatbot (Local Ollama Models Only with RAG - Offline Mode)")

# =============================================================================
# Callback Handler for Streaming Output
# =============================================================================
class StreamlitCallbackHandler(BaseCallbackHandler):
    """A callback handler that streams LLM tokens to a Streamlit placeholder."""
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.placeholder.markdown(self.text)

# =============================================================================
# Utility: Get Available Ollama Models
# =============================================================================
def get_ollama_models():
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

# =============================================================================
# Sidebar Configuration
# =============================================================================
st.sidebar.header("Settings")
ollama_models = get_ollama_models()
if ollama_models:
    selected_model = st.sidebar.radio("Choose Ollama Model", ollama_models)
else:
    st.sidebar.error("No Ollama models detected. Please ensure Ollama is installed and models are available offline.")
    selected_model = None

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

st.sidebar.markdown("---")
st.sidebar.markdown("**About Me**")
st.sidebar.write("Name: Deepak Yadav")
st.sidebar.write("Bio: Passionate about AI and machine learning. Enjoys working on innovative projects and sharing knowledge with the community.")
st.sidebar.markdown("[GitHub](https://github.com/deepak7376)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/dky7376/)")

# =============================================================================
# RAG Prompt Templates (Map & Combine)
# =============================================================================
map_prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template="""Given the following document fragment, extract the relevant details to answer the question.
    
Question: {question}
Fragment: {context}
Partial Answer:"""
)

combine_prompt_template = PromptTemplate(
    input_variables=["question", "summaries"],
    template="""You are given a series of partial answers extracted from different document fragments.
Combine these partial answers into a single, comprehensive, coherent answer to the question.
    
Question: {question}
Partial Answers: {summaries}
Final Answer:"""
)

# =============================================================================
# Initialize QA Components (MapReduce Chain and Retriever)
# =============================================================================
@st.cache_resource(show_spinner=False)
def initialize_qa_components(filepath, model_checkpoint):
    # Load and split the PDF document.
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
        # For offline mode, ensure the "all-MiniLM-L6-v2" model is cached locally.
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(splits, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    except Exception as e:
        logger.exception("Error creating embeddings/vector store: %s", str(e))
        raise ValueError("Failed to create embeddings or vector store.")
    # Initialize the local Ollama LLM.
    try:
        llm = Ollama(model=model_checkpoint)
    except Exception as e:
        logger.exception("Error initializing the Ollama LLM: %s", str(e))
        raise ValueError("Failed to initialize the language model. Check your model configuration.")
    # Manually build the map chain and combine chain.
    try:
        map_chain = LLMChain(llm=llm, prompt=map_prompt_template, verbose=True)
        combine_chain = LLMChain(llm=llm, prompt=combine_prompt_template, verbose=True)
        map_reduce_chain = MapReduceDocumentsChain(
            map_chain=map_chain,
            combine_chain=combine_chain,
            verbose=True
        )
    except Exception as e:
        logger.exception("Error creating MapReduce chain: %s", str(e))
        raise ValueError("Failed to initialize the MapReduce chain.")
    return map_reduce_chain, retriever

# =============================================================================
# Process Query with Streaming Output Using RAG
# =============================================================================
def process_answer(query, map_reduce_chain, retriever):
    # Retrieve relevant documents.
    try:
        docs = retriever.get_relevant_documents(query)
    except Exception as e:
        logger.exception("Error retrieving documents: %s", str(e))
        return "An error occurred during document retrieval."
    # Create a placeholder for incremental output.
    output_placeholder = st.empty()
    callback_handler = StreamlitCallbackHandler(output_placeholder)
    try:
        # Run the map-reduce chain with the retrieved documents.
        final_output = map_reduce_chain.run(
            input_documents=docs,
            question=query,
            callbacks=[callback_handler]
        )
        return final_output
    except OllamaEndpointNotFoundError as e:
        logger.exception("Ollama model endpoint not found: %s", str(e))
        return ("Ollama model endpoint not found. Please ensure that the specified model is pulled locally. "
                "Try running `ollama pull <model>` as suggested in the error message.")
    except Exception as e:
        logger.exception("Error during query processing: %s", str(e))
        return "An error occurred while processing your query. Please try again later."

# =============================================================================
# Initialize or Update the QA Components if a File is Uploaded and a Model is Selected.
# =============================================================================
if uploaded_file is not None and selected_model is not None:
    try:
        os.makedirs("docs", exist_ok=True)
        filepath = os.path.join("docs", uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())
        with st.spinner("Processing document..."):
            map_reduce_chain, retriever = initialize_qa_components(filepath, selected_model)
    except Exception as e:
        st.error(f"Error initializing QA components: {str(e)}")
        map_reduce_chain, retriever = None, None
else:
    map_reduce_chain, retriever = None, None

# =============================================================================
# Session State for Chat Messages
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.markdown(f"<div class='chat-container {role} chat-message'>{content}</div>", unsafe_allow_html=True)

# =============================================================================
# Chat Input and Response Handling
# =============================================================================
if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='chat-container user chat-message'>{prompt}</div>", unsafe_allow_html=True)
    if map_reduce_chain and retriever:
        response = process_answer(prompt, map_reduce_chain, retriever)
    else:
        response = "Please upload a valid PDF file and ensure the QA components are properly initialized."
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f"<div class='chat-container assistant chat-message'>{response}</div>", unsafe_allow_html=True)
