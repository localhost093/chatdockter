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
from langchain_community.llms import Ollama
from langchain_community.llms.ollama import OllamaEndpointNotFoundError
from langchain.prompts import PromptTemplate

# Import the base callback handler for streaming tokens
from langchain.callbacks.base import BaseCallbackHandler

# Set up logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Inject custom CSS for a modern, minimalist UI with light/dark mode and animations.
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

st.title("Custom PDF Chatbot (Local Ollama Models Only)")

def get_ollama_models():
    """
    Uses the Ollama CLI to list available models.
    Parses the output to extract only the model names (first column).
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

# Custom prompt template instructing the assistant to return full text excerpts.
custom_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant that returns full text excerpts.
Given the following document excerpts and a question, return the ENTIRE text excerpt that is relevant to the question.
Do NOT summarize or shorten the text. If there are multiple excerpts, combine them verbatim in the order provided.

Document Excerpts:
{context}

Question: {question}
Answer:"""
)

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to stream LLM tokens to a Streamlit placeholder."""
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.partial_text = ""
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.partial_text += token
        # Update the placeholder with the new partial text
        self.placeholder.markdown(f"<div class='chat-container assistant chat-message'>{self.partial_text}</div>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def initialize_qa_chain(filepath, model_checkpoint):
    """
    Loads the PDF, splits it, creates embeddings, initializes the local Ollama LLM,
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
        # Initialize the Ollama LLM with the selected model.
        llm = Ollama(model=model_checkpoint)
    except Exception as e:
        logger.exception("Error initializing the Ollama LLM: %s", str(e))
        raise ValueError("Failed to initialize the language model. Check your model configuration.")

    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
            chain_type_kwargs={"prompt": custom_prompt_template}
        )
        return qa_chain
    except Exception as e:
        logger.exception("Error creating QA chain: %s", str(e))
        raise ValueError("Failed to initialize the QA chain.")

def process_answer(query, qa_chain):
    """
    Processes the query through the QA chain and returns the generated response.
    Uses a custom callback to stream tokens as they are generated.
    """
    # Create a placeholder for the streaming response.
    placeholder = st.empty()
    callback_handler = StreamlitCallbackHandler(placeholder)
    try:
        # Run the chain with the streaming callback.
        final_text = qa_chain.run(query, callbacks=[callback_handler])
        return final_text
    except OllamaEndpointNotFoundError as e:
        logger.exception("Ollama model endpoint not found: %s", str(e))
        return ("Ollama model endpoint not found. Please ensure that the specified model is pulled locally. "
                "Try running `ollama pull <model>` as suggested in the error message.")
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
            qa_chain = initialize_qa_chain(filepath, selected_model)
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
    role = message["role"]
    content = message["content"]
    st.markdown(f"<div class='chat-container {role} chat-message'>{content}</div>", unsafe_allow_html=True)

# Chat input.
if prompt := st.chat_input("Type your question here..."):
    # Store and display the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='chat-container user chat-message'>{prompt}</div>", unsafe_allow_html=True)

    if qa_chain:
        # Process the answer and stream tokens incrementally.
        response = process_answer(prompt, qa_chain)
    else:
        response = "Please upload a valid PDF file and ensure the QA chain is properly initialized."

    # Save the final assistant response.
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f"<div class='chat-container assistant chat-message'>{response}</div>", unsafe_allow_html=True)

