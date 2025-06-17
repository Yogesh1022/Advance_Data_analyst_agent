import sys
import os
import streamlit as st
import polars as pl
import pandas as pd
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dotenv import load_dotenv

# Add project root to sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

# Load environment variables
load_dotenv()

# First Streamlit command
st.set_page_config(page_title="Advanced Multimodal Data Analyst Agent", layout="wide")

# Delayed imports to avoid side effects
from src.agents.orchestrator_agent import OrchestratorAgent
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger()

# Initialize Together AI client and model name
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

# Initialize OrchestratorAgent
@st.cache_resource
def get_orchestrator_agent():
    return OrchestratorAgent(
        logger=logger,
        llm_model=LLM_MODEL_NAME,
        together_api_key=TOGETHER_API_KEY
    )

orchestrator = get_orchestrator_agent()

# Title
st.title("📊 Advanced Multimodal Data Analyst Agent")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_dataframe" not in st.session_state:
    st.session_state.current_dataframe = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

# File uploader in sidebar
st.sidebar.header("Upload Data File")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "txt", "docx", "pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="file_uploader_widget"
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()

    # Check if the uploaded file is new
    if st.session_state.current_file_name != file_name:
        st.session_state.current_file_name = file_name
        st.session_state.messages.append({"role": "assistant", "content": f"Processing file: **{file_name}**"})
        with st.spinner(f"Reading and processing {file_name}..."):
            try:
                file_content = io.BytesIO(uploaded_file.getvalue())
                processed_data = orchestrator.handle_file_upload(file_content, file_name, file_extension)

                if isinstance(processed_data, pl.DataFrame):
                    st.session_state.current_dataframe = processed_data
                    st.session_state.messages.append({"role": "assistant", "content": f"Successfully loaded **{file_name}**. Here's a preview of the data:"})
                    st.session_state.messages.append({"role": "data_preview", "content": processed_data.head().to_pandas().to_markdown(index=False)})
                elif isinstance(processed_data, str):
                    st.session_state.current_dataframe = None
                    st.session_state.messages.append({"role": "assistant", "content": f"Successfully extracted text from **{file_name}**. Content preview:\n```\n{processed_data[:500]}...\n```"})
                else:
                    st.session_state.current_dataframe = None
                    st.session_state.messages.append({"role": "assistant", "content": f"Successfully processed **{file_name}**, but no structured data or text was extracted."})

            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"Error processing {file_name}: {e}"})
                logger.error(f"Error processing uploaded file: {e}")
    else:
        st.sidebar.info(f"File '{file_name}' is already loaded.")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])
    elif message["role"] == "data_preview":
        with st.chat_message("assistant"):
            st.subheader("Data Preview:")
            try:
                df_preview = pd.read_csv(io.StringIO(message["content"]), sep='|', skipinitialspace=True)
                df_preview = df_preview.iloc[1:].dropna(axis=1, how='all').iloc[:, 1:-1]
                df_preview.columns = [col.strip() for col in df_preview.columns]
                st.dataframe(df_preview)
            except Exception as e:
                st.error(f"Could not display data preview: {e}")
                st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What can I help you with?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
with st.chat_message("assistant"):
    with st.spinner("Analyzing..."):
        try:
            response_content = orchestrator.process_query(
                prompt,
                st.session_state.current_dataframe,
                st.session_state.messages
            )
        except Exception as e:
            response_content = f"Error processing query: {str(e)}"
            logger.error(f"Query error: {str(e)}")

        if isinstance(response_content, dict) and "plot_type" in response_content:
            if response_content["plot_type"] in ["matplotlib", "seaborn"]:
                st.pyplot(response_content["figure"])
                plt.close(response_content["figure"])
            elif response_content["plot_type"] == "plotly":
                st.plotly_chart(response_content["figure"], use_container_width=True)
            st.markdown(response_content["description"])
            st.session_state.messages.append({"role": "assistant", "content": response_content["description"]})
        else:
            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})