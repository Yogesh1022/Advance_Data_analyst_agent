import sys
import os
import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import io
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dotenv import load_dotenv
from together import Together
from typing import List, Dict, Optional
from docx import Document
from pypdf import PdfReader
from PIL import Image
import pytesseract
import easyocr
import scipy.stats as stats

# Setup sys.path
current_script_dir = os.path.dirname(os.path.abspath(__file__))
if current_script_dir not in sys.path:
    sys.path.insert(0, current_script_dir)

# Load environment variables
load_dotenv()

# Logger setup
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('data_analyst_agent.log', maxBytes=1000000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# File Manager
class FileManager:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.logger = logger
        os.makedirs(output_dir, exist_ok=True)

    def save_dataframe(self, df: pl.DataFrame, filename: str):
        try:
            filepath = os.path.join(self.output_dir, filename)
            if filename.endswith('.csv'):
                df.write_csv(filepath)
            elif filename.endswith('.xlsx'):
                df.write_excel(filepath)
            self.logger.info(f"Saved DataFrame to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving DataFrame: {str(e)}")
            raise

    def save_text(self, text: str, filename: str):
        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            self.logger.info(f"Saved text to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving text: {str(e)}")
            raise

# File Readers
class FileReaders:
    def __init__(self):
        self.easy_ocr_reader = None
        self.logger = logger
        tesseract_cmd = os.getenv("TESSERACT_CMD_PATH")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def read_csv(self, file_content: io.BytesIO):
        encodings = ["utf-8", "latin-1", "iso-8859-1"]
        for encoding in encodings:
            try:
                file_content.seek(0)
                return pl.read_csv(file_content, encoding=encoding)
            except Exception as e:
                if encoding == encodings[-1]:
                    raise Exception(f"Failed to read CSV with tried encodings: {str(e)}")
                continue

    def read_xlsx(self, file_content: io.BytesIO, sheet_name=0):
        try:
            file_content.seek(0)
            return pl.read_excel(file_content, sheet_name=sheet_name)
        except Exception as e:
            raise Exception(f"Failed to read XLSX: {str(e)}")

    def read_text(self, file_content: io.BytesIO):
        try:
            file_content.seek(0)
            return file_content.read().decode("utf-8")
        except Exception as e:
            raise Exception(f"Failed to read text file: {str(e)}")

    def read_docx(self, file_content: io.BytesIO):
        try:
            file_content.seek(0)
            doc = Document(file_content)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Failed to read DOCX: {str(e)}")

    def read_pdf(self, file_content: io.BytesIO):
        try:
            file_content.seek(0)
            pdf = PdfReader(file_content)
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")

    def read_image_text(self, file_content: io.BytesIO, ocr_engine="pytesseract"):
        try:
            file_content.seek(0)
            image = Image.open(file_content).convert("RGB")
            if ocr_engine == "easyocr":
                if self.easy_ocr_reader is None:
                    self.easy_ocr_reader = easyocr.Reader(["en"])
                result = self.easy_ocr_reader.readtext(image)
                return "\n".join([text[1] for text in result])
            else:
                return pytesseract.image_to_string(image)
        except Exception as e:
            self.logger.error(f"Failed to read image text: {str(e)}")
            raise

# Data Cleaner
class DataCleaner:
    def __init__(self):
        self.logger = logger

    def initial_checks(self, df: pl.DataFrame):
        self.logger.info(f"Initial Data Info:\n{df.schema}")
        self.logger.info(f"Missing Values:\n{df.select(pl.col('*').is_null().sum())}")
        self.logger.info(f"Duplicate Rows: {df.is_duplicated().sum()}")
        self.logger.info(f"Descriptive Statistics:\n{df.describe()}")

    def standardize_data_types(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            for col in df.columns:
                if df[col].dtype == pl.Utf8:
                    try:
                        df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))
                    except:
                        pass
                if df[col].dtype == pl.Utf8 and col.lower().find('date') != -1:
                    df = df.with_columns(pl.col(col).str.to_datetime(strict=False).alias(col))
            return df
        except Exception as e:
            self.logger.error(f"Error standardizing data types: {str(e)}")
            return df

    def handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            for col in df.columns:
                if df[col].is_null().sum() > 0:
                    if df[col].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
                        median_val = df[col].median()
                        df = df.with_columns(pl.col(col).fill_null(median_val))
                    else:
                        df = df.with_columns(pl.col(col).fill_null("Unknown"))
            return df
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            return df

    def handle_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            for col in df.columns:
                if df[col].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
                    q_low, q_high = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q_high - q_low
                    lower_bound = q_low - 1.5 * iqr
                    upper_bound = q_high + 1.5 * iqr
                    df = df.with_columns(
                        pl.col(col).clip(lower_bound, upper_bound).alias(col)
                    )
            return df
        except Exception as e:
            self.logger.error(f"Error handling outliers: {str(e)}")
            return df

    def run_pipeline(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            self.initial_checks(df)
            return (df.lazy()
                    .pipe(self.standardize_data_types)
                    .pipe(self.handle_missing_values)
                    .pipe(self.handle_outliers)
                    .collect())
        except Exception as e:
            self.logger.error(f"Error running pipeline: {str(e)}")
            return df

# Descriptive Stats
class DescriptiveStats:
    def __init__(self):
        self.logger = logger

    def get_summary(self, df: pl.DataFrame) -> pl.DataFrame:
        try:
            self.logger.info("Generating DataFrame summary")
            return df.describe()
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            raise

# Trend Analysis
class TrendAnalysis:
    def __init__(self):
        self.logger = logger

    def identify_time_series_trends(self, df: pl.DataFrame, time_column: str, value_column: str):
        try:
            if time_column not in df.columns or value_column not in df.columns:
                return "Error: Specified time or value column not found."
            df_copy = df.select([time_column, value_column]).clone()
            df_copy = df_copy.with_columns(
                pl.col(time_column).cast(pl.Datetime, strict=False)
            )
            df_copy = df_copy.filter(
                ~pl.col(time_column).is_null() & ~pl.col(value_column).is_null()
            )
            df_copy = df_copy.sort(time_column)
            if df_copy.height == 0:
                return "No valid time series data after cleaning."
            df_copy = df_copy.with_columns(
                pl.col(value_column).rolling_mean(window_size=3).alias("moving_avg")
            )
            start_value = df_copy.select(pl.col(value_column).first()).item()
            end_value = df_copy.select(pl.col(value_column).last()).item()
            time_min = df_copy.select(pl.col(time_column).min()).item()
            time_max = df_copy.select(pl.col(time_column).max()).item()
            trend_description = f"Over the period from {time_min.strftime('%Y-%m-%d')} to {time_max.strftime('%Y-%m-%d')}, "
            if end_value > start_value * 1.1:
                trend_description += f"the {value_column} shows a significant upward trend, increasing from {start_value:.2f} to {end_value:.2f}."
            elif end_value < start_value * 0.9:
                trend_description += f"the {value_column} shows a significant downward trend, decreasing from {start_value:.2f} to {end_value:.2f}."
            else:
                trend_description += f"the {value_column} remained relatively stable, fluctuating between {start_value:.2f} and {end_value:.2f}."
            return trend_description
        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return f"Error analyzing trends: {str(e)}"

    def find_correlations(self, df: pl.DataFrame):
        try:
            numeric_df = df.select(pl.col(pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64))
            if len(numeric_df.columns) < 2:
                return "Need at least two numerical columns to find correlations."
            correlation_matrix = numeric_df.corr()
            correlation_description = "Here are the correlations between numerical columns:\n"
            correlation_description += correlation_matrix.to_pandas().to_markdown() + "\n\n"
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    corr_val = correlation_matrix[col1][j]
                    if abs(corr_val) > 0.7:
                        relation = "strong positive" if corr_val > 0 else "strong negative"
                        strong_correlations.append(f"'{col1}' and '{col2}' show a {relation} correlation ({corr_val:.2f}).")
            if strong_correlations:
                correlation_description += "Notable correlations:\n" + "\n".join(strong_correlations)
            else:
                correlation_description += "No particularly strong correlations (absolute value > 0.7) found."
            return correlation_description
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
            return f"Error finding correlations: {str(e)}"

# Chart Generator
class ChartGenerator:
    def __init__(self):
        self.logger = logger

    def generate_chart(self, query: str, df: pl.DataFrame) -> dict:
        try:
            self.logger.info(f"Generating chart for query: {query}")
            query_lower = query.lower()
            num_cols = df.select(pl.col(pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64)).columns
            cat_cols = df.select(pl.col(pl.Utf8)).columns
            if "bar" in query_lower and num_cols and cat_cols:
                cat_col, num_col = cat_cols[0], num_cols[0]
                grouped = df.group_by(cat_col).agg(pl.col(num_col).mean())
                plt.figure()
                plt.bar(grouped[cat_col].to_pandas(), grouped[num_col].to_pandas())
                plt.xlabel(cat_col)
                plt.ylabel(num_col)
                plt.title(f"Bar Chart of {num_col} by {cat_col}")
                return {"plot_type": "matplotlib", "figure": plt.gcf(), "description": f"Bar chart of average {num_col} by {cat_col}"}
            elif "line" in query_lower and "date" in df.columns and num_cols:
                num_col = num_cols[0]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["date"].to_pandas(), y=df[num_col].to_pandas(), mode="lines"))
                fig.update_layout(title=f"Line Chart of {num_col} Over Time", xaxis_title="Date", yaxis_title=num_col)
                return {"plot_type": "plotly", "figure": fig, "description": f"Line chart of {num_col} over time"}
            return "Chart type not supported or insufficient data."
        except Exception as e:
            self.logger.error(f"Error generating chart: {str(e)}")
            return f"Error generating chart: {str(e)}"

# LLM Insights
class LLMInsights:
    def __init__(self, api_key: str, model_name: str):
        self.logger = logger
        self.client = Together(api_key=api_key)
        self.model_name = model_name

    def get_text_analysis(self, query: str, text: str, chat_history: List[Dict[str, str]] = None) -> str:
        try:
            system_prompt = (
                "You are a data analyst specializing in text analysis. Provide a clear and concise response "
                "to the user's query based on the provided text content. Use context from the chat history if available."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text content:\n{text[:2000]}\n\nQuery: {query}"}
            ]
            if chat_history:
                for msg in chat_history:
                    if msg["role"] in ["user", "assistant"]:
                        messages.append({"role": msg["role"], "content": msg["content"]})
            self.logger.info(f"Sending text analysis query to LLM: {query}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error in text analysis: {str(e)}")
            return f"Error analyzing text: {str(e)}"

    def get_data_analysis(self, messages: List[Dict[str, str]]) -> str:
        try:
            self.logger.info("Sending data analysis query to LLM")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error in data analysis: {str(e)}")
            return f"Error analyzing data: {str(e)}"

# Orchestrator Agent
class OrchestratorAgent:
    def __init__(self, llm_model: str, together_api_key: str):
        self.logger = logger
        self.file_readers = FileReaders()
        self.data_cleaner = DataCleaner()
        self.descriptive_stats = DescriptiveStats()
        self.trend_analysis = TrendAnalysis()
        self.chart_generator = ChartGenerator()
        self.llm_insights = LLMInsights(together_api_key, llm_model)
        self.file_manager = FileManager()
        self.current_data = None
        self.current_file_name = None

    def handle_file_upload(self, file_content: io.BytesIO, file_name: str, file_extension: str):
        self.logger.info(f"Handling file upload: {file_name}")
        self.current_file_name = file_name
        data = None
        file_content.seek(0)
        try:
            if file_extension == ".csv":
                data = self.file_readers.read_csv(file_content)
            elif file_extension == ".xlsx":
                data = self.file_readers.read_xlsx(file_content)
            elif file_extension == ".txt":
                data = self.file_readers.read_text(file_content)
            elif file_extension == ".docx":
                data = self.file_readers.read_docx(file_content)
            elif file_extension == ".pdf":
                data = self.file_readers.read_pdf(file_content)
            elif file_extension in [".jpg", ".jpeg", ".png"]:
                data = self.file_readers.read_image_text(file_content, ocr_engine="pytesseract")
            self.current_data = data
            if isinstance(data, pl.DataFrame):
                self.logger.info("Performing initial data cleaning...")
                cleaned_data = self.data_cleaner.run_pipeline(data)
                self.current_data = cleaned_data
                self.file_manager.save_dataframe(cleaned_data, f"cleaned_{file_name}")
                return cleaned_data
            else:
                self.file_manager.save_text(data, f"extracted_{file_name}")
                return data
        except Exception as e:
            self.logger.error(f"Error processing uploaded file: {str(e)}")
            raise

    def process_query(self, query: str, dataframe: pl.DataFrame = None, chat_history: List[Dict[str, str]] = None):
        self.logger.info(f"Processing query: {query}")
        if dataframe is None and self.current_data is None:
            return "Please upload a file first to perform analysis."
        active_data = dataframe if dataframe is not None else self.current_data
        if isinstance(active_data, str):
            return self.llm_insights.get_text_analysis(query, active_data, chat_history)
        try:
            data_summary = self.descriptive_stats.get_summary(active_data).to_pandas().to_markdown(index=False)
            data_head = active_data.head().to_pandas().to_markdown(index=False)
            data_columns = ", ".join(active_data.columns)
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame context: {str(e)}")
            return f"Error processing query: {str(e)}"
        system_prompt = (
            "You are a highly knowledgeable data analyst. Answer the user's query based on the provided dataset context. "
            "Provide clear, concise, and accurate responses. If the query involves analysis, use the provided summary, "
            "column names, and data preview. For visualization requests, describe the chart type and key insights."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Dataset columns: {data_columns}\n"
                f"Data summary:\n{data_summary}\n"
                f"Data preview:\n{data_head}\n\n"
                f"Query: {query}"
            )}
        ]
        if chat_history:
            for msg in chat_history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        try:
            if "trend" in query.lower() or "over time" in query.lower():
                time_column = None
                value_column = None
                for col in active_data.columns:
                    if "date" in col.lower() or "time" in col.lower():
                        time_column = col
                    if active_data[col].dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64):
                        value_column = col
                if time_column and value_column:
                    return self.trend_analysis.identify_time_series_trends(active_data, time_column, value_column)
            elif "correlation" in query.lower():
                return self.trend_analysis.find_correlations(active_data)
            elif any(word in query.lower() for word in ["plot", "chart", "graph", "visualize"]):
                chart_result = self.chart_generator.generate_chart(query, active_data)
                if isinstance(chart_result, dict) and "figure" in chart_result:
                    return chart_result
                return chart_result or "Unable to generate chart for this query."
            response = self.llm_insights.get_data_analysis(messages)
            return response
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

# Streamlit App
st.set_page_config(page_title="Advanced Multimodal Data Analyst Agent", layout="wide")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLM_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

@st.cache_resource
def get_orchestrator_agent():
    return OrchestratorAgent(
        llm_model=LLM_MODEL_NAME,
        together_api_key=TOGETHER_API_KEY
    )

orchestrator = get_orchestrator_agent()

st.title("📊 Advanced Multimodal Data Analyst Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_dataframe" not in st.session_state:
    st.session_state.current_dataframe = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

st.sidebar.header("Upload Data File")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file",
    type=["csv", "xlsx", "txt", "docx", "pdf", "jpg", "jpeg", "png"],
    accept_multiple_files=False,
    key="file_uploader_widget"
)

if uploaded_file is not None:
    if uploaded_file.size > 100_000_000:
        st.error("File size exceeds 100 MB limit.")
    else:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()
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
                    st.session_state.messages.append({"role": "assistant", "content": f"Error processing {file_name}: {str(e)}"})
                    logger.error(f"Error processing uploaded file: {str(e)}")
        else:
            st.sidebar.info(f"File '{file_name}' is already loaded.")

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