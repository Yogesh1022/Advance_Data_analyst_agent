import polars as pl
import pandas as pd
import io
from together import Together
from src.data_processing.file_readers import FileReaders
from src.data_processing.data_cleaner import DataCleaner
from src.analysis.descriptive_stats import DescriptiveStats
from src.analysis.trend_analysis import TrendAnalysis
from src.visualization.chart_generator import ChartGenerator
from src.insights.llm_insights import LLMInsights
import logging

class OrchestratorAgent:
    def __init__(self, logger: logging.Logger, llm_model: str, together_api_key: str):
        self.logger = logger
        self.file_readers = FileReaders()
        self.data_cleaner = DataCleaner()
        self.descriptive_stats = DescriptiveStats()
        self.trend_analysis = TrendAnalysis()
        self.chart_generator = ChartGenerator()
        self.llm_insights = LLMInsights(together_api_key, llm_model)
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
                data = self.file_readers.read_image_text(file_content)
            self.current_data = data
            if isinstance(data, pl.DataFrame):
                self.logger.info("Performing initial data cleaning...")
                cleaned_data = self.data_cleaner.run_pipeline(data)
                self.current_data = cleaned_data
                return cleaned_data
            else:
                return data
        except Exception as e:
            self.logger.error(f"Error reading file {file_name}: {str(e)}")
            raise

    def process_query(self, query: str, dataframe: pl.DataFrame = None, chat_history: list = None):
        self.logger.info(f"Processing query: {query}")
        if dataframe is None and self.current_data is None:
            return "Please upload a file first to perform analysis."
        
        active_data = dataframe if dataframe is not None else self.current_data
        
        if isinstance(active_data, str):
            return self.llm_insights.get_text_analysis(query, active_data, chat_history)
        
        # For DataFrame analysis, prepare context for LLM
        try:
            data_summary = self.descriptive_stats.get_summary(active_data).to_pandas().to_markdown(index=False)
            data_head = active_data.head().to_pandas().to_markdown(index=False)
            data_columns = ", ".join(active_data.columns)
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame context: {str(e)}")
            return f"Error processing query: {str(e)}"

        # Construct messages for LLM, including system prompt and chat history
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
            # Handle specific query types
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
            
            # Default to LLM for other queries
            response = self.llm_insights.get_data_analysis(messages)
            return response
        
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"