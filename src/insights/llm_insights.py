import logging
from together import Together
from typing import List, Dict, Optional

class LLMInsights:
    def __init__(self, api_key: str, model_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize LLMInsights with Together AI client.
        
        Args:
            api_key (str): Together AI API key.
            model_name (str): Name of the LLM model (e.g., 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free').
            logger (logging.Logger, optional): Logger instance for logging.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.client = Together(api_key=api_key)
        self.model_name = model_name

    def get_text_analysis(self, query: str, text: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Analyze text content using the Together AI LLM.
        
        Args:
            query (str): User's query about the text.
            text (str): Text content to analyze (e.g., from TXT, DOCX, PDF, or image OCR).
            chat_history (List[Dict[str, str]], optional): Previous chat messages for context.
        
        Returns:
            str: LLM's response to the query.
        """
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
        """
        Analyze data-related queries using the Together AI LLM.
        
        Args:
            messages (List[Dict[str, str]]): List of messages including system prompt, data context, and query.
        
        Returns:
            str: LLM's response to the data query.
        """
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