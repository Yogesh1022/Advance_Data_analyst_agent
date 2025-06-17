Advanced Multimodal Data Analyst Agent

A Streamlit-based data analysis tool that processes and analyzes various file types, including CSV, XLSX, TXT, DOCX, PDF, and images (JPG/PNG). It leverages polars for efficient data handling, Together AI for natural language insights, and Matplotlib/Plotly for visualizations. The agent supports descriptive statistics, trend analysis, correlation detection, text extraction via OCR, and interactive queries through a user-friendly interface.

Features





File Support: Upload and process CSV, XLSX, TXT, DOCX, PDF, and images.



Data Cleaning: Standardizes data types, handles missing values, and removes outliers using polars.



Analysis: Generates descriptive statistics, identifies time-series trends, and computes correlations.



Visualization: Creates bar and line charts with Matplotlib and Plotly.



Text Extraction: Extracts text from PDFs, DOCX, and images using pytesseract (default) or easyocr.



LLM Integration: Answers queries with insights powered by Together AI’s LLM.



Streamlit UI: Interactive web interface for file uploads and natural language queries.



Output Storage: Saves cleaned datasets and extracted text to an output/ directory.

Prerequisites





Python 3.8 or higher



Git installed (download)



Tesseract-OCR installed (download)



A Together AI API key (sign up)

Setup





Clone the Repository:

git clone https://github.com/your-username/data-analyst-agent.git
cd data-analyst-agent



Create a Virtual Environment:

python -m venv venv





On Windows:

venv\Scripts\activate



On Unix/Linux/Mac:

source venv/bin/activate



Install Dependencies:

pip install -r requirements.txt