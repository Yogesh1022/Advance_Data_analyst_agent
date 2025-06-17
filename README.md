Advanced Multimodal Data Analyst Agent
A powerful Streamlit-based application for analyzing and visualizing data from various file formats, including CSV, XLSX, TXT, DOCX, PDF, and images (JPG/PNG). Built with polars for efficient data processing, Together AI for natural language insights, and Matplotlib/Plotly for visualizations, this agent supports data cleaning, statistical analysis, trend detection, correlation analysis, and text extraction via OCR. Users interact through an intuitive web interface to upload files and query data in natural language.
Features

Multimodal File Support: Process CSV, XLSX, TXT, DOCX, PDF, and image files.
Data Cleaning: Automatically standardizes data types, handles missing values, and removes outliers using polars.
Analysis Capabilities:
Descriptive statistics (e.g., mean, min, max).
Time-series trend analysis.
Correlation analysis for numerical columns.


Visualizations: Generate bar and line charts with Matplotlib and Plotly.
Text Extraction: Extract text from PDFs, DOCX, and images using pytesseract (default) or easyocr.
LLM-Powered Insights: Answer complex queries using Together AI’s language model.
Streamlit Interface: User-friendly web UI for file uploads and interactive queries.
Output Management: Save cleaned datasets and extracted text to the output/ directory.

Prerequisites

Python: Version 3.8 or higher.
Git: Installed for cloning the repository (download).
Tesseract-OCR: Required for image text extraction (download).
Together AI API Key: Obtain from Together AI.

Installation

Clone the Repository:
git clone https://github.com/your-username/data-analyst-agent.git
cd data-analyst-agent


Set Up a Virtual Environment:
python -m venv .


On Windows:.\Scripts\activate


On Unix/Linux/Mac:source .bin/activate




Install Dependencies:
pip install -r requirements.txt


Configure Environment Variables:

Create a .env file in the project root:echo TOGETHER_API_KEY=your_api_key > .env
echo TESSERACT_CMD_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe >> .env


Replace your_api_key with your Together AI API key.
Update TESSERACT_CMD_PATH to match your Tesseract installation (e.g., /usr/bin/tesseract on Linux).


Run the Application:
streamlit run data_analyst_agent.py --server.fileWatcherType none


Access the app at http://localhost:8501 in your browser.



Usage

Upload Files:

In the Streamlit sidebar, upload a file (CSV, XLSX, TXT, DOCX, PDF, or image).
File size limit: 100 MB.
Cleaned datasets and extracted text are saved in the output/ directory (e.g., cleaned_data.csv, extracted_text.txt).


Query the Data:

Use the chat input box to ask questions in natural language. Examples:
"What are the column names and data types?"
"Summarize this dataset."
"Calculate the average, minimum, and maximum values for numerical columns."
"Find correlations between numerical columns."
"Analyze trends in sales over time using the date column."
"Extract text from this image."
For health data: "What is the average age of patients?"
For visualizations: "Show a bar chart of heart attack risk by age group."




View Results:

Results include text responses, data previews, or interactive charts.
Logs are saved in data_analyst_agent.log for debugging.



Project Structure
data_analyst_agent/
├── data_analyst_agent.py  # Core application code
├── .env                   # Environment variables (git-ignored)
├── .gitignore             # Excludes sensitive/unnecessary files
├── requirements.txt       # Dependency list
├── README.md              # Project documentation
├── output/                # Processed files (git-ignored)
├── data_analyst_agent.log # Log file (git-ignored)
├── venv/                  # Virtual environment (git-ignored)

Example Workflow

Create a sample CSV:echo "age,cholesterol,heart_risk\n45,200,0\n50,250,1\n60,300,1" > heart_attack.csv


Run the app:streamlit run data_analyst_agent.py --server.fileWatcherType none


Upload heart_attack.csv via the Streamlit sidebar.
Query:
"Summarize this dataset."
"Find correlations between cholesterol and heart_risk."
"Plot a bar chart of heart_risk by age."


Upload an image and query: "Extract text from this image."
Check output/ for saved files.

Troubleshooting

CSV Encoding Issues (e.g., Mobiles.csv):
Open the file in a text editor (e.g., Notepad++) and save with UTF-8 encoding.


Tesseract Errors:
Verify TESSERACT_CMD_PATH points to the Tesseract executable.


Together AI API Issues:
Test the API key:python -c "from together import Together; print(Together(api_key='your_api_key').models.list())"




Dependency Problems:
List installed packages:pip list


Reinstall dependencies:pip install -r requirements.txt




Streamlit Cache:
Clear cache if issues persist:streamlit cache clear





Contributing

Fork the repository.
Create a feature branch:git checkout -b feature/your-feature


Commit changes:git commit -m "Add your feature"


Push to the branch:git push origin feature/your-feature


Open a Pull Request on GitHub.

License
This project is licensed under the MIT License (see LICENSE if included).
Acknowledgments

Powered by Streamlit, Polars, Together AI, Matplotlib, and Plotly.
OCR capabilities via Tesseract and EasyOCR.

