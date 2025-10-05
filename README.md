# MENADevsGenAIHackathon
To run the project, you need to install everything mentioned in the requirements_new.txt file, along with ollama llama3.2:1b and fastapi.
Place the files in the appropriate structure, then run the app_fastapi.py file.

http://localhost:8001/

Of course, make sure Python and HTML (and any other required system files) are installed.

Technologies used in the project:

Natural Language Processing (NLP) – Core
Text Extraction: from PDF, Word:

PyPDF2 for PDF files

python-docx for Word files


Language Model (LLM) – Ollama
Using Ollama with the llama3.2:1b model for:

Document summarization

Key information extraction

Converting texts into organized key-value pairs

Rule-Based NLP
Natural language processing using predefined rules instead of relying on large machine learning models:

Regex-based pattern matching

Document type detection

Field extraction

LLM Integration

Direct HTTP requests to Ollama

Streaming responses

Simple prompt engineering

Data Processing:

Normalization of dates and currencies

Redaction of sensitive information

File format conversion
