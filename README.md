<<<<<<< HEAD
# Azure-Rag
# Azure-Rag, Document-Based chat with Conversational AI

## Overview

A Flask web application utilizing the Azure OpenAI API to provide an interface for users to upload documents, create a vector store and engage in conversations based on uploaded content.

## Features

- **Document Upload**: Users can upload multiple `.docx` and `.pdf` files.
- **Vector Store Creation**: The application creates a vector store based on the uploaded documents.
- **Conversational Interface**: Users can engage in conversations, and the application responds based on the content of the uploaded documents.
- **Document Content Extraction**: Extracts text and tables from `.docx` and `.pdf` files.
- **Session Management**: Stores conversation history and assistant data in user sessions.

## Requirements

- `Python 3.8+`
- `Flask`
- `OpenAI Azure API`
- `tiktoken`
- `numpy`
- `pandas`
- `python-docx`
- `pdfplumber`
- `dotenv`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kiazandifar/Azure-Rag.git
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Set up environment variables in a .env file:
   ```bash
   AZURE_API_KEY=your_azure_api_key
   AZURE_API_VERSION=your_azure_api_version
   AZURE_ENDPOINT=your_azure_endpoint

## Usage

1. Start the Flask application:
   ```bash
   python app.py
2. Open your web browser and navigate to http://localhost:5001/.
3. Upload DOCX or PDF files via the provided interface.
4. Interact with the AI assistant through the chat interface to query information based on the uploaded documents.

## Endpoints

- **GET /** - Home page for the web application.
- **POST /upload** - Endpoint for uploading and processing document files.
- **POST /chat** - Endpoint for interacting with the AI assistant.
- **POST /delete_vector_store** - Endpoint to delete vector stores, assistants, and threads.
- **POST /reset** - Endpoint to reset conversation history.

## File Structure

- `app.py`: Main Flask application file.
- `templates/index.html`: HTML template for the web interface.
- `uploads/: Directory` for storing uploaded files (automatically created).

## Key Functions

- `extract_text_and_tables_from_docx(file_path)`: Extracts text and tables from DOCX files.
- `extract_text_and_tables_from_pdf(file_path)`: Extracts text and tables from PDF files.
- `normalize_text(s)`: Normalizes text content.
- `split_text_into_chunks(text, max_tokens, overlap)`: Splits text into chunks for token management.
- Routes for uploading files, interacting with the assistant, and managing sessions.

## Logging

The application uses Python's built-in `logging` module for logging debug and error messages. You can configure logging levels and handlers as needed.

## Security

**Secret Key**: Replace `app.secret_key = 'your_secret_key'` with a secure, random string for session management.

## License
This project is licensed under the MIT License.
