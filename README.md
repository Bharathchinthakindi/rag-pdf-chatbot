# ABC Tech HR Assistant

This project is a simple HR assistant built using Streamlit and LangChain.

It answers employee questions based on company HR policies stored in a PDF document.
The application uses a RAG (Retrieval-Augmented Generation) pipeline to retrieve relevant information before generating responses.

## Features

* Ask questions about HR policies
* PDF-based knowledge retrieval
* Semantic search using embeddings
* Clean Streamlit interface
* Fast responses using Groq API

## Technologies Used

* Python
* Streamlit
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Groq API

## How It Works

1. HR policy PDF is loaded
2. Text is split into chunks
3. Embeddings are created
4. ChromaDB stores the vectors
5. Relevant chunks are retrieved for user questions
6. LLM generates the final response

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your API key:

```env
OPENAI_API_KEY=your_api_key_here
```

Run the app:

```bash
streamlit run app.py
```

## Example Questions

* What is the resignation notice period?
* Can employees work remotely?
* How many sick leaves are allowed?

## Future Improvements

* Chat history
* Multiple PDF support
* Better UI customization
* Deployment support
