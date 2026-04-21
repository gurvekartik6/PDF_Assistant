# AI PDF Assistant

A comprehensive AI-based application that enables users to upload PDF documents and interact with them using natural language. The system supports question answering, summarization, notes generation, flashcards, and voice interaction.

---

## Features

* Upload single or multiple PDF documents
* Ask questions based on document content (RAG-based system)
* Generate summaries (short and detailed)
* Create structured notes
* Generate flashcards (question–answer format)
* Multi-document semantic search
* Chat history with memory support
* Voice input (speech-to-text)
* Voice output (text-to-speech)
* Display source references from documents

---

## Core Concept

This project is based on Retrieval-Augmented Generation (RAG):

1. Extract text from PDFs
2. Split text into manageable chunks
3. Convert text into embeddings
4. Store embeddings in a vector database
5. Retrieve relevant chunks for a query
6. Generate responses using a large language model

---

## Tech Stack

### Backend

* Python
* LangChain
* FAISS (Vector Database)
* Hugging Face Embeddings
* OpenRouter API (LLM access)

### Frontend

* Streamlit

### Voice Support

* Whisper (Speech-to-Text)
* pyttsx3 or gTTS (Text-to-Speech)

---

## Project Structure

```
project/
│── app.py                 # Main Streamlit application
│── pdf_loader.py          # PDF loading and parsing
│── embeddings.py          # Embedding generation
│── vector_store.py        # FAISS database handling
│── qa_chain.py            # Retrieval and LLM logic
│── voice.py               # Voice input/output
│── utils.py               # Helper functions
│── requirements.txt       # Project dependencies
│── README.md              # Documentation
```

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/your-username/ai-pdf-assistant.git
cd ai-pdf-assistant
```

### 2. Create a Virtual Environment

```
python -m venv venv
source venv/bin/activate   (Linux/Mac)
venv\Scripts\activate      (Windows)
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the root directory:

```
OPENROUTER_API_KEY=your_api_key_here
```

---

## Running the Application

```
streamlit run app.py
```

The application will be accessible in your browser after execution.

---

## Workflow

1. Upload one or more PDF documents
2. The system extracts and processes the text
3. Text is converted into embeddings
4. Embeddings are stored in a FAISS vector database
5. The user submits a query
6. Relevant content is retrieved
7. The language model generates a contextual response

---

## Use Cases

* Academic study assistance
* Technical documentation analysis
* Research paper summarization
* Knowledge base interaction

---

## Limitations

* Performance depends on document quality
* Large documents may increase processing time
* Requires proper chunking and prompt configuration

---

## Future Improvements

* Cloud deployment (Docker or containerized environments)
* Advanced frontend using modern web frameworks
* User authentication and access control
* Persistent cloud-based vector database
* Model selection and configuration interface

---

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request with appropriate documentation.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* LangChain
* Hugging Face
* FAISS
* OpenRouter

---

## Contact

For inquiries or collaboration, please reach out through appropriate professional channels.
