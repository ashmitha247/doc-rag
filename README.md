# StudyMate - RAG PDF Chatbot

**StudyMate** is an AI-powered study assistant that enables intelligent conversations with PDF documents using advanced Retrieval-Augmented Generation (RAG) technology.

## ✨ Features

- 📚 **Multi-PDF Support**: Upload and analyze multiple PDF documents simultaneously
- 🤖 **Perplexity AI Integration**: Powered by Perplexity's "sonar" model for accurate, context-aware responses
- 🔍 **Advanced OCR Support**: Handles scanned PDFs using Tesseract OCR with multiple configurations
- 🎯 **Academic Focus**: Optimized for educational and research content processing
- 🔒 **Secure API Management**: Environment-based API key storage and Streamlit secrets support
- 💡 **Smart Text Processing**: Adaptive text extraction with fallback mechanisms
- 🧠 **Vector Search**: FAISS-powered semantic search for relevant document chunks

## � Live Demo

Try StudyMate directly in your browser without any setup:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://doc-rag.streamlit.app/)

**⏱️ Response Time**: Please allow 1-2 minutes for query processing as the system performs comprehensive document analysis and AI-powered response generation.

## �🛠 Technology Stack

### **Core Framework**
- **Streamlit**: Web application framework for interactive UI
- **Python 3.13+**: Core programming language

### **AI & LLM Integration**
- **LangChain**: Framework for building LLM applications
  - `langchain-core`: Core abstractions and interfaces
  - `langchain-community`: Community-contributed integrations
  - `langchain-openai`: OpenAI-compatible API wrapper for Perplexity
- **Perplexity AI**: Large language model using "sonar" model for responses
- **OpenAI API Compatibility**: Seamless integration through langchain-openai

### **Document Processing**
- **PyMuPDF (fitz)**: High-performance PDF text extraction and manipulation
- **Tesseract OCR**: Optical Character Recognition for scanned documents
- **pytesseract**: Python wrapper for Tesseract OCR engine
- **pdf2image**: PDF to image conversion for OCR preprocessing
- **OpenCV**: Advanced image preprocessing for OCR optimization

### **Vector Database & Embeddings**
- **FAISS**: Facebook AI Similarity Search for efficient vector operations
- **SentenceTransformers**: Local embedding generation using "all-MiniLM-L6-v2" model
- **Transformers**: Hugging Face transformers library for embedding models

### **Text Processing & Chunking**
- **LangChain Text Splitters**: Intelligent document chunking strategies
- **Recursive Character Text Splitter**: Context-aware text segmentation
- **Custom text processing**: Academic-optimized preprocessing pipelines

### **Supporting Libraries**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing foundation
- **Pillow (PIL)**: Image processing and manipulation
- **python-dotenv**: Environment variable management

## 🔄 Application Flow

### **1. Document Ingestion**
```
PDF Upload → Text Extraction → OCR Fallback → Text Validation
```
- **Primary**: PyMuPDF extracts text directly from text-based PDFs
- **Fallback**: Tesseract OCR processes scanned/image-based content
- **Validation**: Quality checks ensure extracted text meets minimum standards

### **2. Text Processing Pipeline**
```
Raw Text → Chunking → Embedding Generation → Vector Storage
```
- **Chunking**: LangChain's RecursiveCharacterTextSplitter breaks documents into semantic chunks
- **Embedding**: SentenceTransformers generates dense vector representations locally
- **Storage**: FAISS indexes vectors for fast similarity search

### **3. Query Processing**
```
User Question → Vector Search → Context Retrieval → LLM Generation → Response
```
- **Similarity Search**: FAISS finds most relevant document chunks
- **Context Assembly**: Retrieved chunks form context for the LLM
- **Response Generation**: Perplexity AI generates contextual answers

### **4. OCR Enhancement Pipeline**
```
Low-Quality Text → Image Preprocessing → Multi-Config OCR → Text Validation
```
- **Preprocessing**: OpenCV applies noise reduction, contrast enhancement
- **Multi-Configuration**: Different PSM (Page Segmentation Mode) settings
- **Quality Control**: Validation ensures OCR output meets academic standards

## 🏗 Architecture Components

### **Frontend Layer**
- **Streamlit Interface**: File upload, progress tracking, chat interface
- **Session Management**: Persistent conversation history and document state
- **Error Handling**: Graceful degradation for processing failures

### **Processing Layer**
- **Document Parser**: Multi-format PDF handling with intelligent fallbacks
- **OCR Engine**: Tesseract integration with preprocessing optimization
- **Text Validator**: Quality assurance for extracted content

### **AI Layer**
- **Vector Database**: FAISS for similarity search and retrieval
- **Embedding Model**: Local SentenceTransformers for document vectorization
- **LLM Integration**: Perplexity API through LangChain abstractions

### **Configuration Layer**
- **Environment Management**: API keys and configuration through .env and Streamlit secrets
- **Cross-Platform Support**: OS-aware Tesseract path configuration
- **Dependency Management**: Comprehensive requirements with version pinning
