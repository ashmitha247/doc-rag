# StudyMate - RAG PDF Chatbot

**StudyMate** is an AI-powered study assistant that enables intelligent conversations with## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 👩‍💻 Author

**ashmitha247** - *StudyMate Development*ments using advanced Retrieval-Augmented Generation (RAG) technology.

## ✨ Features

- 📚 **Multi-PDF Support**: Upload and analyze multiple PDF documents simultaneously
- 🤖 **Perplexity AI Integration**: Powered by Perplexity's "sonar" model for accurate, context-aware responses
- 🔍 **Advanced OCR Support**: Handles scanned PDFs using Tesseract OCR with multiple configurations
- 🎯 **Academic Focus**: Optimized for educational and research content processing
- � **Secure API Management**: Environment-based API key storage and Streamlit secrets support
- 💡 **Smart Text Processing**: Adaptive text extraction with fallback mechanisms
- 🧠 **Vector Search**: FAISS-powered semantic search for relevant document chunks
## 🛠 Technology Stack

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

## 🔍 Technical Highlights

### **RAG Implementation**
- **Semantic Chunking**: Context-preserving document segmentation
- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Context Window Optimization**: Efficient prompt construction for LLM calls

### **OCR Optimization**
- **Adaptive Configuration**: Multiple Tesseract PSM modes for different document types
- **Image Preprocessing**: OpenCV-based enhancement for better OCR accuracy
- **Quality Validation**: Text quality scoring and re-processing triggers

### **Performance Features**
- **Local Embeddings**: No external API calls for vectorization
- **Efficient Storage**: FAISS optimized vector operations
- **Caching**: Session-based caching for processed documents

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 👩‍� Author

**ashmitha247** - *StudyMate Development*

### 1. Create Virtual Environment

```bash
python -m venv myenv
myenv/scripts/activate  # On Windows
# or
source myenv/bin/activate  # On Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Environment Variables

Create a `.env` file in the root directory:
```
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will be available at:
- **Local**: http://localhost:8505
- **Network**: http://your-ip:8505 (for LAN access)

## 🌐 Deploy to Streamlit Cloud (FREE)

1. **Fork this repository** to your GitHub account
2. **Visit**: https://share.streamlit.io
3. **Connect GitHub** and select this repository
4. **Add secrets** in Streamlit dashboard:
   ```
   PERPLEXITY_API_KEY = "your_api_key_here"
   ```
5. **Deploy!** Your app will be live at: `https://your-app-name.streamlit.app`

### 4. Install Tesseract OCR

- **Windows**: Download from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **Mac**: `brew install tesseract`

### 5. Run the Application

```bash
streamlit run app.py
```

## 📖 How to Use

1. **Upload PDFs**: Use the file uploader to select your PDF documents
2. **Process**: Click "Submit & Process" to extract and index content
3. **Ask Questions**: Type your questions about the document content
4. **Get Answers**: Receive detailed, contextual responses from StudyMate

## 🛠 Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Perplexity API (sonar model)
- **PDF Processing**: PyMuPDF (fitz)
- **OCR**: Tesseract with pytesseract
- **Vector Database**: FAISS
- **Embeddings**: SentenceTransformers
- **Text Processing**: LangChain

## 📋 Requirements

- Python 3.8+
- Perplexity API Key
- Tesseract OCR installed
- See `requirements.txt` for full dependencies

## 🔧 Configuration

The application automatically detects and handles:
- Text-based PDFs (direct text extraction)
- Image-based/scanned PDFs (OCR processing)
- Mixed content PDFs (combination approach)

## 💡 Tips for Best Results

- Use high-quality PDF scans for better OCR results
- Academic papers and textbooks work exceptionally well
- Enable debug mode to see processing details
- Ensure PDFs are not password-protected

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## �‍💻 Author

**ashmitha247** - *StudyMate Development*

## �📄 License

This project is open source and available under the MIT License.
