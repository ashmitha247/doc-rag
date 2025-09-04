# StudyMate - RAG PDF Chatbot

**StudyMate** is an AI-powered study assistant that helps you chat with your PDF documents using advanced RAG (Retrieval-Augmented Generation) technology.

## 🌐 Live Demo
**[🚀 Try StudyMate Live](https://studymate-doc-rag.streamlit.app)** *(Deploy using instructions below)*

## ✨ Features

- 📚 **Multi-PDF Support**: Upload and analyze multiple PDF documents
- 🤖 **Perplexity AI Integration**: Powered by Perplexity's "sonar" model for accurate responses
- 🔍 **OCR Support**: Handles scanned PDFs using Tesseract OCR
- 🎯 **Academic Focus**: Optimized for educational and research content
- 🔒 **Secure API Management**: Environment-based API key storage
- 💡 **Smart Text Processing**: Multiple OCR configurations for difficult documents

## 🚀 Quick Start

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
