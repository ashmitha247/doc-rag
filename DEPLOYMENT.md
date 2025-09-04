# StudyMate Deployment Guide

## 🌐 Live Demo
**[StudyMate - Live App](https://studymate-rag.streamlit.app)** *(Will be available after deployment)*

## 🚀 Quick Start

### Local Installation
```bash
# Clone the repository
git clone https://github.com/ashmitha247/doc-rag.git
cd doc-rag

# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "PERPLEXITY_API_KEY=your_api_key_here" > .env

# Run the app
streamlit run app.py
```

### Access the App
- **Local URL**: http://localhost:8505
- **Network URL**: http://your-ip:8505 (accessible to devices on same network)

## 🌟 Streamlit Community Cloud Deployment

1. **Fork/Star this repository**
2. **Visit [share.streamlit.io](https://share.streamlit.io)**
3. **Connect your GitHub account**
4. **Deploy from `ashmitha247/doc-rag`**
5. **Add your Perplexity API key in Streamlit secrets**

## 📱 Mobile Access
The app is fully responsive and works on mobile devices!

## 🔧 Configuration
- Upload size limit: 200MB
- Supported formats: PDF (text + scanned with OCR)
- AI Model: Perplexity "sonar" model
