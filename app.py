import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import base64
import os
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# Configure Tesseract path - flexible for different environments
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:  # Linux/Unix (Streamlit Cloud)
    # Streamlit Cloud should have tesseract in PATH
    pass

# Load environment variables
load_dotenv()

# Update imports for LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate




from datetime import datetime

def get_pdf_text(pdf_docs, debug_ocr=False):
    """
    Extract text from PDF documents using PyMuPDF with OCR fallback.
    """
    text = ""
    total_pages = 0
    total_ocr_pages = 0
    
    for pdf in pdf_docs:
        pdf_bytes = pdf.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        pdf_text = ""
        page_count = pdf_document.page_count  # Store before processing
        ocr_pages = 0
        
        for page_num in range(page_count):
            page = pdf_document[page_num]
            page_text = page.get_text().strip()
            
            # If page has minimal text (likely image-based), use OCR
            if len(page_text) < 50:  # Threshold for image-based pages
                try:
                    if debug_ocr:
                        st.write(f"🔍 OCR processing page {page_num + 1} of {pdf.name}")
                    
                    # Convert PDF page to image for OCR
                    images = convert_from_bytes(pdf_bytes, first_page=page_num+1, last_page=page_num+1, dpi=300)
                    
                    if images:
                        # Try multiple OCR configurations for difficult PDFs
                        ocr_configs = [
                            '--psm 3 --oem 3',  # Default academic configuration
                            '--psm 6 --oem 3',  # Single uniform block
                            '--psm 11 --oem 3', # Sparse text
                            '--psm 13 --oem 3'  # Raw line (treat as single text line)
                        ]
                        
                        best_ocr_text = ""
                        for config in ocr_configs:
                            try:
                                ocr_text = pytesseract.image_to_string(images[0], lang='eng', config=config)
                                if len(ocr_text.strip()) > len(best_ocr_text.strip()):
                                    best_ocr_text = ocr_text
                            except:
                                continue
                        
                        if debug_ocr:
                            st.write(f"📝 OCR extracted {len(best_ocr_text)} characters from page {page_num + 1}")
                        
                        if best_ocr_text.strip():
                            pdf_text += f"{best_ocr_text.strip()}\n"
                            ocr_pages += 1
                            total_ocr_pages += 1
                        else:
                            if debug_ocr:
                                st.warning(f"⚠️ Page {page_num + 1}: OCR could not extract readable text (poor scan quality)")
                            pdf_text += page_text + "\n"
                    else:
                        pdf_text += page_text + "\n"
                        
                except Exception as e:
                    if debug_ocr:
                        st.error(f"OCR failed for page {page_num + 1}: {str(e)}")
                    pdf_text += page_text + "\n"
            else:
                # Use direct text extraction
                pdf_text += page_text + "\n"
            
            total_pages += 1
                
        text += pdf_text
        pdf_document.close()
        
        # Show processing summary (using stored page_count)
        pdf_text_length = len(pdf_text.strip())
        if ocr_pages > 0:
            st.info(f"📄 Processed {pdf.name}: {page_count} pages ({ocr_pages} with OCR)")
        else:
            st.info(f"📄 Processed {pdf.name}: {page_count} pages")
            
        # Warn about poor text extraction
        if pdf_text_length < 100:
            st.warning(f"⚠️ {pdf.name}: Very little text extracted ({pdf_text_length} characters). This may be a poor quality scan.")
        
    if debug_ocr and total_ocr_pages > 0:
        st.write(f"🔍 Total: {len(text)} characters from {total_pages} pages ({total_ocr_pages} OCR pages)")
    elif debug_ocr:
        st.write(f"🔍 Total: {len(text)} characters from {total_pages} pages")
    
    # Final check - if no meaningful text was extracted
    if len(text.strip()) < 10:  # Lowered threshold - 50 was too strict
        st.error("❌ Unable to extract sufficient text from the uploaded PDFs. Please ensure:")
        st.markdown("""
        - PDFs contain readable text (not just images)
        - Scanned PDFs have good quality and contrast
        - PDFs are not password protected or corrupted
        """)
        return None
        
    return text

def get_text_chunks(text, model_name):
    if model_name == "Perplexity":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    elif model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name == "Perplexity":
        # Use local embeddings for Perplexity
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    elif model_name == "Google AI":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    
    if model_name == "Perplexity":
        model = ChatOpenAI(
            model="sonar",
            temperature=0.3,
            openai_api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
    elif model_name == "Google AI":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, model_name, api_key, pdf_docs, conversation_history, debug_ocr=False):
    if api_key is None or pdf_docs is None:
        st.warning("Please upload PDF files and provide API key before processing.")
        return
    
    # Extract text from PDFs
    extracted_text = get_pdf_text(pdf_docs, debug_ocr)
    if extracted_text is None:
        # Error message already shown in get_pdf_text function
        return
        
    text_chunks = get_text_chunks(extracted_text, model_name)
    vector_store = get_vector_store(text_chunks, model_name, api_key)
    user_question_output = ""
    response_output = ""
    
    if model_name == "Perplexity":
        # Use the same local embeddings as in vector store creation
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain("Perplexity", vectorstore=new_db, api_key=api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        user_question_output = user_question
        response_output = response["output_text"]
    elif model_name == "Google AI":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain("Google AI", vectorstore=new_db, api_key=api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        user_question_output = user_question
        response_output = response['output_text']
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

        # conversation_history.append((user_question_output, response_output, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

    # Display the current question and answer
    st.info(f"**Question:** {user_question_output}")
    st.success(f"**Answer:** {response_output}")
    
    if len(conversation_history) == 1:
        conversation_history = []
    elif len(conversation_history) > 1 :
        last_item = conversation_history[-1]  # Son öğeyi al
        conversation_history.remove(last_item) 
    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history):
        with st.expander(f"Q: {question[:100]}..." if len(question) > 100 else f"Q: {question}"):
            st.write(f"**Question:** {question}")
            st.write(f"**Answer:** {answer}")
            st.caption(f"Model: {model_name} | Time: {timestamp} | PDF: {pdf_name}")

    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])

        # df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")
def main():
    st.set_page_config(page_title="StudyMate", page_icon="📚")
    st.header("StudyMate 📚")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    model_name = st.sidebar.radio("Select the Model:", ("Perplexity", "Google AI"))

    api_key = None

    if model_name == "Perplexity":
        # Try to get API key from Streamlit secrets first, then environment variable
        api_key = None
        try:
            api_key = st.secrets["PERPLEXITY_API_KEY"]
        except:
            api_key = os.getenv("PERPLEXITY_API_KEY")
        
        if not api_key:
            api_key = st.sidebar.text_input("Enter your Perplexity API Key:")
            st.sidebar.markdown("Click [here](https://www.perplexity.ai/settings/api) to get an API key.")
        
        if not api_key:
            st.sidebar.warning("Please enter your Perplexity API Key to proceed.")
            return
            
    elif model_name == "Google AI":
        api_key = st.sidebar.text_input("Enter your Google API Key:")
        st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")
        
        if not api_key:
            st.sidebar.warning("Please enter your Google API Key to proceed.")
            return

   
    with st.sidebar:
        st.title("Menu:")
        
        col1, col2 = st.columns(2)
        
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []  # Clear conversation history
            st.session_state.user_question = None  # Clear user question input 
            
            
            api_key = None  # Reset Google API key
            pdf_docs = None  # Reset PDF document
            
        else:
            if clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""  # Temizle
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()  # Son sorguyu kaldır
                else:
                    st.warning("The question in the input will be queried again.")




        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history, False)
        st.session_state.user_question = ""  # Clear user question input 

if __name__ == "__main__":
    main()
