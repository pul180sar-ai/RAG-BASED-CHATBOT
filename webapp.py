import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # To perform word embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter # This for chunking
from pypdf import PdfReader
import faiss
import streamlit as st 
from pdfextractor import text_extractor_pdf

# Load PDF in Side Bar 
st.sidebar.title(':orange[UPLOAD YOUR DOCUMENT HERE (PDF only)]')
file_uploaded = st.sidebar.file_uploader('upload File')
if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)

# Create the main page
st.title(':green[RAG Based CHATBOT]')
tips ='''Follow the steps to use this application:
*Ipload your pdf document in sidebar
* Write your query and start chatting'''
st.write(tips)

# Step 1 : Configure the model

# Configure LLM
key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)
llm_model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Configure  Embedding Model
embedding_model =HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

 
# Step 2 : Chunking (Create chunks)
splitter = RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap = 200)
chunks = splitter.split_text(file_text)

# Step 4 : Create FAISS Vector Store
vector_store = FAISS.from_texts(chunks,embedding_model)

# Step 5 : Configure retriever
retriever = vector_store.as_retriever(search_kwargs={"k":3})

while True:
    query =  st.chat_input('User: ')
    if query.lower() in ['bye','exit','quit','close','end','stop']:
        break 
