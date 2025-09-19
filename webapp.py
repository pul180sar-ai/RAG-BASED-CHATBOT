import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings # To perform word embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter # This for chunking
from pypdf import PdfReader
import faiss
import streamlit as st 
from pdfextractor import text_extractor_pdf

# Create the main page
st.title(':green[RAG Based CHATBOT]')
tips ='''Follow the steps to use this application:
* Upload your pdf document in sidebar.
* Write your query and start chatting.'''
st.write(tips)

# Load PDF in Side Bar 
st.sidebar.title(':orange[UPLOAD YOUR DOCUMENT HERE (PDF only)]')
file_uploaded = st.sidebar.file_uploader('upload File')
if file_uploaded:
    file_text = text_extractor_pdf(file_uploaded)

   

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

    # Lets create a function that takes query and return the generated text
    def generate_response(query):
         # Step 7 : Retrieval (R)

        retrived_docs = retriever.get_relevant_documents(query=query)
        context = ' '.join([doc.page_content for doc in retrived_docs])
    
        # Step 8 : Write a Augmeneted prompt (A)
        prompt= f'''You are a helpful assistant using RAG. Here is the context {context}
        The query asked by user is as follows = {query}'''

        # Step 9: Generation (G)
        content = llm_model.generate_content(prompt)
        return content.text


    # Lets create chatbot in order to start the  converstation
    # Initialize chat if there is no history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Display the history
    for msg in st.session_state.history:
        if msg['role'] == 'user':
            st.write(f'### User: {msg['text']}')
        else:
            st.write(f'### Chatbot: {msg['text']}')
    
    # Input from the user (Using Streamlit Form)
    with st.form('Chat Form',clear_on_submit=True):
        user_input = st.text_input('Enter Your Text Here:')
        send = st.form_submit_button('Send')
    
    # Start the conversation  and append the output and query in history
    if user_input and send:

        st.session_state.history.append({"role":'user',"text":user_input})

        model_output = generate_response(user_input)

        st.session_state.history.append({'role':'chatbot','text':model_output})

        st.rerun()



    