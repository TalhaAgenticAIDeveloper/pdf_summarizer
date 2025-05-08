import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key="AIzaSyDlGuiJOqQePVsQEu5gWiftb74RDGvcq-c")

# Streamlit Page Setup
st.set_page_config(page_title="AI PDF Summarizer", layout="wide")
st.title("PDF AI Summarizer")

# PDF Text Extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                if page.extract_text():
                    text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

# Text Chunking here
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Embedding and Vector Store Creation
@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyDlGuiJOqQePVsQEu5gWiftb74RDGvcq-c")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Conversational Chain Setup
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, "answer is not available in the provided context." Do not provide a wrong answer.
    Context: {context}
    Question: {question}
    Answer:
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

# User Query Handling
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings ,  allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# PDF Interface
pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
if st.button("Submit & Process PDF") and pdf_docs:
    with st.spinner("Processing PDF..."):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("PDF processing completed.")

if pdf_docs:
    user_query = st.text_input("Ask a Question about the uploaded content")
    if user_query:
        with st.spinner("Processing your PDF query..."):
            response = user_input(user_query)
            st.success("Query processed successfully!")
            st.write("Reply:", response)
