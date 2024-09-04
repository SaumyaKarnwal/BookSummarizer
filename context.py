from pyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# Retrieve the API key from the environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# this module does the following 
# --- extract text from pdf
# --- segment the text
# --- embed the text and ungest it in vector store


# read the PDF content and store it as raw text
def get_pdf_content(docs):
    raw_text =""
    
    for doc in docs :
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    
    return raw_text

# chunking
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size =1000,
        chunk_overlap = 200,
        length_function = len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

# creating an embedding and storing it in the vector DB
def get_embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_storage

# final func
def preproccess(docs):
    
    raw_text = get_pdf_content(docs)

    text_chunks = get_chunks(raw_text)
    
    vector_storage = get_embeddings(text_chunks)
    
    return vector_storage