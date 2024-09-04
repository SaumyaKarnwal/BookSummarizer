from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader

# this module does the following 
# --- extract text from pdf
# --- segment the text
# --- embed the text and ungest it in vector store

def get_pdf_content(docs):
    raw_text = ""
    
    for doc in docs:
        pdf_loader = PyPDFLoader(doc)
        pages = pdf_loader.load()  # Load pages using PyPDFLoader's method
        for page in pages:
            raw_text += page.page_content  # Access page content directly
    
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_storage

# final func
def preprocess(docs):
    
    raw_text = get_pdf_content(docs)

    text_chunks = get_chunks(raw_text)
    
    vector_storage = get_embeddings(text_chunks)
    
    return vector_storage