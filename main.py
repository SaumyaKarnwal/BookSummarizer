import streamlit as st
from dotenv import load_dotenv
from web_template import css
import context
import conversation
import os

def main():
    load_dotenv()
    
    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # File path
    pdf_path = ["Democratic-Politics-ClassX.pdf"]


    # Extract text and store it into vector stores
    vector_embeddings = context.preprocess(pdf_path)
    
    
    # Start conversation
    if vector_embeddings:
        st.session_state.conversation = conversation.start_conversation(vector_embeddings)

    st.set_page_config(page_title="PDF Summarizer", page_icon=":books:", layout="wide")

    st.write(css, unsafe_allow_html=True)

    st.header("Ask me anything about 10th Class Political Science")
    query = st.text_input("How can I help you today?")

    if query:
        if st.session_state.conversation:
            conversation.process_query(query)
        else:
            st.error("Conversation not initialized.")

if __name__ == "__main__":
    main()
