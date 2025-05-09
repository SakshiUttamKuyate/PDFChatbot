import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# Load environment variables


# Sidebar UI
with st.sidebar:
    st.title('🤗💬 Chat with your Data')
    st.markdown(''' 
        ## About 
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [Hugging Face Transformers](https://huggingface.co/models)
    ''')

# Function to truncate the context to avoid token limit issues
def truncate_context(text, max_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# Main App
def main():
    st.header("Chat with your own PDF")

    # Session state setup
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None
        st.session_state['texts'] = None
        st.session_state['chat_history'] = []

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None and st.session_state['vectorstore'] is None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_text(text)
        st.success("PDF processed and text split into chunks.")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts, embedding=embeddings)

        st.session_state['vectorstore'] = vectorstore
        st.session_state['texts'] = texts

    # Chat input
    if st.session_state['vectorstore'] is not None:
        st.subheader("Chat Interface")
        user_input = st.chat_input("Ask a question about your PDF")

        if user_input:
            # Save user question
            st.session_state['chat_history'].append({"role": "user", "content": user_input})

            # Perform similarity search
            docs = st.session_state['vectorstore'].similarity_search(user_input, k=3)
            raw_context = "\n\n".join([doc.page_content for doc in docs])
            context = truncate_context(raw_context, max_tokens=512)

            # Load model and tokenizer
            model_name = "google/flan-t5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(inputs["input_ids"], max_length=256, temperature=0.7, do_sample=True)

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Save response
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

        # Display chat history
        for message in st.session_state['chat_history']:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])

if __name__ == '__main__':
    main()
