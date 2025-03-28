import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (for OpenAI API Key)
load_dotenv()

# Streamlit App Title
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# User input: URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# Process button
process_url_clicked = st.sidebar.button("Process URLs")

# FAISS Index Path
faiss_index_path = "faiss_store_openai"

# Placeholder for messages
main_placeholder = st.empty()

# LLM Model
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started...✅✅✅")
    data = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter... Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # ✅ Save FAISS index using built-in method (No Pickle!)
    vectorstore_openai.save_local(faiss_index_path)
    main_placeholder.text("FAISS Index Saved Successfully! 🎉")

# User query input
query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(faiss_index_path):
        embeddings = OpenAIEmbeddings()
        
        # ✅ Load FAISS index safely (No Pickle!)
        vectorstore = FAISS.load_local(
            faiss_index_path, 
            embeddings, 
            allow_dangerous_deserialization=True  # Use only if the file is trusted
        )

        # Retrieve answer
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources (if available)
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split sources by newline
            for source in sources_list:
                st.write(source)
