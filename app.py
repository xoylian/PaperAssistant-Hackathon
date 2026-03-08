import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="PaperAssistant", layout="centered")

st.title("📄 PaperAssistant")
st.markdown(
    """
    Καλωσήρθατε στο **PaperAssistant**!  
    Ένα RAG σύστημα για αναζήτηση και ερωτήσεις σε νομικά έγγραφα.
    """
)

@st.cache_resource
def setup_knowledge_base():
    # Load PDF
    loader = PyPDFLoader("document.pdf")
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    docs_chunks = splitter.split_documents(docs)

    # Embeddings & Chroma DB
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(docs_chunks, embeddings)

    # LLM & QA chain
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        return_source_documents=False,
    )
    return qa_chain

qa_chain = setup_knowledge_base()

question = st.text_input("Ρώτα κάτι σχετικά με το έγγραφο:")

if question:
    with st.spinner("Ψάχνω στα έγγραφα..."):
        result = qa_chain({"query": question})
        answer = result["result"] if "result" in result else str(result)
    st.success(answer)
