import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "document.pdf")

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
    # Check PDF existence
    if not os.path.exists(PDF_PATH):
        st.error(
            f"Το αρχείο PDF δεν βρέθηκε στο project.\n"
            f"Περίμενα να βρω: {PDF_PATH}"
        )
        return None

    # Load PDF
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs_chunks = splitter.split_documents(docs)

    if not docs_chunks:
        st.error(
            "Δεν βρέθηκε κείμενο στο αρχείο PDF. "
            "Βεβαιώσου ότι το 'document.pdf' δεν είναι κενό ή κρυπτογραφημένο."
        )
        return None

    # Embeddings & Chroma DB
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    )
    db = Chroma.from_documents(docs_chunks, embeddings)

    # LLM & QA chain
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4.1",
        temperature=0,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        return_source_documents=False,
    )
    return qa_chain

qa_chain = setup_knowledge_base()

if qa_chain is not None:
    question = st.text_input("Ρώτα κάτι σχετικά με το έγγραφο:")

    if question:
        with st.spinner("Ψάχνω στα έγγραφα..."):
            result = qa_chain({"query": question})
            answer = result["result"] if "result" in result else str(result)
        st.success(answer)