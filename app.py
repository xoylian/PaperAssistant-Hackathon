import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (custom api keys / base_urls από τους διοργανωτές)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)  # optional custom base_url

st.set_page_config(page_title="PaperAssistant: GovTech AI Assistant", layout="centered")

st.title("📄 PaperAssistant: Ο AI Σύμβουλος Επιχειρήσεων")
st.markdown(
    "Ρωτήστε με για οποιαδήποτε διαδικασία του Ελληνικού Δημοσίου. Ψάχνω ζωντανά για τους πιο πρόσφατους νόμους."
)

# --- Συστήματα προετοιμασίας (lazy load για ανάγκη) ---
@st.cache_resource
def get_agent_executor():
    """Δημιουργεί τον PaperAssistant Agent με DuckDuckGo και AgentExecutor."""
    from langchain.chat_models import ChatOpenAI
    from langchain.agents import initialize_agent, AgentType
    from langchain.memory import ConversationBufferMemory
    from langchain_community.tools import DuckDuckGoSearchRun
    tools = [DuckDuckGoSearchRun()]
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    llm_kwargs = {
        "model_name": "gpt-4.1",
        "temperature": 0,
        "openai_api_key": OPENAI_API_KEY,
    }
    if OPENAI_API_BASE:
        llm_kwargs["openai_api_base"] = OPENAI_API_BASE
    llm = ChatOpenAI(**llm_kwargs)
    # Συστηματικό μήνυμα: το "μυστικό" του PaperAssistant
    system_message = (
        "Είσαι ο 'PaperAssistant', ένας κορυφαίος AI σύμβουλος επιχειρήσεων, εξειδικευμένος στα προγράμματα ΕΣΠΑ και τις επιδοτήσεις του Ελληνικού Κράτους. "
        "ΑΠΑΓΟΡΕΥΕΤΑΙ ΑΥΣΤΗΡΑ να δίνεις γενικές απαντήσεις ή να παραπέμπεις τον χρήστη στις αρχικές σελίδες (π.χ. σκέτο espa.gr) για να ψάξει μόνος του. "
        "Ο ρόλος σου είναι να ψάχνεις στο internet και να βρίσκεις τα ΣΥΓΚΕΚΡΙΜΕΝΑ προγράμματα που είναι ΕΝΕΡΓΑ ΑΥΤΗ ΤΗ ΣΤΙΓΜΗ. "
        "Στην απάντησή σου πρέπει να αναφέρεις: 1) Το ακριβές όνομα του προγράμματος. 2) Το ποσοστό ή το ποσό της επιδότησης. 3) Τις προθεσμίες. "
        "Αν δεν υπάρχει κανένα ενεργό πρόγραμμα σήμερα, πρέπει να το πεις ξεκάθαρα. Στο τέλος, δώσε το URL της ΣΥΓΚΕΚΡΙΜΕΝΗΣ προκήρυξης και όχι της αρχικής σελίδας."
    )

    # Agent με memory (CHAT_CONVERSATIONAL) + AgentExecutor (max_iterations=3, handle_parsing_errors)
    executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        agent_kwargs={"system_message": system_message},
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=False,
    )
    return executor


# --- Αρχικοποίηση session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Εμφάνιση ιστορικού chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Ρωτήστε για ΕΣΠΑ, επιδοτήσεις, ή νόμους για επιχειρήσεις..."):
    if not OPENAI_API_KEY:
        st.error("Ορίστε το OPENAI_API_KEY στο .env")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        total_tokens = 0
        with st.spinner("Ο PaperAssistant ψάχνει..."):
            try:
                from langchain_community.callbacks import get_openai_callback
                executor = get_agent_executor()
                with get_openai_callback() as cb:
                    result = executor.invoke({"input": prompt})
                    reply = result.get("output", str(result))
                    total_tokens = cb.total_tokens
            except Exception as e:
                reply = f"Σφάλμα κατά την επεξεργασία: {e}"
        st.markdown(reply)
        st.caption(f"Καταναλώθηκαν: {total_tokens} tokens")

    st.session_state.messages.append({"role": "assistant", "content": reply})
