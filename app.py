from dotenv import load_dotenv

load_dotenv()

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_database(file_path: str) -> None:
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        documents=chunks,
        embedding=create_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def load_retriever():
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=create_embeddings(),
    )

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )


def get_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
""",
            ),
        ]
    )


st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if "db_ready" not in st.session_state:
    st.session_state.db_ready = os.path.exists(CHROMA_DIR)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):
        with st.spinner("Processing document..."):
            create_vector_database(file_path)

        st.session_state.db_ready = True
        st.session_state.pop("retriever", None)
        st.success("Vector database created!")

if st.session_state.db_ready:
    if "retriever" not in st.session_state:
        st.session_state.retriever = load_retriever()

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:
        docs = st.session_state.retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = get_prompt().invoke({
            "context": context,
            "question": query,
        })

        llm = ChatMistralAI(model="mistral-small-latest")
        response = llm.invoke(final_prompt)

        st.write("### AI Answer")
        st.write(response.content)
