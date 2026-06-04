import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

st.title("🏢 ABC Tech HR Assistant")

@st.cache_resource
def build_rag():

    loader = PyPDFLoader("hr_policies_dataset.pdf")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=40
    )

    docs = splitter.split_documents(documents)

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    if os.path.exists("./chroma_store"):

        db = Chroma(
            persist_directory="./chroma_store",
            embedding_function=embedding_model
        )

    else:

        db = Chroma.from_documents(
            docs,
            embedding=embedding_model,
            persist_directory="./chroma_store"
        )

    retriever = db.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an HR assistant for ABC Technologies.

Answer only from the provided HR policies.

If the answer is not available, say:
"This is not covered in HR policies."
"""
        ),

        (
            "human",
            """
Context:
{context}

Question:
{query}
"""
        )
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    rag_chain = (
        {
            "context": retriever | (
                lambda docs: "\n\n".join(
                    d.page_content for d in docs
                )
            ),

            "query": RunnablePassthrough()
        }

        | prompt
        | llm
    )

    return rag_chain

rag_chain = build_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input(
    "Ask about company policies..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching policies..."):

            try:

                response = rag_chain.invoke(question)

                answer = response.content

                st.write(answer)

            except Exception as e:

                answer = f"Error: {e}"

                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
    