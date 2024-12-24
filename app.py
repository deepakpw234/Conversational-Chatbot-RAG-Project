# RAG QA conversational chatbot including chat history
import os
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGING_FACE_API_KEY"]=os.getenv("HUGGING_FACE_API_KEY")


embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

## Setting our streamlit app

st.title("Conversational RAG with pdf's upload and chat history")
st.write("Upload PDF's and chat with content")

groq_api_key = st.text_input("Enter your api key:",type="password")

if groq_api_key:

    llm = ChatGroq(model="Gemma2-9b-It",api_key=groq_api_key)

    session_id = st.text_input("Session_ID",value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    
    upload_files = st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)

    if upload_files:
        documents = []    
        for upload_file in upload_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(upload_file.getvalue())
                file_name=upload_file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits,embedding=embedding)
        retriever = vectorstore.as_retriever()


        contextulized_q_system_prompt = (
            "Given a chat history and latest user question "
            "which might reference in the context of the chat hisotory "
            "formulate a standalone question which can be understood "
            "without a chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return as it is"
        )

        contextulized_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextulized_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextulized_q_prompt)

        # Answer question

        system_prompt = (
            "You are an assitance for question-answering task. "
            "Use the following pieces of retrieved context ot answer "
            "the question. If you don't know the answer then say that "
            " you don't know. Use three sentences maximum and keep it concise"
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()

            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Enter your question:")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke({
                "input":user_input},
                config={"configurable":{"session_id":session_id}}
            )

            # st.write(st.session_state.store)
            st.write(response['answer'])
            # st.write("chat history",session_history.messages)

else:
    st.warning("Please enter the groq api key")