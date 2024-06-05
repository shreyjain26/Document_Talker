from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import bs4
import os


load_dotenv()


genai.configure(api_key='GOOGLE_API_KEY')


def get_pdf_text(pdf_docs):
    """Returns extracted text from PDF documents"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for pdf in pdf_reader.pages:
            text += pdf.extract_text()
    return text


def get_splitters(chunk_size=2000, overlap=400):
    """Returns parent and child splitters that split the text into different chunk sizes."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter


def load_data_from_website(weblinks):
    """Returns data from website links"""
    weblinks = [weblink for weblink in weblinks.split(',')]
    loader = WebBaseLoader(
        web_paths=(weblinks),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    content = ""
    docs = loader.load()
    for doc in docs:
        content += doc.page_content
    return content


def get_content(docs, weblinks=None, mode="Doc"):
    "Returns a string containing the content of documents and weblinks."
    documents = ""
    if mode == "Doc":
        documents += get_pdf_text(docs)
    
    elif mode == "Web":
        documents = load_data_from_website(weblinks)
    
    elif mode == "combined":
        documents += get_pdf_text(docs)
        documents += load_data_from_website(weblinks)
    return documents


def create_retriever(documents, chunk_size=10000, overlap=1000):
    """Creates a knowledge base that acts as a retriever."""
    text_splitter = get_splitters(chunk_size=chunk_size, overlap=overlap)
    splits = text_splitter.split_text(documents[0])
    gemini_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(
        texts=splits,
        embedding=gemini_embedding,
        )
    vectorstore.save_local("FAISS_index")
    retriever = vectorstore.as_retriever(search_kwargs={"k":3})
    return retriever


def prepare_context():
    """Returns the Context Template for the LLM."""
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return contextualize_q_prompt


def prepare_prompt():
    """Returns the prompt for the given input for the LLM."""
    system_prompt = (
        "Your are an assistant for question-answering tasks."
        "Use the following pieces of retrieved context to answer."
        "If you don't know the answer to a question, say that you do not know it."
        "Output images if you think it would help better understand the answer, specify if the image is from knowledge base or if it is produced by you."
        "Keep the answers concise."
        "\n\n"
        "{context}"
    )

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return contextualize_prompt


def get_retrieval_chain(retriever):
    """Returns the retrieval chain consisting of the model."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        max_output_tokens=512,
        temperature=0.6,
        top_k=3,
        convert_system_message_to_human=True,
    )
    contextual_q_prompt = prepare_context()
    history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextual_q_prompt
    )
    qa_prompt = prepare_prompt()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return retrieval_chain


def rag_chain(docs):
    "Returns the final rag chain pipeline."

    store = {}

    retriever = create_retriever(docs)
    retriever_chain=get_retrieval_chain(retriever) 

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        retriever_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def chat(input, llm_chain):
    "Returns responses to user messages."
    result = llm_chain.invoke(
            {"input": f"{input}"},
            config={
                "configurable": {"session_id": "abc123"}
            },
        )
    return result