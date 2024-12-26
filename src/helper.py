import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from llama_cpp import Llama
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationaBufferMemory
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
os.environ["LLAMA_API_KEY"] = LLAMA_API_KEY

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceBgeEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    llm = Llama(model_path="./models/7B/llama-model.gguf")
    memory = ConversationaBufferMemory(memory_key = "chat_history", return_message = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever = vector_store.as_retriever(), memory = memory)
    return conversation_chain
