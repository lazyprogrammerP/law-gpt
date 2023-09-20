from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=512)


def load_txt(document_path: str):
    loader = TextLoader(file_path=document_path)
    return loader.load_and_split(text_splitter=TEXT_SPLITTER)


def load_pdf(document_path: str):
    loader = PyPDFLoader(file_path=document_path)
    return loader.load_and_split(text_splitter=TEXT_SPLITTER)
