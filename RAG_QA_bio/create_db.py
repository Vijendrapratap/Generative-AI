import os, shutil
from glob import glob
from pathlib import Path
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = "data"  # The directory containing input PDF documents
CHROMA_PATH = "chroma"  # The directory to store the Chroma database

def load_documents():
    """Loads PDF documents from the specified data directory.

    Returns:
        list[Document]: A list of Langchain Document objects representing the loaded PDFs.
    """

    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")  # Loads PDFs from 'data' directory
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    """Splits documents into smaller chunks for efficient embedding.

    Args:
        documents: A list of Langchain Document objects.

    Returns:
        list[Document]: A list of smaller documents representing the chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Target chunk size (can be adjusted for your data)
        chunk_overlap=100,  # Overlap between chunks for smoother transitions
        length_function=len,  # Measures length in standard characters
        add_start_index=True  # Includes start index metadata for referencing
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]  # Example of accessing a chunk
    print(document.page_content)
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    """Embeds text chunks and saves them to a Chroma vector database.

    Args:
        chunks: A list of Langchain Document objects representing text chunks.
    
    Returns:
        Chroma: The created Chroma database instance.
    """

    model_name = "sentence-transformers/all-mpnet-base-v2"  # Model for text embeddings
    model_kwargs = {"device": "cpu"}  # Run on CPU for broader compatibility
    encode_kwargs = {"normalize_embeddings": True}  # Ensure embeddings are normalized

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Clear existing database if it exists

    vectordb = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    vectordb.persist()  # Save the database to disk
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")
    return vectordb

def generate_data_store():
    """Orchestrates the full process of loading, splitting, and saving data to Chroma.

    Returns:
        Chroma: The generated Chroma vector database.
    """

    documents = load_documents()
    chunks = split_text(documents)
    vectordb = save_to_chroma(chunks)
    return vectordb


