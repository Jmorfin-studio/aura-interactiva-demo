# ingest_data.py (Versión corregida)

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# --- CONFIGURACIÓN ---
DATA_PATH = "aura_knowledge_base/"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


def load_documents(directory_path):
    """
    Carga todos los documentos de un directorio. DirectoryLoader seleccionará
    automáticamente el cargador correcto para los tipos de archivo conocidos.
    """
    print(f"Cargando documentos desde: {directory_path}")
    
    loader = DirectoryLoader(
        directory_path,
        glob="**/*",  # Esto asegura que busque en subdirectorios también
        show_progress=True,
        use_multithreading=True,
        silent_errors=True # Ignora archivos que no puede leer
    )
    
    documents = loader.load()
    print(f"Se cargaron {len(documents)} documentos.")
    return documents


def split_documents(documents):
    """
    Divide los documentos cargados en fragmentos más pequeños (chunks).
    """
    print("Dividiendo documentos en fragmentos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Se crearon {len(chunks)} fragmentos.")
    return chunks


def create_and_store_embeddings(index_name, chunks):
    """
    Crea los embeddings para cada fragmento y los almacena en el índice de Pinecone.
    """
    print(f"Conectando al índice de Pinecone: {index_name}")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("Creando embeddings y subiéndolos a Pinecone... (Esto puede tardar unos minutos)")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        index_name=index_name
    )
    print("¡Proceso completado! Los embeddings han sido almacenados en Pinecone.")


def main():
    """
    Función principal para ejecutar todo el pipeline de ingesta.
    """
    documents = load_documents(DATA_PATH)
    
    if not documents:
        print("No se encontraron documentos para procesar. Verifica que la carpeta 'aura_knowledge_base' no esté vacía. Abortando.")
        return
        
    chunks = split_documents(documents)
    
    create_and_store_embeddings(PINECONE_INDEX_NAME, chunks)


if __name__ == "__main__":
    main()