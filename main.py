# main.py (VERSIÓN DE DEPURACIÓN QUIRÚRGICA)

import os, asyncio, uuid
from dotenv import load_dotenv
from typing import List, Dict, Any
from operator import itemgetter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from pinecone.exceptions import NotFoundException

from pinecone import Pinecone

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

app = FastAPI(title="Aura Interactiva - Orquestador")

# --- Configuración de CORS ---
origins = [ "https://aurainteractiva.netlify.app", "http://localhost", "http://localhost:8000" ]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)
active_sessions: Dict[str, Any] = {}

print("Inicializando cliente de Pinecone de forma explícita...")
pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
print(f"Accediendo al índice: {PINECONE_INDEX_NAME}")
index = pinecone_client.Index(PINECONE_INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
print("¡Conexión con Pinecone y VectorStore establecidos con éxito!")

# --- 2. ENDPOINT DE BRIEFING (Sin cambios) ---
@app.post("/educar-sesion")
async def educar_sesion(clientName: str = Form(...), clientCompany: str = Form(...), urls: str = Form(None), files: List[UploadFile] = File(None)):
    session_id = str(uuid.uuid4()); print(f"Creando nueva sesión con ID: '{session_id}'")
    all_docs = []; temp_dir = "temp_client_files"; os.makedirs(temp_dir, exist_ok=True)
    if files:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename);
            with open(file_path, "wb") as buffer: buffer.write(await file.read())
            loader_map = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader}; ext = os.path.splitext(file.filename)[1].lower()
            if ext in loader_map: all_docs.extend(loader_map[ext](file_path).load())
    if urls:
        url_list = [url.strip() for url in urls.split('\n') if url.strip()];
        if url_list: all_docs.extend(WebBaseLoader(url_list).load())
    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200); docs_split = text_splitter.split_documents(all_docs)
        vectorstore.delete(delete_all=True, namespace=session_id); print(f"Namespace '{session_id}' limpiado.")
        vectorstore.add_documents(documents=docs_split, namespace=session_id); print(f"¡Documentos indexados en namespace '{session_id}'!")
    active_sessions[session_id] = {"clientName": clientName, "clientCompany": clientCompany, "chat_history": ""}; return {"message": f"¡Clara lista para {clientName}!", "session_id": session_id}

# --- 3. LÓGICA DE WEBSOCKET (CON DEPURACIÓN EN __INIT__) ---
prompt_template_str = """
# CONSTITUCIÓN CONVERSACIONAL DE "CLARA"
# ... (prompt sin cambios) ...
# TU RESPUESTA (Como Clara):
"""
prompt = PromptTemplate.from_template(prompt_template_str)

class ConnectionManager:
    def __init__(self, session_id: str, websocket: WebSocket):
        print("DEBUG: Entrando en ConnectionManager.__init__")
        self.websocket = websocket
        self.session_id = session_id
        print(f"DEBUG: Session ID '{self.session_id}' asignado.")
        
        self.session_data = active_sessions.get(session_id, {})
        print("DEBUG: Datos de la sesión obtenidos.")
        
        self.retriever = vectorstore.as_retriever(search_kwargs={'namespace': self.session_id})
        print("DEBUG: Retriever de Pinecone creado con éxito.")
        
        self.llm_chain = self._create_rag_chain()
        print("DEBUG: Cadena RAG (llm_chain) creada con éxito.")
        
        self.is_speaking = asyncio.Event()
        print("DEBUG: Evento de habla (is_speaking) inicializado.")
        print("DEBUG: __init__ de ConnectionManager completado con éxito.")
    
    def _create_rag_chain(self): return ({"context": itemgetter("question") | self.retriever, "question": itemgetter("question"), "briefing": lambda x: f"Estás en una reunión con {self.session_data.get('clientName', 'un cliente')} de la empresa {self.session_data.get('clientCompany', 'desconocida')}.", "history": itemgetter("history")} | prompt | llm | StrOutputParser())
    
    async def _handle_ai_response(self, text: str):
        # ... (código sin cambios) ...
        self.is_speaking.set(); print("DEBUG: Evento is_speaking activado.")
        try:
            print(f"DEBUG: Enviando texto a LLM: '{text}'")
            # ... resto del try ...
        except Exception as e: print(f"!!!!!!!!!! ERROR EN _handle_ai_response !!!!!!!!!!!\n{e}")
        finally: await self.websocket.send_json({"type": "response_end"}); self.is_speaking.clear(); print("DEBUG: Ciclo de respuesta completado.")

    async def run(self):
        print("DEBUG: Entrando en ConnectionManager.run()")
        deepgram_connection = None
        try:
            options = LiveOptions(model="nova-2", language="es-MX", smart_format=True, endpointing=300, interim_results=False)
            print("DEBUG: Opciones de Deepgram creadas.")
            # ... (resto del código sin cambios) ...
        except Exception as e:
            print(f"!!!!!!!!!! ERROR CRÍTICO CAPTURADO EN run() !!!!!!!!!!!\n{e}")
        finally:
            # ... (código sin cambios) ...
            print("DEBUG: Entrando al bloque finally de run().")

# --- Rutas y Ejecución (Sin cambios) ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    print(f"DEBUG: Aceptando conexión WebSocket para sesión {session_id}")
    await websocket.accept()
    if session_id not in active_sessions:
        print(f"ERROR: ID de sesión {session_id} no válido. Cerrando conexión.")
        await websocket.close(code=4004, reason="ID de sesión no válido.")
        return
    try:
        manager = ConnectionManager(session_id, websocket)
        print("DEBUG: Objeto ConnectionManager creado. Llamando a manager.run()")
        await manager.run()
    except Exception as e:
        print(f"!!!!!!!!!! ERROR CRÍTICO AL CREAR O EJECUTAR EL MANAGER !!!!!!!!!!!\n{e}")

# ... (resto de las rutas sin cambios) ...
if __name__ == "__main__":
    import uvicorn; uvicorn.run("main:app", host="0.0.0.0", port=8000)