# main.py (VERSIÓN DE DEPURACIÓN QUIRÚRGICA DEFINITIVA)

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

# --- 3. LÓGICA DE WEBSOCKET (CON DEPURACIÓN) ---
prompt_template_str = """...""" # No es necesario cambiarlo
prompt = PromptTemplate.from_template(prompt_template_str)

class ConnectionManager:
    def __init__(self, session_id: str, websocket: WebSocket):
        self.websocket = websocket; self.session_id = session_id; self.session_data = active_sessions.get(session_id, {}); self.retriever = vectorstore.as_retriever(search_kwargs={'namespace': self.session_id}); self.llm_chain = self._create_rag_chain(); self.is_speaking = asyncio.Event()
    
    def _create_rag_chain(self): return ({"context": itemgetter("question") | self.retriever, "question": itemgetter("question"), "briefing": lambda x: f"Estás en una reunión con {self.session_data.get('clientName', 'un cliente')} de la empresa {self.session_data.get('clientCompany', 'desconocida')}.", "history": itemgetter("history")} | prompt | llm | StrOutputParser())
    
    async def _handle_ai_response(self, text: str):
        # ... (código sin cambios) ...
        pass

    async def run(self):
        print("DEBUG: Entrando en ConnectionManager.run()")
        deepgram_connection = None
        try:
            print("DEBUG: PASO 1 - Creando opciones de Deepgram...")
            options = LiveOptions(model="nova-2", language="es-MX", smart_format=True, endpointing=300, interim_results=False)
            
            print("DEBUG: PASO 2 - Creando objeto de conexión de Deepgram...")
            deepgram_connection = deepgram_client.listen.asynclive.v("1")

            async def on_message(self_inner, result, **kwargs):
                # ... (código sin cambios) ...
                pass
            
            print("DEBUG: PASO 3 - Registrando callback 'on_message'...")
            deepgram_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            
            print("DEBUG: PASO 4 - Intentando iniciar la conexión con Deepgram (await deepgram.start)...")
            await deepgram_connection.start(options)
            print("!!!!!!!!!! SI VES ESTE MENSAJE, DEEPGRAM SE CONECTÓ CON ÉXITO !!!!!!!!!!")

            print("DEBUG: PASO 5 - Entrando al bucle while para recibir audio.")
            while True: 
                data = await self.websocket.receive_bytes()
                await deepgram_connection.send(data)
                
        except Exception as e:
            # Hacemos el error extremadamente visible
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!! ERROR CRÍTICO CAPTURADO EN run() !!!!!!!!!!!")
            print(f"TIPO DE ERROR: {type(e)}")
            print(f"MENSAJE DE ERROR: {e}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        finally:
            print("DEBUG: Entrando al bloque finally de run() para cerrar la conexión.")
            # ... (código sin cambios) ...

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in active_sessions:
        await websocket.close(code=4004, reason="ID de sesión no válido.")
        return
    manager = ConnectionManager(session_id, websocket)
    await manager.run()

# ... (resto del código sin cambios) ...
if __name__ == "__main__":
    import uvicorn; uvicorn.run("main:app", host="0.0.0.0", port=8000)