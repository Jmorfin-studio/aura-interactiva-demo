# main.py (VERSIÓN FINAL, REFACTORIZADA Y ROBUSTA)

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
origins = ["https://aurainteractiva.netlify.app", "http://localhost", "http://localhost:8000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Componentes de IA globales (se cargan una sola vez) ---
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

# --- 2. ENDPOINT DE BRIEFING ---
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
    active_sessions[session_id] = {"clientName": clientName, "clientCompany": clientCompany, "chat_history": ""}
    return {"message": f"¡Clara lista para {clientName}!", "session_id": session_id}

# --- 3. LÓGICA DE WEBSOCKET (REFACTORIZADA) ---
prompt_template_str = """
# CONSTITUCIÓN CONVERSACIONAL DE "CLARA"
# Eres Clara, una asistente digital experta y carismática de "Aura Interactiva". Tu propósito es inspirar y demostrar el "arte de lo posible". Hablas de forma concisa y natural en español. Siempre terminas tus respuestas con una pregunta para mantener la conversación viva.
# BRIEFING DE REUNIÓN ACTUAL: {briefing}
# CONOCIMIENTO RELEVANTE DE LA SESIÓN: {context}
# HISTORIAL DE CONVERSACIÓN: {history}
# PREGUNTA DEL USUARIO: {question}
# TU RESPUESTA (Como Clara):
"""
prompt = PromptTemplate.from_template(prompt_template_str)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in active_sessions:
        await websocket.close(code=4004, reason="ID de sesión no válido.")
        return
    
    print(f"Conexión WebSocket para sesión '{session_id}' aceptada. Preparando orquestador...")
    
    session_data = active_sessions[session_id]
    retriever = vectorstore.as_retriever(search_kwargs={'namespace': session_id})
    rag_chain = ({"context": itemgetter("question") | retriever, "question": itemgetter("question"), "briefing": lambda x: f"Estás en una reunión con {session_data.get('clientName', 'un cliente')} de la empresa {session_data.get('clientCompany', 'desconocida')}.", "history": itemgetter("history")} | prompt | llm | StrOutputParser())
    
    deepgram_connection = None
    try:
        options = LiveOptions(model="nova-2", language="es-MX", smart_format=True, endpointing=300, interim_results=False)
        deepgram_connection = deepgram_client.listen.asynclive.v("1")

        is_speaking = asyncio.Event()

        async def handle_ai_response(text: str):
            is_speaking.set()
            try:
                session_data["chat_history"] += f"Humano: {text}\n"
                full_response = await rag_chain.ainvoke({"question": text, "history": session_data["chat_history"]})
                print(f"Respuesta generada por IA: {full_response}")
                session_data["chat_history"] += f"Clara: {full_response}\n"
                await websocket.send_json({"type": "ai_response", "data": full_response})
            except Exception as e:
                print(f"Error durante la orquestación: {e}")
            finally:
                await websocket.send_json({"type": "response_end"})
                is_speaking.clear()

        async def on_message(self_inner, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if transcript and not is_speaking.is_set():
                asyncio.create_task(handle_ai_response(transcript))
        
        deepgram_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        await deepgram_connection.start(options)
        print("Orquestador listo y escuchando.")
        
        while True:
            data = await websocket.receive_bytes()
            await deepgram_connection.send(data)

    except WebSocketDisconnect:
        print(f"Cliente desconectado de sesión {session_id}")
    except Exception as e:
        print(f"Error crítico en la conexión WebSocket: {e}")
    finally:
        if deepgram_connection:
            await deepgram_connection.finish()
        if session_id in active_sessions:
            del active_sessions[session_id]
            print(f"Sesión {session_id} limpiada.")


# --- Rutas de Archivos Estáticos ---
@app.get("/briefing", response_class=FileResponse)
async def get_briefing_page(): return "briefing.html"
@app.get("/index.html", response_class=FileResponse)
async def get_index_explicitly(): return "index.html"
@app.get("/", response_class=FileResponse)
async def get_index_page(): return "index.html"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)