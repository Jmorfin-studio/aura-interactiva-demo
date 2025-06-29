# main.py (Versión 9.0 - El Intento Definitivo con D-ID)

import os, asyncio, uuid, base64, httpx, certifi
from dotenv import load_dotenv
from typing import List, Dict, Any
from operator import itemgetter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
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
from elevenlabs.client import ElevenLabs

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
load_dotenv(); OPENAI_API_KEY = os.getenv("OPENAI_API_KEY"); PINECONE_API_KEY = os.getenv("PINECONE_API_KEY"); PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT"); PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME"); DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY"); ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY"); DID_API_KEY = os.getenv("DID_API_KEY")
CLARA_IMAGE_URL = "https://i.ibb.co/ynch1sP3/clara-avatar.png"
app = FastAPI(title="Aura Interactiva - Orquestador F1"); llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3); embeddings = OpenAIEmbeddings(model="text-embedding-3-small"); vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings); elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY); deepgram_client = DeepgramClient(DEEPGRAM_API_KEY); active_sessions: Dict[str, Any] = {}; app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 2. ENDPOINT DE BRIEFING ---
@app.post("/educar-sesion")
async def educar_sesion(clientName: str = Form(...), clientCompany: str = Form(...), urls: str = Form(None), files: List[UploadFile] = File(None)):
    session_id = str(uuid.uuid4()); all_docs = []; temp_dir = "temp_client_files"; os.makedirs(temp_dir, exist_ok=True)
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
        vectorstore.add_documents(documents=docs_split, namespace=session_id); print(f"¡Documentos indexados en namespace '{session_id}'!")
    active_sessions[session_id] = {"clientName": clientName, "clientCompany": clientCompany, "chat_history": ""}
    return {"message": f"¡Clara lista para {clientName}!", "session_id": session_id}

# --- 3. LÓGICA DE STREAMING Y WEBSOCKET ---
prompt_template_str = """
# CONSTITUCIÓN CONVERSACIONAL DE "CLARA"
# Eres Clara, una asistente digital experta y carismática de "Aura Interactiva". Tu propósito es inspirar y demostrar el "arte de lo posible". Hablas de forma concisa y natural. Siempre terminas tus respuestas con una pregunta para mantener la conversación viva.
# BRIEFING DE REUNIÓN ACTUAL: {briefing}
# CONOCIMIENTO RELEVANTE DE LA SESIÓN: {context}
# HISTORIAL DE CONVERSACIÓN: {history}
# PREGUNTA DEL USUARIO: {question}
# TU RESPUESTA (Como Clara):
"""
prompt = PromptTemplate.from_template(prompt_template_str)

class ConnectionManager:
    def __init__(self, session_id: str, websocket: WebSocket):
        self.websocket = websocket; self.session_id = session_id; self.session_data = active_sessions.get(session_id, {}); self.retriever = vectorstore.as_retriever(search_kwargs={'namespace': self.session_id}); self.llm_chain = self._create_rag_chain(); self.is_speaking = asyncio.Event()
    
    def _create_rag_chain(self):
        return ({"context": itemgetter("question") | self.retriever, "question": itemgetter("question"), "briefing": lambda x: f"Estás en una reunión con {self.session_data.get('clientName', 'un cliente')}.", "history": itemgetter("history")} | prompt | llm | StrOutputParser())
    
    async def _handle_ai_response(self, text: str):
        self.is_speaking.set(); ssl_context = httpx.create_ssl_context(verify=certifi.where())
        try:
            self.session_data["chat_history"] += f"Humano: {text}\n"; full_response = await self.llm_chain.ainvoke({"question": text, "history": self.session_data["chat_history"]})
            print(f"Respuesta generada por IA: {full_response}"); self.session_data["chat_history"] += f"Clara: {full_response}\n"
            
            print("Enviando solicitud a D-ID con la voz de ElevenLabs..."); DID_API_URL = "https://api.d-id.com/clips"
            auth_string = DID_API_KEY.encode('utf-8'); encoded_auth = base64.b64encode(auth_string).decode('utf-8')
            headers = {"Authorization": f"Basic {encoded_auth}", "Content-Type": "application/json"}
            
            # --- Payload FINAL: Construido según las indicaciones del soporte de D-ID ---
            payload = {
                "script": {
                    "type": "text",
                    "input": full_response,
                    "provider": {
                        "type": "elevenlabs",
                        "voice_id": "SZfY4K69FwXus87eayHK" # ID de la voz "Rachel" de ElevenLabs
                    }
                },
                "source_url": CLARA_IMAGE_URL, # Usamos la URL de tu imagen de Clara
                "config": {"result_format": "mp4"}
            }
            
            async with httpx.AsyncClient(verify=ssl_context) as client:
                response = await client.post(DID_API_URL, headers=headers, json=payload, timeout=90)
                response.raise_for_status(); clip_data = response.json(); clip_id = clip_data.get("id")
                if not clip_id: raise Exception(f"D-ID no devolvió un ID de clip. Respuesta: {clip_data}")
                
                print(f"Clip en proceso con ID: {clip_id}. Esperando resultado..."); video_url = None
                for _ in range(45): 
                    await asyncio.sleep(2); get_response = await client.get(f"{DID_API_URL}/{clip_id}", headers=headers)
                    get_response.raise_for_status(); result_data = get_response.json()
                    if result_data.get("status") == "done":
                        video_url = result_data.get("result_url"); break
                
                if not video_url: raise Exception("El clip de D-ID no se completó a tiempo.")
                print(f"Video listo en: {video_url}"); await self.websocket.send_json({"type": "video_url", "data": video_url})

        except httpx.HTTPStatusError as e:
            print(f"Error HTTP de D-ID: {e.response.status_code} - {e.response.text}"); await self.websocket.send_json({"type": "error", "message": f"Error de D-ID: {e.response.status_code}"})
        except Exception as e:
            print(f"Error durante la orquestación: {e}"); await self.websocket.send_json({"type": "error", "message": str(e)})
        finally:
            try:
                if self.websocket.client_state.name == 'CONNECTED': await self.websocket.send_json({"type": "response_end"})
            except Exception: pass
            self.is_speaking.clear(); print("Ciclo de respuesta completado.")

    async def run(self):
        deepgram_connection = None
        try:
            options = LiveOptions(model="nova-2", language="es-MX", smart_format=True, endpointing=300, interim_results=False)
            deepgram_connection = deepgram_client.listen.asynclive.v("1")
            
            async def on_message(self_inner, result, **kwargs):
                transcript = result.channel.alternatives[0].transcript
                if transcript and not self.is_speaking.is_set():
                    print(f"\nUsuario dijo: {transcript}"); asyncio.create_task(self._handle_ai_response(transcript))
            
            deepgram_connection.on(LiveTranscriptionEvents.Transcript, on_message); await deepgram_connection.start(options)
            
            while True: 
                data = await self.websocket.receive_bytes(); await deepgram_connection.send(data)
        except WebSocketDisconnect: print(f"Cliente desconectado de sesión {self.session_id}")
        except Exception as e: print(f"Error en la conexión WebSocket: {e}")
        finally:
            if deepgram_connection: await deepgram_connection.finish()

# --- Rutas y Ejecución ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept();
    if session_id not in active_sessions: await websocket.close(code=4004, reason="ID de sesión no válido"); return
    manager = ConnectionManager(session_id, websocket)
    await manager.run()
@app.get("/briefing", response_class=FileResponse)
async def get_briefing_page(): return "briefing.html"
@app.get("/", response_class=FileResponse)
async def get_index_page(): return "index.html"
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)