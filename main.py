# main.py (VERSIÓN FINAL, ARQUITECTURA DE PRODUCCIÓN ROBUSTA)

import os, asyncio, uuid
from dotenv import load_dotenv
from typing import List, Dict, Any
from operator import itemgetter
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# ... (resto de imports sin cambios) ...
from pinecone import Pinecone

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
# ... (código sin cambios) ...

# --- COMPONENTES GLOBALES (CARGA PESADA AL INICIO) ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
# ... (resto sin cambios) ...

# --- 2. ENDPOINT DE BRIEFING (Sin cambios) ---
# ... (código sin cambios) ...

# --- 3. LÓGICA DE WEBSOCKET (ARQUITECTURA DE COLA DE TAREAS) ---
prompt_template_str = """...""" # Sin cambios
prompt = PromptTemplate.from_template(prompt_template_str)

# Creamos una cola para procesar las peticiones de la IA en segundo plano
# Esto evita que el WebSocket se bloquee.
request_queue = asyncio.Queue()
response_queues = {}

async def ai_processing_worker():
    print("Trabajador de IA iniciado, esperando tareas...")
    while True:
        websocket, session_id, user_text = await request_queue.get()
        
        try:
            print(f"[{session_id}] Procesando: '{user_text}'")
            session_data = active_sessions[session_id]
            retriever = vectorstore.as_retriever(search_kwargs={'namespace': session_id})
            rag_chain = ({"context": itemgetter("question") | retriever, "question": itemgetter("question"), "briefing": lambda x: f"Reunión con {session_data.get('clientName')}", "history": itemgetter("history")} | prompt | llm | StrOutputParser())
            
            session_data["chat_history"] += f"Humano: {user_text}\n"
            full_response = await rag_chain.ainvoke({"question": user_text, "history": session_data["chat_history"]})
            session_data["chat_history"] += f"Clara: {full_response}\n"
            
            print(f"[{session_id}] Respuesta generada: '{full_response}'")
            if session_id in response_queues:
                await response_queues[session_id].put(full_response)
        except Exception as e:
            print(f"!!!!!!!!!! ERROR EN EL TRABAJADOR DE IA !!!!!!!!!!!\n{e}")
            if session_id in response_queues:
                await response_queues[session_id].put(f"Error al procesar: {e}")
        finally:
            request_queue.task_done()

@app.on_event("startup")
async def startup_event():
    # Iniciamos el trabajador de IA cuando la aplicación arranca
    asyncio.create_task(ai_processing_worker())

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in active_sessions:
        await websocket.close(code=4004, reason="ID de sesión no válido.")
        return
    
    response_queue = asyncio.Queue()
    response_queues[session_id] = response_queue

    print(f"Conexión WebSocket para sesión '{session_id}' aceptada y estable.")
    
    async def receive_from_client():
        try:
            while True:
                user_text = await websocket.receive_text()
                await request_queue.put((websocket, session_id, user_text))
        except WebSocketDisconnect:
            pass

    async def send_to_client():
        try:
            while True:
                response_text = await response_queue.get()
                await websocket.send_json({"type": "ai_response", "data": response_text})
                await websocket.send_json({"type": "response_end"})
        except asyncio.CancelledError:
            pass

    receive_task = asyncio.create_task(receive_from_client())
    send_task = asyncio.create_task(send_to_client())

    try:
        await asyncio.gather(receive_task, send_task)
    except WebSocketDisconnect:
        print(f"Cliente desconectado de sesión {session_id}")
    finally:
        receive_task.cancel()
        send_task.cancel()
        del response_queues[session_id]
        if session_id in active_sessions:
            del active_sessions[session_id]
            print(f"Sesión {session_id} y recursos limpiados.")

# --- Rutas de Archivos Estáticos (Sin cambios) ---
# ... (código sin cambios) ...