# main.py (VERSIÓN FINAL CON TIMEOUTS DE IA)

import os, asyncio, uuid
from dotenv import load_dotenv
# ... (resto de imports sin cambios) ...
from pinecone import Pinecone

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
# ... (código sin cambios) ...

# --- 2. ENDPOINT DE BRIEFING (Sin cambios) ---
# ... (código sin cambios) ...

# --- 3. LÓGICA DE WEBSOCKET (CON TIMEOUTS) ---
prompt_template_str = """...""" # Sin cambios
prompt = PromptTemplate.from_template(prompt_template_str)

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in active_sessions:
        await websocket.close(code=4004, reason="ID de sesión no válido.")
        return

    print(f"Conexión WebSocket para sesión '{session_id}' aceptada y estable.")
    session_data = active_sessions[session_id]
    retriever = vectorstore.as_retriever(search_kwargs={'namespace': session_id})
    rag_chain = ({"context": itemgetter("question") | retriever, "question": itemgetter("question"), "briefing": lambda x: f"Reunión con {session_data.get('clientName')}", "history": itemgetter("history")} | prompt | llm | StrOutputParser())

    try:
        while True:
            user_text = await websocket.receive_text()
            print(f"Texto recibido del cliente: '{user_text}'")
            session_data["chat_history"] += f"Humano: {user_text}\n"

            # --- INICIO DE LA CORRECCIÓN CON TIMEOUT ---
            try:
                # Ponemos un "vigilante" de 30 segundos a la llamada de la IA.
                print("Invocando la cadena RAG con un timeout de 30 segundos...")
                full_response = await asyncio.wait_for(
                    rag_chain.ainvoke({"question": user_text, "history": session_data["chat_history"]}),
                    timeout=30.0
                )
                print(f"Respuesta generada por IA: '{full_response}'")
                session_data["chat_history"] += f"Clara: {full_response}\n"
                
                await websocket.send_json({"type": "ai_response", "data": full_response})

            except asyncio.TimeoutError:
                print("!!!!!!!!!! TIMEOUT: La cadena RAG tardó más de 30 segundos en responder. !!!!!!!!!!!")
                error_message = "Lo siento, estoy tardando más de lo normal en pensar. ¿Podrías repetirme la pregunta?"
                await websocket.send_json({"type": "ai_response", "data": error_message})
            
            except Exception as e:
                print(f"!!!!!!!!!! ERROR EN LA CADENA RAG: {e} !!!!!!!!!!!")
                error_message = "Tuve un problema al procesar tu solicitud. Intentémoslo de nuevo."
                await websocket.send_json({"type": "ai_response", "data": error_message})
            
            finally:
                await websocket.send_json({"type": "response_end"})
                print("Ciclo de conversación completado.")
            # --- FIN DE LA CORRECCIÓN CON TIMEOUT ---

    except WebSocketDisconnect:
        print(f"Cliente desconectado de sesión {session_id}")
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]
            print(f"Sesión {session_id} limpiada.")

# --- Rutas de Archivos Estáticos (Sin cambios) ---
# ... (código sin cambios) ...