# main.py (VERSIÓN DE DIAGNÓSTICO PING-PONG)
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Aura Interactiva - Test de Conexión")

# --- Configuración de CORS ---
origins = ["https://aurainteractiva.netlify.app", "http://localhost", "http://localhost:8000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/educar-sesion")
async def fake_educar_sesion():
    print("Recibida petición de briefing falsa. Respondiendo con éxito.")
    return {"message": "¡Prueba de conexión lista!", "session_id": "ping_pong_session"}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"[{session_id}] Conexión de prueba PING-PONG aceptada y estable.")
    try:
        while True:
            # Esperamos a recibir cualquier texto del cliente
            data = await websocket.receive_text()
            print(f"[{session_id}] PING recibido del cliente: '{data}'")
            
            # Respondemos con un "PONG"
            response = f"PONG: Recibí tu mensaje '{data}' con éxito."
            await websocket.send_text(response)
            print(f"[{session_id}] PONG enviado al cliente.")
    except WebSocketDisconnect:
        print(f"[{session_id}] Cliente desconectado de PING-PONG.")
    except Exception as e:
        print(f"[{session_id}] Error en PING-PONG: {e}")

# --- Rutas para servir los archivos HTML ---
@app.get("/briefing", response_class=FileResponse)
async def get_briefing_page(): return "briefing.html"
@app.get("/index.html", response_class=FileResponse)
async def get_index_explicitly(): return "index.html"
@app.get("/", response_class=FileResponse)
async def get_index_page(): return "index.html"