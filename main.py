# main.py (VERSIÓN DE PRUEBA MINIMALISTA PARA VERIFICAR LÍMITES DE RECURSOS)
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Aura Interactiva - Test de Conexión")

# --- Configuración de CORS ---
origins = ["https://aurainteractiva.netlify.app", "http://localhost", "http://localhost:8000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Endpoint de Briefing Falso (para que el flujo no se rompa) ---
@app.post("/educar-sesion")
async def fake_educar_sesion():
    print("Recibida petición de briefing falsa. Respondiendo con éxito.")
    # Devolvemos un ID de sesión falso para que el frontend pueda continuar.
    return {"message": "¡Prueba de conexión lista!", "session_id": "test_session_123"}

# --- WebSocket Minimalista ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"!!! ÉXITO !!! Conexión WebSocket para sesión '{session_id}' aceptada y ESTABLE.")
    print("El servidor ahora esperará en un bucle. La conexión debe mantenerse abierta.")
    try:
        while True:
            # Esperamos indefinidamente. En una app real, aquí recibiríamos datos.
            await asyncio.sleep(60) # Dormimos para no consumir CPU
            print(f"[{session_id}] Conexión sigue viva...")

    except WebSocketDisconnect:
        print(f"Cliente desconectado de la sesión de prueba '{session_id}'.")
    except Exception as e:
        print(f"Error inesperado en el WebSocket de prueba: {e}")
    finally:
        print(f"Cerrando conexión de prueba para la sesión '{session_id}'.")


# --- Rutas para servir los archivos HTML ---
@app.get("/briefing", response_class=FileResponse)
async def get_briefing_page(): return "briefing.html"
@app.get("/index.html", response_class=FileResponse)
async def get_index_explicitly(): return "index.html"
@app.get("/", response_class=FileResponse)
async def get_index_page(): return "index.html"