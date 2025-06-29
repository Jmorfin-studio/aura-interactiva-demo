# main_test.py (Prueba de Eco - Versión Final Corregida)
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from deepgram import Deepgram

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

app = FastAPI(title="Prueba de Tubería de Audio")
deepgram_client = Deepgram(DEEPGRAM_API_KEY)

@app.websocket("/ws_test")
async def websocket_test_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Cliente de prueba conectado. Esperando audio...")
    
    try:
        # Forma correcta de iniciar la conexión para deepgram-sdk==2.12.0
        deepgram_connection = deepgram_client.listen.live.v("1", {
            "model": "nova-2",
            "language": "es-MX",
            "smart_format": True,
            "encoding": "linear16", # Es una buena práctica especificarlo
            "sample_rate": 16000     # Y también
        })

        async def on_message(self_inner, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if transcript:
                print(f"Deepgram transcribió: '{transcript}'")
                # DEVOLVEMOS EL TEXTO AL NAVEGADOR
                await websocket.send_text(f"Escuché que dijiste: {transcript}")

        deepgram_connection.on('Transcript', on_message)

        # El bucle principal que recibe audio del navegador y lo envía a Deepgram
        while True:
            data = await websocket.receive_bytes()
            deepgram_connection.send(data)

    except WebSocketDisconnect:
        print("Cliente de prueba desconectado.")
        if 'deepgram_connection' in locals() and deepgram_connection:
            await deepgram_connection.finish()
    except Exception as e:
        print(f"Error en WebSocket de prueba: {e}")
        if 'deepgram_connection' in locals() and deepgram_connection:
            await deepgram_connection.finish()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_test:app", host="0.0.0.0", port=8000)