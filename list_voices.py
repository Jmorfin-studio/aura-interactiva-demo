# list_voices.py
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

# Carga tu clave de API
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not ELEVENLABS_API_KEY:
    print("Error: No se encontró la ELEVENLABS_API_KEY en el archivo .env")
    exit()

print("Conectando a ElevenLabs para obtener tu lista de voces personal...")

try:
    # Inicializa el cliente
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    
    # Pide la lista de todas las voces disponibles para tu cuenta
    voices = client.voices.get_all()
    
    print("\n--- ¡VOCES DISPONIBLES EN TU CUENTA! ---")
    found_spanish_voice = False
    for voice in voices.voices:
        # Buscamos voces que hablen español
        if voice.category == 'premade' and 'es' in [lang.language_id for lang in voice.labels.get('supported_languages', [])]:
            print(f"  --> VOZ EN ESPAÑOL ENCONTRADA:")
            print(f"      Nombre: {voice.name}")
            print(f"      ID: {voice.voice_id} <--- ¡COPIA ESTE ID!")
            found_spanish_voice = True
    
    if not found_spanish_voice:
        print("\nNo se encontraron voces predefinidas en español. Aquí están todas las voces:")
        for voice in voices.voices:
             print(f"  - Nombre: {voice.name}, ID: {voice.voice_id}")

    print("\nCopia el ID de una de las voces (idealmente una en español) y pégalo en main.py.")

except Exception as e:
    print(f"\nOcurrió un error al conectar con ElevenLabs: {e}")
    print("Verifica que tu API Key de ElevenLabs en el archivo .env sea correcta y tenga permisos.")