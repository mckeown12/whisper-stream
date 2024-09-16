from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import whisperx
import numpy as np
import logging
import asyncio
import numpy as np
from fastapi import WebSocket
import whisperx
from concurrent.futures import ProcessPoolExecutor
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)



logger = logging.getLogger(__name__)

# Initialize ProcessPoolExecutor
process_pool = ProcessPoolExecutor(max_workers=2)

# Load WhisperX model (do this once, outside the websocket handler)
model = whisperx.load_model("base", device="cpu", compute_type="float32")

def transcribe_audio(audio_data):
    try:
        result = model.transcribe(audio_data)
        text = ''
        if 'segments' in result:
            for segment in result['segments']:
                text += segment['text'] + ' '
        return text
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return ""

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    audio_buffer = []
    SAMPLE_RATE = 16000
    CHUNK_DURATION = 5  # seconds
    OVERLAP_DURATION = 1  # seconds
    MIN_SILENCE_LENGTH = 500  # milliseconds
    SILENCE_THRESHOLD = -40  # dB
    
    try:
        while True:
            # Receive audio chunk from the client
            audio_chunk = await websocket.receive_bytes()
            
            # Convert bytes to numpy array
            chunk_np = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Append to buffer
            audio_buffer.extend(chunk_np)
            
            # If we have collected enough audio, process it
            if len(audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION:
                # Create overlapping chunks
                chunk_samples = SAMPLE_RATE * CHUNK_DURATION
                overlap_samples = SAMPLE_RATE * OVERLAP_DURATION
                
                full_chunk = np.array(audio_buffer[:chunk_samples])
                audio_buffer = audio_buffer[chunk_samples - overlap_samples:]
                
                # Convert to AudioSegment for silence detection
                audio_segment = AudioSegment(
                    full_chunk.tobytes(),
                    frame_rate=SAMPLE_RATE,
                    sample_width=4,
                    channels=1
                )
                
                # Detect non-silent parts
                non_silent_ranges = detect_nonsilent(
                    audio_segment,
                    min_silence_len=MIN_SILENCE_LENGTH,
                    silence_thresh=SILENCE_THRESHOLD
                )
                
                # Process each non-silent part
                for start, end in non_silent_ranges:
                    chunk_to_process = full_chunk[start * SAMPLE_RATE // 1000 : end * SAMPLE_RATE // 1000]
                    
                    # Use ProcessPoolExecutor to run transcription in a separate process
                    future = process_pool.submit(transcribe_audio, chunk_to_process)
                    
                    # Asynchronously wait for the result
                    text = await asyncio.get_event_loop().run_in_executor(None, future.result)
                    
                    if text:
                        await websocket.send_text(text)
                        logger.info(f"Sent transcription: {text}")
                
    except Exception as e:
        logger.error(f"Error during websocket communication: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)