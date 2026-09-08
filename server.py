import queue
import threading
import base64
import numpy as np
import torch
import sounddevice as sd
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from df.enhance import init_df, enhance

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize DeepFilterNet
print("Loading DeepFilterNet3 model...")
model, df_state, _ = init_df()
SR = df_state.sr()

# Audio config
CAPTURE_BLOCK = 480       # 10ms capture blocks (low-latency mic capture)
PROCESS_SIZE  = SR * 3    # Process 3 seconds at a time (144000 samples) for max clarity
CHANNELS = 1              # Mono — single mic input
TARGET_RMS = 0.08         # Target RMS level for volume normalization
MAX_GAIN = 10.0           # Max gain multiplier to prevent amplifying silence/noise

# State
audio_active = False
stream = None
audio_queue = queue.Queue()
process_thread = None

def capture_callback(indata, frames, time_info, status):
    """Captures mono mic audio into a queue — fast, no processing here."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def processing_loop():
    """Separate thread: accumulates audio, processes with DeepFilterNet, streams to clients."""
    buffer = np.zeros((0, CHANNELS), dtype=np.float32)

    while audio_active:
        try:
            chunk = audio_queue.get(timeout=0.1)
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) >= PROCESS_SIZE:
                to_process = buffer[:PROCESS_SIZE]
                buffer = buffer[PROCESS_SIZE:]

                # Extract mono channel
                primary_chunk = to_process[:, 0]

                # DeepFilterNet enhancement
                audio_tensor = torch.from_numpy(primary_chunk).float().unsqueeze(0)
                enhanced = enhance(model, df_state, audio_tensor)
                enhanced_np = enhanced.squeeze(0).cpu().numpy()

                # Metrics (computed before normalization for accurate NR%)
                input_rms = float(np.sqrt(np.mean(primary_chunk ** 2)))
                output_rms = float(np.sqrt(np.mean(enhanced_np ** 2)))
                nr = 0.0
                if input_rms > 0.001:
                    nr = max(0.0, min(100.0, (1.0 - output_rms / input_rms) * 100.0))

                # Volume normalization — consistent output loudness
                if output_rms > 1e-6:
                    gain = min(TARGET_RMS / output_rms, MAX_GAIN)
                    enhanced_np = enhanced_np * gain
                    enhanced_np = np.clip(enhanced_np, -1.0, 1.0)

                # Downsample waveform for visualization
                step = max(1, len(primary_chunk) // 128)

                # Encode enhanced audio as base64 float32
                audio_b64 = base64.b64encode(
                    enhanced_np.astype(np.float32).tobytes()
                ).decode("ascii")

                socketio.emit("audio_metrics", {
                    "input_level": min(1.0, input_rms * 10),
                    "output_level": min(1.0, output_rms * 10),
                    "noise_reduction": round(nr, 1),
                    "input_waveform": primary_chunk[::step].tolist()[:128],
                    "output_waveform": enhanced_np[::step].tolist()[:128],
                    "audio_b64": audio_b64,
                    "frames": len(enhanced_np)
                })

        except queue.Empty:
            continue

    # Drain queue on stop
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@socketio.on("start_audio")
def handle_start():
    global stream, audio_active, process_thread
    if not audio_active:
        try:
            audio_active = True

            # Start processing thread
            process_thread = threading.Thread(target=processing_loop, daemon=True)
            process_thread.start()

            # Start mic capture (Mono)
            stream = sd.InputStream(
                samplerate=SR, blocksize=CAPTURE_BLOCK, channels=CHANNELS,
                callback=capture_callback
            )
            stream.start()

            socketio.emit("status", {
                "active": True, "sr": SR,
                "block_size": PROCESS_SIZE,
                "process_ms": int(PROCESS_SIZE / SR * 1000)
            })
            print("Audio stream started (Mono).")
        except Exception as e:
            audio_active = False
            socketio.emit("error", {"message": str(e)})
            print(f"Error starting stream: {e}")

@socketio.on("stop_audio")
def handle_stop():
    global stream, audio_active, process_thread
    if audio_active:
        audio_active = False
        if stream is not None:
            stream.stop()
            stream.close()
            stream = None
        if process_thread is not None:
            process_thread.join(timeout=2)
            process_thread = None
        socketio.emit("status", {"active": False})
        print("Audio stream stopped.")

@socketio.on("connect")
def handle_connect():
    socketio.emit("status", {
        "active": audio_active, "sr": SR,
        "block_size": PROCESS_SIZE,
        "process_ms": int(PROCESS_SIZE / SR * 1000)
    })
    print("Client connected.")

if __name__ == "__main__":
    print(f"DeepFilterNet3 loaded | SR: {SR} Hz | Process chunk: {PROCESS_SIZE} samples ({PROCESS_SIZE/SR*1000:.0f}ms)")
    print(f"Dashboard: http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
