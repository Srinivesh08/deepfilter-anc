import queue
import threading
import base64
import numpy as np
import torch
import sounddevice as sd
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from df.enhance import init_df, enhance
from numba import njit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize DeepFilterNet
print("Loading DeepFilterNet3 model...")
model, df_state, _ = init_df()
SR = df_state.sr()

# Audio config
CAPTURE_BLOCK = 480       # 10ms capture blocks (low-latency mic capture)
PROCESS_SIZE  = SR * 3    # Process 3 seconds at a time (144000 samples) for max clarity
CHANNELS = 2              # 0 = Left (DFN), 1 = Right (NLMS Reference)

# State
audio_active = False
stream = None
audio_queue = queue.Queue()
process_thread = None

# NLMS State configuration
NLMS_FILTER_LENGTH = 256
nlms_state = {
    "enabled": False,
    "mu": 0.05,
    "w": np.zeros(NLMS_FILTER_LENGTH, dtype=np.float32),
    "ref_history": np.zeros(NLMS_FILTER_LENGTH - 1, dtype=np.float32)
}

@njit
def nlms_process(primary, reference_padded, w, mu, epsilon=1e-4):
    """
    primary: DFN output, shape (N,)
    reference_padded: ref_history + reference_chunk, shape (N + filter_length - 1,)
    w: filter weights, shape (filter_length,)
    Returns: enhanced (N,), updated_w (filter_length,)
    """
    N = len(primary)
    filter_length = len(w)
    enhanced = np.zeros(N, dtype=np.float32)
    
    for n in range(N):
        # The reference slice for the current sample
        # At n=0, slice is 0 to filter_length. We reverse it so x[0] is newest.
        x = reference_padded[n : n + filter_length][::-1]
        
        # Estimate noise
        est_noise = np.dot(w, x)
        
        # Residual (enhanced speech)
        e = primary[n] - est_noise
        enhanced[n] = e
        
        # Update weights
        norm = np.dot(x, x)
        w = w + (mu / (epsilon + norm)) * e * x
        
    return enhanced, w

def capture_callback(indata, frames, time_info, status):
    """Just captures stereo mic audio into a queue — fast, no processing here."""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def processing_loop():
    """Separate thread: accumulates audio, processes with DeepFilterNet and NLMS, streams to clients."""
    buffer = np.zeros((0, CHANNELS), dtype=np.float32)

    while audio_active:
        try:
            chunk = audio_queue.get(timeout=0.1)
            buffer = np.concatenate([buffer, chunk])

            if len(buffer) >= PROCESS_SIZE:
                to_process = buffer[:PROCESS_SIZE]
                buffer = buffer[PROCESS_SIZE:]

                # Split channels
                primary_chunk = to_process[:, 0]
                reference_chunk = to_process[:, 1]

                # 1. DeepFilterNet on Primary (Left)
                audio_tensor = torch.from_numpy(primary_chunk).float().unsqueeze(0)
                enhanced_dfn = enhance(model, df_state, audio_tensor)
                enhanced_dfn_np = enhanced_dfn.squeeze(0).cpu().numpy()
                
                # Metrics pre-NLMS
                input_rms = float(np.sqrt(np.mean(primary_chunk ** 2)))
                output_rms = float(np.sqrt(np.mean(enhanced_dfn_np ** 2)))
                
                # 2. NLMS (if enabled)
                if nlms_state["enabled"]:
                    w = nlms_state["w"]
                    ref_history = nlms_state["ref_history"]
                    mu = nlms_state["mu"]
                    
                    # Pad reference with history from previous chunk
                    ref_padded = np.concatenate([ref_history, reference_chunk])
                    
                    # Process via Numba JIT function
                    enhanced_final_np, new_w = nlms_process(enhanced_dfn_np, ref_padded, w, mu)
                    
                    # Update state
                    nlms_state["w"] = new_w
                    nlms_state["ref_history"] = reference_chunk[-(NLMS_FILTER_LENGTH - 1):]
                else:
                    enhanced_final_np = enhanced_dfn_np

                # Metrics post-NLMS
                final_rms = float(np.sqrt(np.mean(enhanced_final_np ** 2)))
                nr = 0.0
                if input_rms > 0.001:
                    nr = max(0.0, min(100.0, (1.0 - final_rms / input_rms) * 100.0))

                # Downsample waveform for visualization
                step = max(1, len(primary_chunk) // 128)

                # Encode enhanced audio as base64 float32
                audio_b64 = base64.b64encode(
                    enhanced_final_np.astype(np.float32).tobytes()
                ).decode("ascii")

                socketio.emit("audio_metrics", {
                    "input_level": min(1.0, input_rms * 10),
                    "output_level": min(1.0, final_rms * 10),
                    "noise_reduction": round(nr, 1),
                    "input_waveform": primary_chunk[::step].tolist()[:128],
                    "output_waveform": enhanced_final_np[::step].tolist()[:128],
                    "audio_b64": audio_b64,
                    "frames": len(enhanced_final_np),
                    "nlms_active": nlms_state["enabled"]
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

            # Reset NLMS state on start
            nlms_state["w"] = np.zeros(NLMS_FILTER_LENGTH, dtype=np.float32)
            nlms_state["ref_history"] = np.zeros(NLMS_FILTER_LENGTH - 1, dtype=np.float32)

            # Start processing thread
            process_thread = threading.Thread(target=processing_loop, daemon=True)
            process_thread.start()

            # Start mic capture (Stereo: channels=2)
            stream = sd.InputStream(
                samplerate=SR, blocksize=CAPTURE_BLOCK, channels=CHANNELS,
                callback=capture_callback
            )
            stream.start()

            socketio.emit("status", {
                "active": True, "sr": SR,
                "block_size": PROCESS_SIZE,
                "process_ms": int(PROCESS_SIZE / SR * 1000),
                "nlms": nlms_state["enabled"],
                "mu": nlms_state["mu"]
            })
            print("Audio stream started (Stereo).")
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

@socketio.on("set_nlms")
def handle_set_nlms(data):
    if "enabled" in data:
        nlms_state["enabled"] = bool(data["enabled"])
        if not nlms_state["enabled"]:
             # Reset filter weights when turned off
             nlms_state["w"] = np.zeros(NLMS_FILTER_LENGTH, dtype=np.float32)
    if "mu" in data:
        nlms_state["mu"] = float(data["mu"])
    
    socketio.emit("status", {
        "active": audio_active, "sr": SR,
        "block_size": PROCESS_SIZE,
        "process_ms": int(PROCESS_SIZE / SR * 1000),
        "nlms": nlms_state["enabled"],
        "mu": nlms_state["mu"]
    })

@socketio.on("connect")
def handle_connect():
    socketio.emit("status", {
        "active": audio_active, "sr": SR,
        "block_size": PROCESS_SIZE,
        "process_ms": int(PROCESS_SIZE / SR * 1000),
        "nlms": nlms_state["enabled"],
        "mu": nlms_state["mu"]
    })
    print("Client connected.")

if __name__ == "__main__":
    print(f"DeepFilterNet3 + NLMS loaded | SR: {SR} Hz | Process chunk: {PROCESS_SIZE} samples ({PROCESS_SIZE/SR*1000:.0f}ms)")
    print(f"Dashboard: http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
