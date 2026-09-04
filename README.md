# 🎙️ DeepFilter ANC — Real-Time Neural Noise Cancellation

A real-time noise cancellation system powered by [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet), featuring a web-based dashboard with live audio visualization and **remote listening** — stream enhanced audio to your phone over WiFi.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)
![DeepFilterNet](https://img.shields.io/badge/DeepFilterNet-3-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 What It Does

1. **Captures** audio from your laptop's microphone in real-time
2. **Processes** it through DeepFilterNet3 (a deep neural network for speech enhancement)
3. **Streams** the enhanced (noise-cancelled) audio to any device on your local network
4. **Visualizes** input/output waveforms, noise reduction %, and audio levels on a beautiful web dashboard

### Use Case

Sit your laptop in a noisy room, open the dashboard on your phone, and listen to clean, noise-free audio from across the room — or even from another room on the same WiFi.

---

## 📐 Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Microphone  │────▶│  Python Server   │────▶│  Web Dashboard    │
│  (Laptop)    │     │  (DeepFilterNet) │     │  (Phone/Browser)  │
└─────────────┘     └──────────────────┘     └───────────────────┘
                           │                         │
                    Captures 10ms        Ring buffer + Web Audio
                    blocks, processes    API for gapless playback
                    3s chunks for
                    max quality
```

**Server Pipeline:**
- `sounddevice` captures mic audio in small 10ms blocks (low-latency capture)
- Audio accumulates in a thread-safe queue
- A processing thread feeds 3-second chunks to DeepFilterNet3
- Enhanced audio + metrics are streamed to all connected clients via Socket.IO

**Client Pipeline:**
- Receives enhanced audio as base64-encoded Float32 data
- Feeds it into a **ring buffer** (30-second capacity)
- A `ScriptProcessorNode` continuously reads from the ring buffer for **gapless playback**
- Pre-buffers 3.5 seconds before starting to prevent choppy audio

---

## 🛠️ Current Implementation

### ✅ DeepFilterNet3 (Active)

The current noise suppression uses **DeepFilterNet3**, a state-of-the-art deep learning model for real-time speech enhancement. It processes audio at 48kHz and effectively removes:

- Background chatter
- Fan / AC noise
- Traffic noise
- Keyboard typing
- General ambient noise

> **Note:** Processing is done in 3-second chunks for maximum clarity. This adds ~6.5 seconds of total latency (3s processing + 3.5s buffer) but produces near-offline quality.

### 🔮 Future Enhancements (Planned)

There are two planned approaches for **residual noise suppression** on top of DeepFilterNet:

#### Option 1: NLMS Adaptive Filter
- **Requires:** A second microphone (reference mic) pointed away from the speaker to capture ambient noise
- **How it works:** The Normalized Least Mean Squares (NLMS) algorithm adaptively subtracts the reference noise signal from the primary mic signal
- **Best for:** Environments with consistent, directional noise sources
- **Pros:** Very effective when a clean noise reference is available; low computational cost
- **Cons:** Requires additional hardware (second mic); needs careful mic placement

#### Option 2: Spectral Gating + Wiener Filter
- **Requires:** No additional hardware
- **How it works:**
  - **Spectral Gating:** Estimates the noise floor from quiet segments and gates (silences) frequency bins below the threshold
  - **Wiener Filter:** Estimates the clean speech spectrum and applies an optimal filter to minimize mean squared error between the estimate and actual clean signal
- **Best for:** General-purpose residual noise cleanup after DeepFilterNet
- **Pros:** No extra hardware needed; works as a post-processing stage
- **Cons:** May slightly affect speech quality if over-applied

---

## 📋 Prerequisites

- **Python 3.12** (Python 3.14 is NOT compatible with DeepFilterNet)
- **Rust & Cargo** — required to compile `deepfilterlib` ([Install via rustup](https://rustup.rs/))
- **Visual Studio Build Tools** — required on Windows for Rust compilation
- A microphone
- (Optional) A phone on the same WiFi network for remote listening

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/deepfilter-anc.git
cd deepfilter-anc
```

### 2. Install Rust (if not already installed)

```bash
# Windows — download and run from https://rustup.rs/
# Linux/macOS:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 3. Create Virtual Environment

```bash
# Windows
py -3.12 -m venv venv
.\venv\Scripts\Activate

# Linux/macOS
python3.12 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Windows Users:** If you get a "Cargo not found" error, make sure `%USERPROFILE%\.cargo\bin` is on your PATH. You may need to restart your terminal after installing Rust.

> ⚠️ **Windows 11 Users:** If you get "Application Control policy has blocked this file", you need to disable **Smart App Control** in Windows Security → App & browser control → Smart App Control settings.

### 5. Run the Server

```bash
python server.py
```

The first run will automatically download the DeepFilterNet3 model weights (~100MB).

### 6. Open the Dashboard

- **On your laptop:** http://localhost:5000
- **On your phone:** http://YOUR_LOCAL_IP:5000 (shown in the terminal output)

### 7. Start Noise Cancellation

1. Click the ⏻ **Power** button on the dashboard to start capturing & processing
2. On your phone, tap the 🎧 **Listen** button and wait for the buffer to fill
3. Enjoy clean, noise-free audio!

---

## 📁 Project Structure

```
deepfilter-anc/
├── server.py          # Flask + Socket.IO server with DeepFilterNet processing
├── index.html         # Web dashboard (visualizations + audio playback)
├── main.py            # Standalone CLI version (no web UI)
├── requirements.txt   # Python dependencies
├── README.md          # This file
└── venv/              # Virtual environment (not tracked in git)
```

### Key Files

| File | Description |
|------|-------------|
| `server.py` | Main server — captures mic audio, processes through DeepFilterNet3, streams enhanced audio + metrics via Socket.IO |
| `index.html` | Real-time dashboard with waveform visualizations, level meters, noise reduction ring, and gapless audio playback via Web Audio API |
| `main.py` | Simple standalone script that runs DeepFilterNet in a terminal (no web UI, plays through laptop speakers) |

---

## ⚙️ Configuration

You can tune these parameters in `server.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CAPTURE_BLOCK` | `480` (10ms) | Mic capture block size. Smaller = more responsive capture |
| `PROCESS_SIZE` | `SR * 3` (3s) | DeepFilterNet processing chunk size. Larger = better quality but more latency |

And in `index.html`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PRE_BUFFER_SECONDS` | `3.5` | Seconds of audio to buffer before playback starts |
| `RING_CAPACITY` | `48000 * 30` | Ring buffer size in samples (30 seconds) |

### Latency vs Quality Tradeoff

| PROCESS_SIZE | Latency | Quality |
|-------------|---------|---------|
| `SR // 2` (0.5s) | ~2s | Good |
| `SR * 1` (1s) | ~3s | Better |
| `SR * 3` (3s) | ~6.5s | Best (near-offline) |

---

## 🌐 Network Requirements

- All devices must be on the **same WiFi network**
- The server binds to `0.0.0.0:5000` (all network interfaces)
- No internet connection required after initial setup

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cargo not found` | Install Rust via [rustup.rs](https://rustup.rs/), restart terminal |
| `Application Control policy blocked` | Disable Smart App Control (Windows 11) |
| `Python 3.14 not compatible` | Use Python 3.12 — `py -3.12 -m venv venv` |
| `torchaudio.backend.common not found` | Already patched — the code handles this automatically |
| Phone can't connect | Check firewall allows port 5000; ensure same WiFi network |
| Audio is choppy | Increase `PROCESS_SIZE` and `PRE_BUFFER_SECONDS` |

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) by Hendrik Schröter — the neural network powering the noise cancellation
- [PyTorch](https://pytorch.org/) — deep learning framework
- [sounddevice](https://python-sounddevice.readthedocs.io/) — audio I/O
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/) — real-time web communication
