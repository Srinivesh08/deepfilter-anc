import sounddevice as sd
import numpy as np
import torch
from df.enhance import init_df, enhance

model, df_state, _ = init_df()
SR = df_state.sr()
BLOCK_SIZE = 2048  # smaller = lower latency, but more CPU overhead per block

def callback(indata, outdata, frames, time_info, status):
    if status:
        print(status)
    audio = torch.from_numpy(indata[:, 0]).unsqueeze(0)
    enhanced = enhance(model, df_state, audio)
    enhanced_np = enhanced.squeeze(0).cpu().numpy()
    # Pad/trim to match output block size exactly
    if len(enhanced_np) < frames:
        enhanced_np = np.pad(enhanced_np, (0, frames - len(enhanced_np)))
    else:
        enhanced_np = enhanced_np[:frames]
    outdata[:, 0] = enhanced_np

print(f"Starting real-time enhancement at {SR} Hz. Press Ctrl+C to stop.")
with sd.Stream(samplerate=SR, blocksize=BLOCK_SIZE, channels=1, callback=callback):
    input("Press Enter to stop...\n")
