import soundfile as sf
import numpy as np
import os

# Check the newest output
wav_path = r'C:\Users\soloo\Documents\DSU-VSTOPIA\Output\5_18_40_1_27_2026__bsroformer-scnet_xl_ihf_2026-01-27_05-19-24\5_18_40_1_27_2026_\bass.wav'

print(f'File: {wav_path}')
print(f'Exists: {os.path.exists(wav_path)}')
print(f'Size: {os.path.getsize(wav_path)} bytes')
print()

try:
    info = sf.info(wav_path)
    print(f'Format: {info.format}')
    print(f'Subtype: {info.subtype}')
    print(f'Channels: {info.channels}')
    print(f'Samplerate: {info.samplerate}')
    print(f'Frames: {info.frames}')
    print(f'Duration: {info.duration:.2f}s')
    
    # Read and check audio data
    data, sr = sf.read(wav_path)
    print(f'\nAudio data:')
    print(f'  Shape: {data.shape}')
    print(f'  Type: {data.dtype}')
    print(f'  Min: {data.min():.4f}')
    print(f'  Max: {data.max():.4f}')
    print(f'  RMS: {np.sqrt(np.mean(data**2)):.4f}')
    
    # Check if audio is silent
    if np.max(np.abs(data)) < 0.001:
        print('\n  WARNING: Audio appears silent!')
    else:
        print('\n  Audio has content (not silent)')
        
except Exception as e:
    print(f'Error: {e}')
