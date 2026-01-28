"""Create a 4-minute test audio file by looping the short test audio."""

import os
import soundfile as sf
import numpy as np

SHORT_AUDIO = r"C:\Users\soloo\Desktop\shared_runtime\tests\audio\test_mix.wav"
LONG_AUDIO = r"C:\Users\soloo\Desktop\shared_runtime\tests\audio\test_mix_4min.wav"
TARGET_DURATION = 4 * 60  # 4 minutes in seconds

def main():
    print(f"Loading: {SHORT_AUDIO}")
    audio, sr = sf.read(SHORT_AUDIO)
    
    duration = len(audio) / sr
    print(f"Original duration: {duration:.2f}s at {sr}Hz")
    
    # Calculate how many times to loop
    loops_needed = int(np.ceil(TARGET_DURATION / duration))
    print(f"Looping {loops_needed} times to reach {TARGET_DURATION}s")
    
    # Create long audio by tiling
    long_audio = np.tile(audio, (loops_needed, 1) if audio.ndim == 2 else loops_needed)
    
    # Trim to exact duration
    target_samples = TARGET_DURATION * sr
    long_audio = long_audio[:target_samples]
    
    actual_duration = len(long_audio) / sr
    print(f"Final duration: {actual_duration:.2f}s")
    
    # Save
    print(f"Saving to: {LONG_AUDIO}")
    sf.write(LONG_AUDIO, long_audio, sr)
    
    # Verify
    size_mb = os.path.getsize(LONG_AUDIO) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
    print("Done!")

if __name__ == "__main__":
    main()
