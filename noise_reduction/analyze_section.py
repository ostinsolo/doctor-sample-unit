#!/usr/bin/env python3
"""
Analyze a specific section of audio and visualize what's happening.
"""

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os

def analyze_section(audio_path, start_time, end_time, output_dir=None):
    """Analyze a specific section of audio."""
    
    print(f"📂 Loading: {os.path.basename(audio_path)}")
    audio, sr = sf.read(audio_path)
    
    # Convert to mono for analysis
    if len(audio.shape) > 1:
        audio_mono = np.mean(audio, axis=1)
    else:
        audio_mono = audio
    
    # Extract section
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    section = audio_mono[start_sample:end_sample]
    
    print(f"📍 Analyzing {start_time:.2f}s - {end_time:.2f}s")
    print(f"   Duration: {len(section)/sr:.2f}s")
    
    # Compute spectrogram
    n_fft = 2048
    hop_length = 512
    
    stft = librosa.stft(section.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
    mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    # Find frequency peaks
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mean_spectrum = np.mean(np.abs(stft), axis=1)
    
    # Find peaks in the spectrum
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(mean_spectrum, height=np.max(mean_spectrum) * 0.1)
    
    print(f"\n🔍 Frequency peaks found:")
    for peak in peaks[:10]:  # Top 10 peaks
        print(f"   {freqs[peak]:.0f} Hz")
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Spectrogram
    img = librosa.display.specshow(mag_db, sr=sr, hop_length=hop_length, 
                                    x_axis='time', y_axis='hz', ax=axes[0])
    axes[0].set_title(f'Spectrogram: {start_time:.2f}s - {end_time:.2f}s')
    axes[0].set_ylim(0, 10000)  # Focus on 0-10kHz
    fig.colorbar(img, ax=axes[0], format='%+2.0f dB')
    
    # Spectrum
    axes[1].plot(freqs, librosa.amplitude_to_db(mean_spectrum))
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title('Average Spectrum')
    axes[1].set_xlim(0, 10000)
    axes[1].axvline(x=2000, color='r', linestyle='--', alpha=0.5, label='2kHz')
    axes[1].axvline(x=4000, color='r', linestyle='--', alpha=0.5, label='4kHz')
    axes[1].axvline(x=6000, color='r', linestyle='--', alpha=0.5, label='6kHz')
    axes[1].legend()
    
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = os.path.dirname(audio_path)
    
    plot_path = os.path.join(output_dir, 'analysis_spectrogram.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\n📊 Saved spectrogram: {plot_path}")
    plt.close()
    
    return freqs, mean_spectrum, peaks


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python analyze_section.py <audio_file> <start_time> <end_time>")
        print("\nExample:")
        print("  python analyze_section.py audio.wav 2.5 4.0")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    start_time = float(sys.argv[2])
    end_time = float(sys.argv[3])
    
    print("=" * 60)
    print("ANALYZING AUDIO SECTION")
    print("=" * 60)
    
    analyze_section(audio_path, start_time, end_time)
    
    print("\n✅ Done! Check the spectrogram output.")


