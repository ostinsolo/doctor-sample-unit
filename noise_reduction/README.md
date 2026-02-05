# Envelope-Matched Noise Reduction

A specialized noise reduction tool designed for **amplitude-correlated noise** — noise that follows the audio signal's amplitude envelope rather than static background noise (like hiss that gets louder when the audio gets louder).

## ⭐ Working Technique

**The primary working method is Envelope-Matched Subtraction** (implemented in `denoise.py`). This is the technique that has shown to work well for amplitude-correlated noise.

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for a quick overview of which techniques work and which are experimental.

## The Problem

Standard noise reduction techniques assume noise is **stationary** (constant level). But some noise types are more challenging:

- **Amplitude-correlated noise**: Noise that gets louder when the audio gets louder
- **Equipment artifacts**: Hiss or buzz that tracks the signal level
- **Modulated interference**: Noise that follows the source's envelope
- **Recording chain noise**: Preamp hiss that scales with gain

Traditional spectral denoisers fail on these because they expect noise to be constant. When noise follows the signal, these tools either:
- Remove too little (noise still audible during loud parts)
- Remove too much (destroy the audio trying to catch the noise)

## Our Solution: Envelope-Matched Subtraction

This tool uses a **noise profile** to learn the noise characteristics, then **dynamically scales** the noise reduction based on the audio's amplitude envelope.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENVELOPE-MATCHED SUBTRACTION                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   AUDIO INPUT                         NOISE PROFILE                     │
│        │                                    │                           │
│        ▼                                    ▼                           │
│   ┌─────────┐                         ┌─────────┐                       │
│   │  STFT   │                         │  STFT   │                       │
│   │(Spectro)│                         │(Spectro)│                       │
│   └────┬────┘                         └────┬────┘                       │
│        │                                   │                            │
│        ▼                                   ▼                            │
│   ┌──────────┐                       ┌─────────────┐                    │
│   │ Envelope │                       │Noise Profile│                    │
│   │ Tracking │                       │  (Averaged) │                    │
│   └────┬─────┘                       └──────┬──────┘                    │
│        │                                    │                           │
│        │    ┌───────────────────────────────┘                           │
│        │    │                                                           │
│        ▼    ▼                                                           │
│   ┌─────────────────┐                                                   │
│   │ Scale Noise to  │  ← Key innovation: noise estimate                 │
│   │ Match Envelope  │    follows the audio amplitude                    │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │    Spectral     │  Subtract scaled noise from                       │
│   │   Subtraction   │  each time-frequency bin                          │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │     ISTFT       │                                                   │
│   │ (Reconstruct)   │                                                   │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│     DENOISED OUTPUT                                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Process

1. **Load Audio & Noise Profile**
   - Audio is loaded preserving stereo channels
   - Noise profile is converted to mono for analysis

2. **STFT (Short-Time Fourier Transform)**
   - Converts audio from time domain to time-frequency domain
   - Creates a spectrogram showing frequency content over time
   - Uses 2048-sample windows with 512-sample hops

3. **Envelope Tracking**
   - Calculates the RMS (root mean square) amplitude over time
   - Applies attack/release smoothing to prevent artifacts:
     - **Attack**: How fast the envelope rises (10ms default)
     - **Release**: How fast the envelope falls (100-300ms)
   - Slower release prevents "pumping" on drums/kicks

4. **Transient Protection** (Drums Mode)
   - Detects sudden loud attacks (kicks, snares, etc.)
   - Holds the noise reduction level during transients
   - Prevents noise from "poking through" during attacks

5. **Noise Scaling**
   - The noise profile is scaled based on the current envelope level
   - Louder audio → more noise estimated → more subtraction
   - Quieter audio → less noise estimated → less subtraction

6. **Spectral Subtraction**
   - For each time-frequency bin:
     ```
     output = max(audio - noise_estimate, floor)
     ```
   - A floor (5% of original) prevents complete silence

7. **ISTFT (Inverse STFT)**
   - Converts back from time-frequency to time domain
   - Reconstructs the denoised audio waveform

8. **Stereo Processing**
   - Each channel processed independently
   - Preserves the original stereo image

## Installation

### Recommended: Using `uv` (Fast & Modern)

```bash
# Create virtual environment and install dependencies (one command)
uv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
uv pip install -r requirements.txt
```

### Alternative: Using `pip` (Traditional)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy librosa soundfile scipy matplotlib
```

**Note:** If you encounter compilation errors, see [RECREATE_VENV.md](RECREATE_VENV.md) for troubleshooting.

## Usage

### Basic Usage

```bash
# Activate environment
source venv/bin/activate

# Denoise an audio file
python denoise.py "input.wav" "noise-profile.wav"

# Specify output filename
python denoise.py "input.wav" "noise-profile.wav" "output.wav"
```

### Drums Mode (for kicks, drums, percussive sounds)

```bash
python denoise.py --drums "kick.wav" "noise-profile.wav"
```

Use this when you hear **pumping** (noise bouncing with the beat) or **noise on transient attacks**.

### Slow Mode (for sustained sounds, pads)

```bash
python denoise.py --slow "pad.wav" "noise-profile.wav"
```

Use this for sustained sounds where you want very smooth noise reduction.

### Mode Comparison

| Mode | Algorithm | Release Time | Best For |
|------|-----------|--------------|----------|
| Default | Envelope-matched | 100ms | Vocals, melodic content |
| `--drums` | Envelope-matched | 300ms | Kicks, drums, percussive |
| `--slow` | Envelope-matched | 500ms | Pads, sustained sounds |

## Parameters (Advanced)

The core algorithm accepts these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_fft` | 2048 | FFT size. Larger = better frequency resolution |
| `hop_length` | 512 | STFT hop. Smaller = smoother, more CPU |
| `envelope_window` | 0.05s | Window for RMS envelope calculation |
| `subtraction_factor` | 0.9 | How aggressively to subtract (0-1) |
| `release_time` | 0.1s | Envelope release time (prevents pumping) |
| `attack_time` | 0.01s | Envelope attack time |
| `hold_time` | 0.05s | Hold time after transient detection |

## Creating a Good Noise Profile

For best results, your noise profile should:

1. **Contain ONLY the noise** — no desired audio content
2. **Be recorded with the same equipment** and settings as your audio
3. **Be at least 2-3 seconds long** for reliable statistics
4. **Capture the full dynamic range** of the noise

### How to Record a Noise Profile

1. Record a silent passage where only the noise is present
2. Or isolate a section between audio events (before/after a take)
3. Or record with the same gain settings but no input signal

### What Makes a Good Profile

```
GOOD: [steady hiss/noise for 3+ seconds]
BAD:  [noise with occasional audio bleed]
BAD:  [very short noise sample < 1 second]
BAD:  [noise recorded at different gain setting]
```

## Troubleshooting

### Problem: Pumping (noise bounces with the beat)

**Solution**: Use `--drums` or `--slow` mode for longer release time

```bash
python denoise.py --drums "file.wav" "noise-profile.wav"
```

### Problem: Noise audible on transient attacks (kicks, snares)

**Solution**: The transient protection should handle this. If not, the noise might not be amplitude-correlated — try a different approach.

### Problem: Audio sounds muffled/dull

**Cause**: Over-subtraction removing wanted high frequencies

**Solutions**:
- Use a more accurate noise profile
- The algorithm already preserves 5% of original signal as floor
- Consider if this noise type needs a different approach

### Problem: Artifacts (musical noise, chirping)

**Cause**: Spectral subtraction can create "musical noise" artifacts

**Note**: Various methods were tested to remove these artifacts but none proved successful. If artifacts are problematic, you may need to:
- Use longer FFT size (modify code: `n_fft=4096`)
- Accept some residual noise for cleaner sound
- Adjust subtraction factor to be less aggressive

## Technical Details

### Why Envelope Matching Works

Traditional spectral subtraction:
```
output[f,t] = audio[f,t] - noise_profile[f]  (constant)
```

Our envelope-matched approach:
```
scale = audio_envelope[t] / noise_envelope_mean
output[f,t] = audio[f,t] - noise_profile[f] * scale  (dynamic)
```

This dynamic scaling means:
- During loud passages: more aggressive noise reduction
- During quiet passages: gentler noise reduction
- Noise reduction **follows** the signal, just like the noise does

### Attack/Release Smoothing

Without smoothing, the envelope would change instantly, causing artifacts:

```
Raw envelope:      ▄▄▄▄████▄▄▄▄▄▄▄████▄▄▄▄  (instant changes)
Smoothed:          ▁▂▃▄████▇▆▅▄▃▂▁▂▃████▇▆  (gradual transitions)
```

The attack/release times control how the envelope responds:
- **Fast attack (10ms)**: Quickly follows rising signals
- **Slow release (100-300ms)**: Slowly decays after loud sounds

### Transient Protection

Transients (drum hits, plucks) have very fast attacks. Without protection:

```
Kick hit:     │████│         │████│
Envelope:     │▄▄▄▄│▃▂▁▁▁▁▁▁▁│▄▄▄▄│
Noise redux:  │low │high     │low │  ← noise pokes through!
```

With transient protection:

```
Kick hit:     │████│         │████│
Envelope:     │▂▂▂▂│▂▂▁▁▁▁▁▁▁│▂▂▂▂│  ← held during transient
Noise redux:  │med │med→high │med │  ← consistent!
```

## Methods Tested During Development

We tested many approaches before settling on envelope-matched subtraction:

| Method | Result | Why It Failed/Succeeded |
|--------|--------|------------------------|
| Spectral Gating (Audacity-style) | ❌ | Doesn't track amplitude |
| noisereduce (stationary) | ❌ | Assumes constant noise |
| noisereduce (non-stationary) | ❌ | Not enough adaptation |
| DeepFilterNet3 (deep learning) | ❌ | Trained on different noise types |
| Demucs (source separation) | ❌ | Designed for stems, not denoising |
| VoiceFixer (AI restoration) | ❌ | Designed for speech |
| Multi-band de-hiss | ❌ | Too aggressive on high frequencies |
| **Envelope-Matched Subtraction** | ✅ | **Designed for amplitude-correlated noise** |

## File Structure

```
noise_reduction/
├── denoise.py              # Main denoising tool
├── noise-profile.wav       # Your noise sample
├── README.md               # This documentation
├── LICENSE
├── venv/                   # Python virtual environment
└── [audio files]           # Your audio files and outputs
```

## Dependencies

- Python 3.8+
- NumPy — array processing
- librosa — audio analysis (STFT, resampling)
- soundfile — audio I/O (preserves stereo)
- SciPy — signal processing

## License

MIT License — see LICENSE file.

## Contributing

This tool was developed to solve a specific problem (amplitude-correlated noise from a particular recording setup). If you have improvements or find bugs, feel free to contribute!

### Potential Improvements

- [ ] GUI interface
- [ ] Real-time processing
- [ ] Adjustable subtraction factor from command line
- [ ] Batch processing multiple files
- [ ] Noise profile learning from multiple samples
