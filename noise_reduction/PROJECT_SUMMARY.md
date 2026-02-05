# Project Summary: Noise Reduction Techniques

## ✅ Working Technique: Envelope-Matched Subtraction

**File:** `denoise.py`

This is the **primary and working** noise reduction method. It's designed specifically for **amplitude-correlated noise** - noise that follows the audio signal's amplitude envelope.

### Why This Works
- Dynamically scales noise reduction based on audio envelope
- Handles noise that gets louder when audio gets louder
- Includes attack/release smoothing to prevent artifacts
- Has transient protection for drums/kicks

### Usage
```bash
python denoise.py <audio_file> <noise_profile> [output_file]

# Modes:
python denoise.py --drums <file> <noise>  # For kicks/drums (300ms release)
python denoise.py --slow <file> <noise>   # For pads/sustained (500ms release)
```

---

## 🔧 Utility Scripts

### `analyze_section.py` - Audio Analysis Tool
**Status:** Utility - For debugging and analysis

Analyzes specific sections of audio and creates spectrograms. Useful for:
- Finding frequency peaks
- Visualizing noise characteristics
- Testing frequency band removal

---

## ❌ Techniques That Were Tested But Didn't Work

According to the README, these were tested but failed:
- Spectral Gating (Audacity-style)
- noisereduce (stationary/non-stationary)
- DeepFilterNet3 (deep learning)
- Demucs (source separation)
- VoiceFixer (AI restoration)
- Multi-band de-hiss
- Birdie/artifact removal methods (fix_birdies.py) - Could not successfully remove musical noise artifacts

---

## 📦 Dependencies

See `requirements.txt` for the minimal set of dependencies needed.

**Core:**
- numpy
- librosa
- soundfile
- scipy

**Optional:**
- matplotlib (for analysis scripts)

**Not Needed (can be removed):**
- DeepFilterNet
- demucs
- voicefixer
- tensorflow
- torch
- All Jupyter/notebook packages (unless you use notebooks)

---

## 🎯 Recommended Workflow

1. **Start with:** `denoise.py` (envelope-matched subtraction)
2. **For analysis:** Use `analyze_section.py` to understand the noise

---

## 📁 File Organization

```
noise_reduction/
├── denoise.py              ✅ PRIMARY - Working technique
├── analyze_section.py      🔧 UTILITY - Analysis tool
├── requirements.txt        📦 Dependencies
├── README.md               📖 Full documentation
├── PROJECT_SUMMARY.md      📋 This file
└── noise-profile.wav       🎵 Your noise sample
```
