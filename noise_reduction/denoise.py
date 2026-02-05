#!/usr/bin/env python3
"""
Envelope-Matched Noise Reduction Tool

Removes amplitude-correlated noise from audio using a noise profile.
Designed for noise that follows the audio signal's amplitude envelope.
Supports both mono and stereo audio files.

Usage:
    python denoise.py <audio_file> <noise_profile> [output_file]

Example:
    python denoise.py input.wav noise-profile.wav denoised_output.wav
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf


def envelope_matched_subtraction(audio, noise, sr, n_fft=2048, hop_length=512, 
                                 envelope_window=0.05, subtraction_factor=0.9,
                                 release_time=0.1, attack_time=0.01,
                                 transient_protection=True, hold_time=0.05):
    """
    Removes noise that follows the audio amplitude envelope.
    
    This method matches the noise envelope to the audio envelope before 
    performing spectral subtraction. Designed for dynamic noise that 
    modulates with the source signal.
    
    Args:
        audio: Input audio signal (numpy array) - mono, shape (samples,)
        noise: Noise profile signal (numpy array) - mono, shape (samples,)
        sr: Sample rate
        n_fft: FFT size for STFT (default: 2048)
        hop_length: Hop length for STFT (default: 512)
        envelope_window: Window size in seconds for envelope calculation (default: 0.05)
        subtraction_factor: How aggressively to subtract noise 0-1 (default: 0.9)
        release_time: Envelope release time in seconds (default: 0.1) - slower = less pumping
        attack_time: Envelope attack time in seconds (default: 0.01)
        transient_protection: Keep noise reduction consistent during transients (default: True)
        hold_time: Time to hold noise reduction level after transient detection (default: 0.05s)
    
    Returns:
        Denoised audio signal (numpy array)
    """
    # Calculate audio envelope (RMS)
    frame_length = int(envelope_window * sr)
    audio_envelope = np.array([
        np.sqrt(np.mean(audio[i:i+frame_length]**2)) 
        for i in range(0, len(audio) - frame_length, hop_length)
    ])
    
    # Apply attack/release smoothing to prevent pumping
    if len(audio_envelope) > 1:
        # Convert time to samples (at envelope rate)
        envelope_sr = sr / hop_length
        attack_samples = max(1, int(attack_time * envelope_sr))
        release_samples = max(1, int(release_time * envelope_sr))
        hold_samples = max(1, int(hold_time * envelope_sr))
        
        # Detect transients (sudden increases)
        transient_mask = np.zeros(len(audio_envelope), dtype=bool)
        if transient_protection:
            envelope_diff = np.diff(audio_envelope, prepend=audio_envelope[0])
            threshold = np.std(envelope_diff) * 2
            transient_indices = np.where(envelope_diff > threshold)[0]
            
            # Extend transient mask with hold time
            for idx in transient_indices:
                start = max(0, idx)
                end = min(len(transient_mask), idx + hold_samples)
                transient_mask[start:end] = True
        
        # Attack/release envelope follower
        smoothed_envelope = np.zeros_like(audio_envelope)
        smoothed_envelope[0] = audio_envelope[0]
        
        for i in range(1, len(audio_envelope)):
            if audio_envelope[i] > smoothed_envelope[i-1]:
                # Attack: fast rise
                alpha = 1.0 / attack_samples
                smoothed_envelope[i] = alpha * audio_envelope[i] + (1 - alpha) * smoothed_envelope[i-1]
            else:
                # Release: slow decay (prevents pumping)
                alpha = 1.0 / release_samples
                smoothed_envelope[i] = alpha * audio_envelope[i] + (1 - alpha) * smoothed_envelope[i-1]
        
        # Apply transient protection: during transients, use lower envelope value
        # This keeps noise reduction more consistent during attacks
        if transient_protection:
            # Find the envelope level just before each transient region
            protected_envelope = smoothed_envelope.copy()
            in_transient = False
            pre_transient_level = smoothed_envelope[0]
            
            for i in range(len(smoothed_envelope)):
                if transient_mask[i]:
                    if not in_transient:
                        # Just entered transient, remember pre-transient level
                        pre_transient_level = smoothed_envelope[max(0, i-1)]
                        in_transient = True
                    # During transient, use the lower of current or pre-transient level
                    # This keeps noise reduction high during the attack
                    protected_envelope[i] = min(smoothed_envelope[i], pre_transient_level * 1.5)
                else:
                    in_transient = False
            
            audio_envelope = protected_envelope
        else:
            audio_envelope = smoothed_envelope
    
    # Calculate noise envelope
    noise_envelope = np.array([
        np.sqrt(np.mean(noise[i:i+frame_length]**2)) 
        for i in range(0, min(len(noise) - frame_length, len(audio) - frame_length), hop_length)
    ])
    
    # STFT
    audio_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    noise_stft = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length)
    
    # Calculate noise spectral profile (averaged)
    noise_profile = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
    
    # Scale noise profile by ratio of envelopes
    if len(audio_envelope) > 0 and len(noise_envelope) > 0:
        # Extend envelope to match STFT frames
        n_frames = audio_stft.shape[1]
        audio_env_interp = np.interp(
            np.linspace(0, len(audio_envelope)-1, n_frames),
            np.arange(len(audio_envelope)),
            audio_envelope
        )
        noise_env_mean = np.mean(noise_envelope)
        
        # Scale factor: how much louder/quieter is the audio compared to noise
        scale_factor = audio_env_interp / (noise_env_mean + 1e-10)
        scale_factor = np.clip(scale_factor, 0.1, 10.0)  # Limit scaling
        
        # Create time-varying noise estimate
        noise_estimate = noise_profile * scale_factor
        
        # Spectral subtraction with magnitude flooring
        audio_mag = np.abs(audio_stft)
        audio_phase = np.angle(audio_stft)
        
        output_mag = np.maximum(audio_mag - subtraction_factor * noise_estimate, 0.05 * audio_mag)
        output_stft = output_mag * np.exp(1j * audio_phase)
        
        output = librosa.istft(output_stft, hop_length=hop_length)
        return output
    else:
        return audio


def denoise_audio(audio_path, noise_path, output_path=None, subtraction_factor=0.9,
                  release_time=0.1, attack_time=0.01):
    """
    Main function to denoise an audio file using a noise profile.
    Handles both mono and stereo audio files.
    
    Args:
        audio_path: Path to the audio file to denoise
        noise_path: Path to the noise profile audio file
        output_path: Path for the output file (optional, auto-generated if not provided)
        subtraction_factor: Noise reduction strength 0-1 (default: 0.9)
    
    Returns:
        Path to the denoised output file
    """
    print(f"📂 Loading audio: {os.path.basename(audio_path)}")
    
    # Load audio preserving original channels using soundfile
    audio, sr = sf.read(audio_path)
    
    # Determine if stereo or mono
    if len(audio.shape) == 1:
        n_channels = 1
        is_stereo = False
        print(f"   Duration: {len(audio)/sr:.2f}s @ {sr}Hz (MONO)")
    else:
        n_channels = audio.shape[1]
        is_stereo = True
        print(f"   Duration: {audio.shape[0]/sr:.2f}s @ {sr}Hz (STEREO - {n_channels} channels)")
    
    print(f"📂 Loading noise profile: {os.path.basename(noise_path)}")
    noise, noise_sr = sf.read(noise_path)
    
    # If noise is stereo, convert to mono for the profile
    if len(noise.shape) > 1:
        noise = np.mean(noise, axis=1)
        print(f"   Duration: {len(noise)/noise_sr:.2f}s @ {noise_sr}Hz (converted to mono for profile)")
    else:
        print(f"   Duration: {len(noise)/noise_sr:.2f}s @ {noise_sr}Hz")
    
    # Resample noise to match audio sample rate if different
    if noise_sr != sr:
        noise = librosa.resample(noise.astype(np.float32), orig_sr=noise_sr, target_sr=sr)
        print(f"   Resampled noise profile to {sr}Hz")
    
    print("🔧 Applying envelope-matched noise reduction...")
    
    if is_stereo:
        # Process each channel separately
        denoised_channels = []
        for ch in range(n_channels):
            print(f"   Processing channel {ch + 1}/{n_channels}...")
            channel_audio = audio[:, ch].astype(np.float32)
            denoised_ch = envelope_matched_subtraction(
                channel_audio, noise.astype(np.float32), sr, 
                subtraction_factor=subtraction_factor,
                release_time=release_time,
                attack_time=attack_time
            )
            denoised_channels.append(denoised_ch)
        
        # Find minimum length (ISTFT may produce slightly different lengths)
        min_len = min(len(ch) for ch in denoised_channels)
        denoised = np.column_stack([ch[:min_len] for ch in denoised_channels])
    else:
        # Mono processing
        denoised = envelope_matched_subtraction(
            audio.astype(np.float32), noise.astype(np.float32), sr, 
            subtraction_factor=subtraction_factor,
            release_time=release_time,
            attack_time=attack_time
        )
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(os.path.dirname(audio_path), f"{base_name}_denoised.wav")
    
    print(f"💾 Saving: {output_path}")
    sf.write(output_path, denoised, sr)
    
    print("✅ Done!")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nModes:")
        print("  (default)  Envelope-matched subtraction (100ms release)")
        print("  --drums    Envelope-matched with slower release (300ms) for kicks/drums")
        print("  --slow     Envelope-matched with very slow release (500ms) for pads")
        sys.exit(1)
    
    # Check for mode flags
    args = sys.argv[1:]
    drums_mode = "--drums" in args
    slow_mode = "--slow" in args
    
    # Remove flags from args
    args = [a for a in args if not a.startswith("--")]
    
    if len(args) < 2:
        print("Error: Need at least audio_file and noise_profile")
        sys.exit(1)
    
    audio_file = args[0]
    noise_file = args[1]
    output_file = args[2] if len(args) > 2 else None
    
    # Use envelope-matched method
    if slow_mode:
        release_time = 0.5  # Very slow - 500ms
        print("🥁 Using SLOW mode (500ms release) for sustained sounds")
    elif drums_mode:
        release_time = 0.3  # Slower - 300ms for drums
        print("🥁 Using DRUMS mode (300ms release) to prevent pumping")
    else:
        release_time = 0.1  # Default - 100ms
    
    denoise_audio(audio_file, noise_file, output_file, release_time=release_time)
