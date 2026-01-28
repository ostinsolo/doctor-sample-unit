import math
import os
import wave


def write_wav(path, seconds=3.0, sample_rate=44100, freqs=(220.0, 440.0)):
    num_samples = int(seconds * sample_rate)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)

        for i in range(num_samples):
            t = i / sample_rate
            left = sum(math.sin(2 * math.pi * f * t) for f in freqs) / len(freqs)
            right = sum(math.sin(2 * math.pi * (f * 1.5) * t) for f in freqs) / len(freqs)
            left_i = int(max(-1.0, min(1.0, left)) * 32767)
            right_i = int(max(-1.0, min(1.0, right)) * 32767)
            wf.writeframesraw(left_i.to_bytes(2, "little", signed=True) +
                              right_i.to_bytes(2, "little", signed=True))


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, "audio")
    write_wav(os.path.join(audio_dir, "test_mix.wav"), seconds=3.5, freqs=(220.0, 330.0, 440.0))
    write_wav(os.path.join(audio_dir, "test_mix_alt.wav"), seconds=4.0, freqs=(110.0, 275.0, 550.0))
    print(f"Wrote test files to: {audio_dir}")


if __name__ == "__main__":
    main()
