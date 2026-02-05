# Test Environments: AudioSep & FlowSep

Correct `pyproject.toml` specs for uv so both projects install without missing modules.

## AudioSep (`test_audiosep/`)

**Python:** 3.10

```bash
cd test_audiosep
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch torchaudio torchvision "torchlibrosa==0.1.0" lightning transformers librosa scipy soundfile pyyaml ftfy braceexpand webdataset h5py huggingface-hub pandas wget regex tqdm scikit-learn
```

**Key dependencies:**
- torch, torchaudio, torchvision
- torchlibrosa==0.1.0
- lightning, transformers
- librosa, scipy, soundfile
- ftfy, braceexpand, webdataset, h5py
- huggingface-hub, pandas, wget
- pyyaml, regex, tqdm, scikit-learn

**Checkpoints** (download before running):
- `checkpoint/audiosep_base_4M_steps.ckpt`
- `checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt`

Use `download_models.py` or HuggingFace Space `badayvedat/AudioSep`.

---

## FlowSep (`test_flowsep/`)

**Python:** 3.10

```bash
cd test_flowsep
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch torchaudio torchvision pytorch-lightning transformers diffusers einops omegaconf kornia timm "librosa>=0.9,<0.10" soundfile resampy pyyaml tqdm pandas wandb matplotlib ipdb safetensors taming-transformers
```

**Key dependencies:**
- torch, torchaudio, torchvision
- pytorch-lightning, transformers, diffusers
- einops, omegaconf, kornia, timm
- librosa>=0.9,<0.10 (API compatibility)
- soundfile, resampy
- wandb, matplotlib, ipdb
- taming-transformers (LPIPS, discriminator)

**Checkpoints** (from Zenodo 13869712):
- `model_logs/pretrained/v2_100k.ckpt`
- `model_logs/pretrained/vae.ckpt`

**Note:** FlowSep targets CUDA. On Mac, MPS/CPU patches are applied but the project has many CUDA assumptions. The `taming-transformers` PyPI package may not match the expected `taming.modules.*` layout; if imports fail, clone and copy:

```bash
git clone https://github.com/CompVis/taming-transformers
cp -r taming-transformers/taming test_flowsep/
```

---

## Quick reference

| Project   | Python | Main stack                    | Status      |
|----------|--------|-------------------------------|-------------|
| AudioSep | 3.10   | torch, lightning, CLAP       | ✅ Working  |
| FlowSep  | 3.10   | torch, lightning, diffusers   | ⚠️ CUDA/taming |
