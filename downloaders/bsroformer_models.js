/**
 * bsroformer.js - BS-RoFormer Module for SplitWizard
 * 
 * State-of-the-art music source separation using BS-RoFormer architecture.
 * Import this module into setupSW+.js for BS-RoFormer support.
 * 
 * Features:
 * - 4-stem separation (bass/drums/vocals/other)
 * - 6-stem separation (+ piano/guitar)
 * - Best-in-class vocal extraction
 * - Drum kit separation (kick/snare/toms/hh/cymbals)
 * - Audio processing (denoise, dereverb)
 * 
 * Created by Ostin Solo
 * Website: ostinsolo.co.uk
 * 
 * @version 1.11.1
 * 
 * v1.11.1 Changes:
 * - Fixed llvmlite.libs detection in GitHub Actions workflow
 * - Windows builds now correctly include msvcp140-*.dll for librosa/numba
 * 
 * v1.11.0 Changes:
 * - Windows: Upgraded to PyTorch 2.10.0 (latest stable)
 * - Windows CUDA: Now uses CUDA 12.6 for better GPU support
 * - Fixed llvmlite.dll bundling (msvcp140-*.dll now included)
 * - Comprehensive torch subpackages for reliable freezing
 * - Skip pip upgrade to avoid WinError 5 permission issues
 * 
 * v1.10.0 Changes:
 * - Upgraded ARM build to PyTorch 2.5.0 for native MPS ComplexFloat support
 * - ~15% faster on Apple Silicon: STFT/ISTFT now run on GPU (was CPU fallback)
 * - Added apollo.py MPS workarounds for look2hear models
 * - Only scatter_add_ still requires CPU (PyTorch limitation)
 * - Full backward compatibility with try/except fallbacks
 * 
 * v1.8.9 Changes:
 * - Key insight: MPS doesn't support ComplexFloat at ALL (not even .cpu())
 * - Strategy: All complex operations stay on CPU, only final recon_audio goes to MPS
 * - ~6-10x faster inference on M-series Macs
 * - CPU fallback for unsupported complex operations (STFT/iSTFT, scatter_add)
 * 
 * v1.8.4 Changes:
 * - FIXED: "bad magic number" error when running on different machines
 * - Added explicit torch subpackages to ensure complete bundling
 * - Added cleanup for .pth files that contained absolute paths
 * - Build is now fully portable across macOS machines
 * - Total: 29 models
 * 
 * v1.8.3 Changes:
 * - NEW: --fast mode for ~2x speedup (vectorized chunking)
 * - NEW: --precision flag for matmul precision control
 * - NEW: --ensemble mode to combine multiple models
 * - Short audio padding (--fast now works on ANY file length)
 * - Intel threading optimizations (KMP_AFFINITY, KMP_BLOCKTIME)
 * - Fixed OpenMP library conflicts
 * 
 * v1.7.0 Changes:
 * - Fixed BS-RoFormer use_shared_bias bug (all BS-RoFormer models now work)
 * - Added 5 special models: aspiration, chorus_male_female, bleed_suppressor, crowd, denoise_debleed
 * - Added guitar_becruily model for single instrument isolation
 */

const fs = require('fs');
const path = require('path');
const { exec, spawn } = require('child_process');
const os = require('os');
const http = require('http');
const https = require('https');
const { URL } = require('url');
const { isFileSizeMatch } = require('./model_integrity');

/**
 * Torch 2.x checkpoints are often ZIP-based. If the download is interrupted,
 * PyTorch throws: "PytorchStreamReader failed reading zip archive: failed finding central directory"
 *
 * This checks for the ZIP End Of Central Directory signature (PK\\x05\\x06)
 * within the last ~66KB (the max comment scan window).
 *
 * Only used for `.ckpt` files to avoid false positives on legacy `.th` pickles.
 * @param {string} filePath
 * @returns {boolean} true if file looks like a ZIP with a central directory
 */
function hasZipCentralDirectory(filePath) {
  try {
    const stat = fs.statSync(filePath);
    if (!stat.isFile()) return false;
    const len = stat.size;
    if (len < 64) return false;

    const fd = fs.openSync(filePath, 'r');
    try {
      const maxScan = 0x10000 + 22; // 64KB + EOCD min size
      const readLen = Math.min(len, maxScan);
      const start = Math.max(0, len - readLen);
      const buf = Buffer.alloc(readLen);
      fs.readSync(fd, buf, 0, readLen, start);

      // EOCD signature
      const sig0 = 0x50, sig1 = 0x4b, sig2 = 0x05, sig3 = 0x06;
      for (let i = buf.length - 4; i >= 0; i--) {
        if (buf[i] === sig0 && buf[i + 1] === sig1 && buf[i + 2] === sig2 && buf[i + 3] === sig3) {
          return true;
        }
      }
      return false;
    } finally {
      fs.closeSync(fd);
    }
  } catch (e) {
    return false;
  }
}

/**
 * Parse human-readable size (e.g., "871 MB", "3.5 GB") to bytes.
 * Returns null if parsing fails.
 * @param {string} sizeStr
 * @returns {number|null}
 */
/**
 * Validate checkpoint file size against expected size string.
 * Uses a small tolerance to account for rounding in the size string.
 * @param {string} checkpointPath
 * @param {string} expectedSizeStr
 * @param {Function} Max
 * @returns {boolean}
 */
function validateCheckpointSize(checkpointPath, expectedSizeStr, Max) {
  const result = isFileSizeMatch(checkpointPath, expectedSizeStr, {
    tolerancePercent: 0.2,
    minBytes: 5 * 1024 * 1024
  });
  if (!result.ok && Max && typeof Max.post === 'function') {
    const actualBytes = result.actualBytes !== null ? result.actualBytes : 'unknown';
    Max.post(`  ⚠ Size mismatch for ${path.basename(checkpointPath)}: expected ~${expectedSizeStr}, got ${actualBytes} bytes`);
  }
  return result.ok;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return 'unknown';
  const mb = bytes / 1e6;
  const mib = bytes / (1024 * 1024);
  return `${mb.toFixed(1)} MB (${mib.toFixed(1)} MiB)`;
}

function formatBytesShort(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return null;
  return `${Math.round(bytes / 1e6)}MB`;
}

function getSizeCachePath(appDirs) {
  const { bsroformerModelsDir } = getBSRoformerDirs(appDirs);
  return path.join(bsroformerModelsDir, 'model_sizes.json');
}

function readSizeCache(appDirs) {
  try {
    const cachePath = getSizeCachePath(appDirs);
    if (!fs.existsSync(cachePath)) return {};
    const data = JSON.parse(fs.readFileSync(cachePath, 'utf8'));
    if (data && typeof data === 'object') {
      return data;
    }
  } catch (e) {}
  return {};
}

function writeSizeCache(appDirs, cache) {
  try {
    const cachePath = getSizeCachePath(appDirs);
    ensureBSRoformerDirs(appDirs);
    fs.writeFileSync(cachePath, JSON.stringify(cache, null, 2));
  } catch (e) {}
}

function fetchRemoteSize(url, Max, callback, depth = 0) {
  if (depth > 5) {
    callback(null);
    return;
  }
  let parsedUrl;
  try {
    parsedUrl = new URL(url);
  } catch (e) {
    callback(null);
    return;
  }
  const protocol = parsedUrl.protocol === 'https:' ? https : http;
  const req = protocol.request(url, { method: 'HEAD' }, (res) => {
    const status = res.statusCode || 0;
    if ([301, 302, 303, 307, 308].includes(status) && res.headers.location) {
      let redirectUrl = res.headers.location;
      try {
        redirectUrl = new URL(redirectUrl, url).href;
      } catch (e) {}
      fetchRemoteSize(redirectUrl, Max, callback, depth + 1);
      return;
    }
    if (status === 403 || status === 405) {
      fetchRemoteSizeViaRange(url, Max, callback, depth);
      return;
    }
    const len = parseInt(res.headers['content-length'], 10);
    if (Number.isFinite(len) && len > 0) {
      callback(len);
      return;
    }
    callback(null);
  });
  req.on('error', () => callback(null));
  req.end();
}

function fetchRemoteSizeViaRange(url, Max, callback, depth = 0) {
  if (depth > 5) {
    callback(null);
    return;
  }
  let parsedUrl;
  try {
    parsedUrl = new URL(url);
  } catch (e) {
    callback(null);
    return;
  }
  const protocol = parsedUrl.protocol === 'https:' ? https : http;
  const req = protocol.request(url, { method: 'GET', headers: { Range: 'bytes=0-0' } }, (res) => {
    const status = res.statusCode || 0;
    if ([301, 302, 303, 307, 308].includes(status) && res.headers.location) {
      let redirectUrl = res.headers.location;
      try {
        redirectUrl = new URL(redirectUrl, url).href;
      } catch (e) {}
      fetchRemoteSizeViaRange(redirectUrl, Max, callback, depth + 1);
      return;
    }
    const contentRange = res.headers['content-range'];
    if (contentRange) {
      const match = contentRange.match(/\/(\d+)$/);
      if (match) {
        const total = parseInt(match[1], 10);
        if (Number.isFinite(total) && total > 0) {
          callback(total);
          res.destroy();
          return;
        }
      }
    }
    const len = parseInt(res.headers['content-length'], 10);
    if (Number.isFinite(len) && len > 0) {
      callback(len);
      res.destroy();
      return;
    }
    callback(null);
  });
  req.on('error', () => callback(null));
  req.end();
}

/**
 * Check if a model checkpoint looks valid (size + zip central directory when applicable).
 * @param {string} checkpointPath
 * @param {Object} model
 * @param {Function} Max
 * @returns {boolean}
 */
function isCheckpointValid(checkpointPath, model, Max) {
  if (!fs.existsSync(checkpointPath)) return false;
  if (model && model.checkpoint && model.checkpoint.size) {
    if (!validateCheckpointSize(checkpointPath, model.checkpoint.size, Max)) return false;
  }
  if (path.extname(checkpointPath).toLowerCase() === '.ckpt') {
    if (!hasZipCentralDirectory(checkpointPath)) {
      if (Max && typeof Max.post === 'function') {
        Max.post(`  ⚠ Detected corrupted/incomplete checkpoint (missing ZIP central directory): ${path.basename(checkpointPath)}`);
      }
      return false;
    }
  }
  return true;
}

// ============================================================================
// DOWNLOAD LOCK - Prevent concurrent executable downloads
// ============================================================================
let bsroformerDownloadInProgress = false;
let bsroformerDownloadCallbacks = []; // Queue of callbacks waiting for download


// ============================================================================
// DIRECTORY CONFIGURATION
// ============================================================================

/**
 * Get BS-RoFormer directories (call this with getAppDirs from setupSW+.js)
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {Object} BS-RoFormer specific directories
 * 
 * NOTE: bsroformerDir is LEGACY - was used for mss-separate executable.
 * DSU shared runtime now provides BS-RoFormer functionality.
 * Only bsroformerModelsDir, bsroformerWeightsDir, bsroformerConfigsDir are actively used.
 */
function getBSRoformerDirs(appDirs) {
  const { thirdParty, modelsDir } = appDirs;
  // LEGACY: bsroformerDir was for mss-separate executable (no longer used with DSU)
  const bsroformerDir = path.join(thirdParty, 'bsroformer');
  // ACTIVE: Model directories for weights and configs
  const bsroformerModelsDir = path.join(modelsDir, 'bsroformer');
  const bsroformerWeightsDir = path.join(bsroformerModelsDir, 'weights');
  const bsroformerConfigsDir = path.join(bsroformerModelsDir, 'configs');
  return { bsroformerDir, bsroformerModelsDir, bsroformerWeightsDir, bsroformerConfigsDir };
}

/**
 * Ensure BS-RoFormer model directories exist
 * Note: Only creates model directories, not legacy executable directory (bsroformerDir)
 * DSU runtime is used instead of standalone executable
 * @param {Object} appDirs - The app directories from getAppDirs()
 */
function ensureBSRoformerDirs(appDirs) {
  const { bsroformerModelsDir, bsroformerWeightsDir, bsroformerConfigsDir } = getBSRoformerDirs(appDirs);
  // Only create models directories, not the legacy executable directory
  [bsroformerModelsDir, bsroformerWeightsDir, bsroformerConfigsDir].forEach(d => {
    if (!fs.existsSync(d)) {
      fs.mkdirSync(d, { recursive: true });
    }
  });
}

/**
 * Get path to BS-RoFormer executable
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {string} Path to executable
 */
function getBSRoformerPath(appDirs) {
  const { bsroformerDir } = getBSRoformerDirs(appDirs);
  // NOTE: current release archives extract `mss-separate` directly into `bsroformerDir`
  // (no `dist/` subfolder). Keep a small fallback for older layouts.
  const direct = os.platform() === 'win32'
    ? path.join(bsroformerDir, 'mss-separate.exe')
    : path.join(bsroformerDir, 'mss-separate');

  if (fs.existsSync(direct)) return direct;

  const legacy = os.platform() === 'win32'
    ? path.join(bsroformerDir, 'dist', 'mss-separate.exe')
    : path.join(bsroformerDir, 'dist', 'mss-separate');

  return legacy;
}

// ============================================================================
// MODEL DEFINITIONS
// ============================================================================

/**
 * All available BS-RoFormer models
 * Config files are bundled in the frozen executable
 * Only weights need to be downloaded (and custom configs for special models)
 */
const BSROFORMER_MODELS = {
  // === 4-STEM (bass/drums/vocals/other) ===
  // NOTE: htdemucs models removed - use Demucs executable for those
  'scnet_xl_ihf': {
    category: '4-stem',
    description: '4-stem BEST - SDR 10.08',
    speed: '~90s',
    quality: 'SDR 10.08',
    stems: ['drums', 'bass', 'other', 'vocals'],
    checkpoint: {
      url: 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt',
      filename: 'scnet_xl_ihf.ckpt',
      size: '204 MB'
    },
    // Config needs to be downloaded (not bundled yet)
    config: {
      url: 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/config_musdb18_scnet_xl_more_wide_v5.yaml',
      filename: 'config_scnet_xl_ihf.yaml'
    },
    modelType: 'scnet'
  },
  'bsroformer_4stem': {
    category: '4-stem',
    description: '4-stem HIGH QUALITY - SDR 9.65',
    speed: '~60s',
    quality: 'SDR 9.65',
    stems: ['bass', 'drums', 'vocals', 'other'],
    checkpoint: {
      url: 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt',
      filename: 'bsroformer_4stem.ckpt',
      size: '527 MB'
    },
    bundledConfig: 'configs/config_bs_roformer_4stem.yaml',
    modelType: 'bs_roformer'
  },

  // === 6-STEM (+ piano/guitar) ===
  // NOTE: htdemucs_6stem removed - use Demucs executable for that
  'logic_roformer': {
    category: '6-stem',
    description: '6-stem BEST BASS - 40x less bass bleed',
    speed: '~60s',
    quality: 'Excellent bass separation',
    stems: ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano'],
    checkpoint: {
      url: 'https://huggingface.co/ChenTechnology/logic_bsroformer/resolve/main/logic_roformer.pt',
      filename: 'logic_roformer.ckpt',
      size: '667 MB'
    },
    // Config is bundled in the executable
    bundledConfig: 'configs/config_logic_roformer.yaml',
    modelType: 'bs_roformer'
  },
  'bsrofo_sw': {
    category: '6-stem',
    description: '6-stem BEST - By jarredou',
    speed: '~48s',
    quality: 'High',
    stems: ['bass', 'drums', 'vocals', 'other', 'piano', 'guitar'],
    checkpoint: {
      url: 'https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt',
      filename: 'bsrofo_sw_fixed.ckpt',
      size: '667 MB'
    },
    bundledConfig: 'configs/config_bsrofo_sw_fixed.yaml',
    modelType: 'bs_roformer'
  },

  // === VOCALS ===
  'resurrection_vocals': {
    category: 'vocals',
    description: 'BS-RoFormer Resurrection - Vocals (pcunwa)',
    speed: '~155s',
    quality: 'High quality vocal extraction',
    stems: ['vocals'],  // Single stem (use --extract-instrumental for inst)
    checkpoint: {
      url: 'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection.ckpt',
      filename: 'resurrection_vocals.ckpt',
      size: '205 MB'
    },
    bundledConfig: 'configs/config_resurrection_vocals.yaml',
    modelType: 'bs_roformer'
  },
  'resurrection_inst': {
    category: 'instrumental',
    description: 'BS-RoFormer Resurrection - Instrumental',
    speed: '~120s',
    quality: 'High quality karaoke/instrumental',
    stems: ['other'],  // Single stem (other = instrumental)
    checkpoint: {
      url: 'https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.ckpt',
      filename: 'resurrection_inst.ckpt',
      size: '204 MB'
    },
    bundledConfig: 'configs/config_resurrection_inst.yaml',
    modelType: 'bs_roformer'
  },
  
  // === GABOX INSTRUMENTAL MODELS ===
  'gabox_inst_fv7z': {
    category: 'instrumental',
    description: 'Gabox MelBand Inst Fv7z - Fullness 29.96',
    speed: '~130s',
    quality: 'Fullness: 29.96, Bleedless: 44.61',
    stems: ['instrumental'],  // Single stem (lowercase)
    checkpoint: {
      url: 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv7z.ckpt',
      filename: 'gabox_inst_fv7z.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_gabox_inst.yaml',
    modelType: 'mel_band_roformer'
  },
  'gabox_inst_fv8': {
    category: 'instrumental',
    description: 'Gabox MelBand Inst Fv8 (v2)',
    speed: '~130s',
    quality: 'Updated version',
    stems: ['instrumental'],  // Single stem (lowercase)
    checkpoint: {
      url: 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/Inst_GaboxFv8.ckpt',
      filename: 'gabox_inst_fv8.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_gabox_inst.yaml',
    modelType: 'mel_band_roformer'
  },
  'gabox_inst_fv4': {
    category: 'instrumental',
    description: 'Gabox MelBand Inst Fv4 - Not muddy',
    speed: '~130s',
    quality: 'Clean instrumental, not muddy',
    stems: ['instrumental'],  // Single stem (lowercase)
    checkpoint: {
      url: 'https://huggingface.co/GaboxR67/MelBandRoformers/resolve/main/melbandroformers/instrumental/inst_Fv4.ckpt',
      filename: 'gabox_inst_fv4.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_gabox_inst.yaml',
    modelType: 'mel_band_roformer'
  },
  
  // === ANAME 4-STEM (BETTER DRUMS/BASS) ===
  'aname_4stem_large': {
    category: '4-stem',
    description: 'Aname 4-stem Large - BEST Drums 9.72',
    speed: '~180s',
    quality: 'Drums: 9.72, Bass: 9.40',
    stems: ['drums', 'bass', 'other', 'vocals'],
    checkpoint: {
      url: 'https://huggingface.co/Aname-Tommy/melbandroformer4stems/resolve/main/mel_band_roformer_4stems_large_ver1.ckpt',
      filename: 'aname_4stem_large.ckpt',
      size: '3.5 GB'
    },
    bundledConfig: 'configs/config_aname_4stem_large.yaml',
    modelType: 'mel_band_roformer'
  },
  
  // === REVIVE VOCAL MODELS (PCUNWA) ===
  'revive2_vocals': {
    category: 'vocals',
    description: 'BS-RoFormer Revive 2 - HIGHEST Bleedless 40.07',
    speed: '~120s',
    quality: 'Bleedless: 40.07, SDR: 10.97',
    stems: ['vocals'],  // Single stem (use --extract-instrumental for inst)
    checkpoint: {
      url: 'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive2.ckpt',
      filename: 'revive2_vocals.ckpt',
      size: '610 MB'
    },
    bundledConfig: 'configs/config_revive.yaml',
    modelType: 'bs_roformer'
  },
  'revive3e_vocals': {
    category: 'vocals',
    description: 'BS-RoFormer Revive 3e - Maximum Fullness',
    speed: '~120s',
    quality: 'Maximum fullness, preserves most audio',
    stems: ['vocals'],  // Single stem (use --extract-instrumental for inst)
    checkpoint: {
      url: 'https://huggingface.co/pcunwa/BS-Roformer-Revive/resolve/main/bs_roformer_revive3e.ckpt',
      filename: 'revive3e_vocals.ckpt',
      size: '610 MB'
    },
    bundledConfig: 'configs/config_revive.yaml',
    modelType: 'bs_roformer'
  },
  
  // === UNWA V1E+ INSTRUMENTAL ===
  'inst_v1e_plus': {
    category: 'instrumental',
    description: 'Unwa V1e+ Inst - Fullness 37.89, Less noise',
    speed: '~130s',
    quality: 'Fullness: 37.89, Bleedless: 36.53',
    stems: ['other'],  // Single stem (other = instrumental)
    checkpoint: {
      url: 'https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e_plus.ckpt',
      filename: 'inst_v1e_plus.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_inst_v1e_plus.yaml',
    modelType: 'mel_band_roformer'
  },
  
  // === KARAOKE MODELS (LEAD/BACKING SEPARATION) ===
  'karaoke_becruily': {
    category: 'karaoke',
    description: 'Becruily Karaoke - BEST vocals/instrumental separation',
    speed: '~180s',
    quality: 'Best harmony detection, best LV/BV differentiation',
    stems: ['vocals', 'instrumental'],
    checkpoint: {
      url: 'https://huggingface.co/becruily/mel-band-roformer-karaoke/resolve/main/mel_band_roformer_karaoke_becruily.ckpt',
      filename: 'karaoke_becruily.ckpt',
      size: '1.6 GB'
    },
    bundledConfig: 'configs/config_karaoke_becruily.yaml',
    modelType: 'mel_band_roformer'
  },
  'karaoke_aufr33': {
    category: 'karaoke',
    description: 'Aufr33/Viperx Karaoke - SDR 10.19',
    speed: '~130s',
    quality: 'SDR: 10.19',
    stems: ['karaoke'],  // Single stem (karaoke = instrumental)
    checkpoint: {
      url: 'https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model/resolve/main/mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
      filename: 'karaoke_aufr33.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_karaoke_aufr33.yaml',
    modelType: 'mel_band_roformer'
  },
  'vocals_melband': {
    category: 'vocals',
    description: 'Vocals BEST - SDR 10.98',
    speed: '~54s',
    quality: 'SDR 10.98',
    stems: ['vocals'],  // Single stem (use --extract-instrumental for inst)
    checkpoint: {
      url: 'https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt',
      filename: 'MelBandRoformer_kj.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml',
    modelType: 'mel_band_roformer'
  },
  'vocals_bsroformer_viperx': {
    category: 'vocals',
    description: 'Vocals HIGH QUALITY - SDR 10.87',
    speed: '~125s',
    quality: 'SDR 10.87',
    stems: ['vocals'],  // Single stem (use --extract-instrumental for inst)
    checkpoint: {
      url: 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt',
      filename: 'vocals_bsroformer_viperx.ckpt',
      size: '610 MB'
    },
    bundledConfig: 'configs/viperx/model_bs_roformer_ep_317_sdr_12.9755.yaml',
    modelType: 'bs_roformer'
  },
  'vocals_mdx23c': {
    category: 'vocals',
    description: 'Vocals FAST - SDR 10.17',
    speed: '~30s',
    quality: 'SDR 10.17',
    stems: ['vocals', 'other'],
    checkpoint: {
      url: 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt',
      filename: 'vocals_mdx23c.ckpt',
      size: '427 MB'
    },
    bundledConfig: 'configs/config_vocals_mdx23c.yaml',
    modelType: 'mdx23c'
  },

  // === AUDIO PROCESSING ===
  'denoise': {
    category: 'processing',
    description: 'Remove noise - SDR 27.99',
    speed: 'Medium',
    quality: 'SDR 27.99',
    stems: ['dry'],  // Single stem (dry = clean audio)
    checkpoint: {
      url: 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.7/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
      filename: 'denoise.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_denoise_mel_band_roformer.yaml',
    modelType: 'mel_band_roformer'
  },
  'dereverb': {
    category: 'processing',
    description: 'Remove reverb - SDR 19.17',
    speed: 'Medium',
    quality: 'SDR 19.17',
    stems: ['noreverb'],  // Single stem (noreverb = dry audio)
    checkpoint: {
      url: 'https://huggingface.co/anvuew/dereverb_mel_band_roformer/resolve/main/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
      filename: 'dereverb.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_dereverb_mel_band_roformer.yaml',
    modelType: 'mel_band_roformer'
  },

  // === DRUM KIT SEPARATION ===
  // NOTE: drumsep_htdemucs, drums_htdemucs, bass_htdemucs removed - use Demucs executable
  'drumsep_mdx23c_aufr33': {
    category: 'drumsep',
    description: 'Drum kit 6-stem (kick/snare/toms/hh/ride/crash)',
    speed: '~40s',
    quality: 'kick: 14.54, snare: 9.79',
    stems: ['kick', 'snare', 'toms', 'hh', 'ride', 'crash'],
    checkpoint: {
      url: 'https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt',
      filename: 'drumsep_mdx23c_aufr33.ckpt',
      size: '400 MB'
    },
    // Needs custom config (not bundled)
    config: {
      url: 'https://github.com/jarredou/models/releases/download/aufr33-jarredou_MDX23C_DrumSep_model_v0.1/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.yaml',
      filename: 'config_drumsep_mdx23c_aufr33.yaml'
    },
    modelType: 'mdx23c'
  },
  'drumsep_mdx23c_jarredou': {
    category: 'drumsep',
    description: 'Drum kit 5-stem BEST (kick/snare/toms/hh/cymbals)',
    speed: '~40s',
    quality: 'kick: 16.66, snare: 11.53',
    stems: ['kick', 'snare', 'toms', 'hh', 'cymbals'],
    checkpoint: {
      url: 'https://github.com/jarredou/models/releases/download/DrumSep/drumsep_5stems_mdx23c_jarredou.ckpt',
      filename: 'drumsep_mdx23c_jarredou.ckpt',
      size: '400 MB'
    },
    // Needs custom config (not bundled)
    config: {
      url: 'https://github.com/jarredou/models/releases/download/DrumSep/config_mdx23c.yaml',
      filename: 'config_drumsep_mdx23c_jarredou.yaml'
    },
    modelType: 'mdx23c'
  },

  // === SINGLE INSTRUMENT ISOLATION ===
  'guitar_becruily': {
    category: 'instrument',
    description: 'Guitar isolation - Community model by becruily',
    speed: '~37s',
    quality: 'SDR 14.22 (estimated)',
    stems: ['guitar'],  // Single stem output (lowercase)
    checkpoint: {
      url: 'https://huggingface.co/xavriley/source_separation_mirror/resolve/main/becruily_guitar.ckpt',
      filename: 'becruily_guitar.ckpt',
      size: '43 MB'
    },
    // Needs custom config
    config: {
      url: 'https://huggingface.co/xavriley/source_separation_mirror/resolve/main/config_guitar_becruily.yaml',
      filename: 'config_guitar_becruily.yaml'
    },
    modelType: 'mel_band_roformer'
  },

  // === SPECIAL PURPOSE MODELS (from python-audio-separator) ===
  'aspiration': {
    category: 'specialized',
    description: 'De-breathe vocals - SDR 18.98 (Sucial)',
    speed: '~60s',
    quality: 'SDR 18.98 - removes breath sounds',
    stems: ['aspiration', 'other'],  // Two stems (aspiration = breaths)
    checkpoint: {
      url: 'https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/aspiration_mel_band_roformer_sdr_18.9845.ckpt',
      filename: 'aspiration.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_aspiration_mel_band_roformer.yaml',
    modelType: 'mel_band_roformer'
  },

  'apollo_vocal_msst': {
    category: 'specialized',
    description: 'Vocal Restoration (baicai1145 - 194MB, inference optimized)',
    speed: '~45s',
    quality: 'Restores MP3-compressed vocals to high quality',
    stems: ['restored', 'addition'],
    checkpoint: {
      url: 'https://huggingface.co/baicai1145/Apollo-vocal-msst/resolve/main/model_apollo_vocals_ep_54.ckpt',
      filename: 'apollo_vocal_msst.ckpt',
      size: '194 MB'
    },
    config: {
      url: 'https://huggingface.co/baicai1145/Apollo-vocal-msst/resolve/main/config_apollo_vocals_ep_54.yaml',
      filename: 'config_apollo_vocal_msst.yaml'
    },
    modelType: 'apollo'
  },

  'apollo_vocal_msst_full': {
    category: 'specialized',
    description: 'Vocal Restoration (baicai1145 - 616MB, full checkpoint)',
    speed: '~60s',
    quality: 'Full training checkpoint - maximum quality vocal restoration',
    stems: ['restored', 'addition'],
    checkpoint: {
      url: 'https://huggingface.co/baicai1145/Apollo-vocal-msst/resolve/main/epoch=54-val_loss=-17.6221.ckpt',
      filename: 'apollo_vocal_msst_full.ckpt',
      size: '616 MB'
    },
    config: {
      url: 'https://huggingface.co/baicai1145/Apollo-vocal-msst/resolve/main/config_apollo_vocals_ep_54.yaml',
      filename: 'config_apollo_vocal_msst.yaml'
    },
    modelType: 'apollo'
  },

  'apollo_official_msst': {
    category: 'specialized',
    description: 'Official JusperLee Apollo - General audio restoration',
    speed: '~45s',
    quality: 'Original Apollo model for lossy audio restoration',
    stems: ['restored', 'addition'],
    checkpoint: {
      url: 'https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin',
      filename: 'apollo_official.bin',
      size: '63 MB'
    },
    config: {
      url: 'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/config_apollo.yaml',
      filename: 'config_apollo_official.yaml'
    },
    modelType: 'apollo'
  },
  'chorus_male_female': {
    category: 'specialized',
    description: 'Separate male/female vocals in duets - SDR 24.12 (Sucial)',
    speed: '~90s',
    quality: 'SDR 24.12 - separates male/female in duets',
    stems: ['male', 'female'],
    checkpoint: {
      url: 'https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt',
      filename: 'chorus_male_female.ckpt',
      size: '610 MB'
    },
    bundledConfig: 'configs/config_chorus_male_female_bs_roformer.yaml',
    modelType: 'bs_roformer'
  },
  'bleed_suppressor': {
    category: 'processing',
    description: 'Remove vocal bleed from instrumentals (unwa-97chris)',
    speed: '~60s',
    quality: 'Removes vocal artifacts from inst stems',
    stems: ['instrumental'],  // Single stem (lowercase)
    checkpoint: {
      url: 'https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/mel_band_roformer_bleed_suppressor_v1.ckpt',
      filename: 'bleed_suppressor.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_mel_band_roformer_bleed_suppressor_v1.yaml',
    modelType: 'mel_band_roformer'
  },
  'crowd': {
    category: 'specialized',
    description: 'Extract crowd/audience noise - SDR 8.71 (aufr33-viperx)',
    speed: '~60s',
    quality: 'SDR 8.71 - extracts audience/crowd sounds',
    stems: ['crowd'],  // Single stem
    checkpoint: {
      url: 'https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
      filename: 'crowd.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_crowd.yaml',
    modelType: 'mel_band_roformer'
  },
  'denoise_debleed': {
    category: 'processing',
    description: 'Remove noise and bleed from instrumentals (Gabox)',
    speed: '~60s',
    quality: 'Cleans fullness model artifacts',
    stems: ['instrumental'],  // Single stem (lowercase)
    checkpoint: {
      url: 'https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/mel_band_roformer_denoise_debleed_gabox.ckpt',
      filename: 'denoise_debleed.ckpt',
      size: '871 MB'
    },
    bundledConfig: 'configs/config_mel_band_roformer_instrumental_gabox.yaml',
    modelType: 'mel_band_roformer'
  }
};

// ============================================================================
// RELEASE URLS
// ============================================================================

const BSROFORMER_RELEASES = {
  // v1.11.0: Windows upgrade to PyTorch 2.10.0, CUDA 12.6
  // - Fixed llvmlite.dll bundling for Windows
  // - Comprehensive torch subpackages for reliable freezing
  // v1.10.0: PyTorch 2.5 + MPS optimization (~15% faster on Apple Silicon)
  // - STFT/ISTFT now run on GPU natively (was CPU fallback)
  // - Only scatter_add_ still requires CPU (PyTorch limitation)
  // v1.8.4: Fixed portable build (bad magic number error), all v1.8.3 features included
  macos_intel: 'https://github.com/ostinsolo/BS-RoFormer-freeze/releases/download/v1.8.4/mss-separate-macos-intel-v1.8.4.tar.gz',
  macos_arm: 'https://github.com/ostinsolo/BS-RoFormer-freeze/releases/download/v1.11.0/mss-separate-macos-arm.tar.gz',
  windows: 'https://github.com/ostinsolo/BS-RoFormer-freeze/releases/download/v1.11.1-win/mss-separate-win-cpu.zip',
  windows_cuda: 'https://github.com/ostinsolo/BS-RoFormer-freeze/releases/download/v1.11.1-win/mss-separate-win-cuda.7z'
};

// ============================================================================
// EXTRACTION NORMALIZATION (Windows archives may contain nested parent folders)
// ============================================================================

function normalizeBSRoformerLayout(appDirs, Max) {
  const { bsroformerDir } = getBSRoformerDirs(appDirs);
  const exeName = os.platform() === 'win32' ? 'mss-separate.exe' : 'mss-separate';

  const directExe = path.join(bsroformerDir, exeName);
  if (fs.existsSync(directExe)) return true;

  // 1) Flatten dist/ if present (legacy)
  const distDir = path.join(bsroformerDir, 'dist');
  if (fs.existsSync(distDir) && fs.statSync(distDir).isDirectory()) {
    try {
      const items = fs.readdirSync(distDir);
      for (const item of items) {
        const src = path.join(distDir, item);
        const dst = path.join(bsroformerDir, item);
        if (!fs.existsSync(dst)) fs.renameSync(src, dst);
      }
      try { fs.rmdirSync(distDir); } catch (e) {}
    } catch (e) {}
    if (fs.existsSync(directExe)) return true;
  }

  // 2) Search for the executable inside nested folders
  let foundExe = null;
  const maxNodes = 8000;
  let seen = 0;
  const queue = [bsroformerDir];

  while (queue.length && seen < maxNodes && !foundExe) {
    const dir = queue.shift();
    let entries = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch (e) {
      continue;
    }
    for (const ent of entries) {
      seen++;
      if (seen >= maxNodes) break;
      const full = path.join(dir, ent.name);
      if (ent.isFile() && ent.name.toLowerCase() === exeName.toLowerCase()) {
        foundExe = full;
        break;
      }
      if (ent.isDirectory()) {
        // Safety skip (these are our model dirs, not exe bundle)
        if (ent.name === 'weights' || ent.name === 'configs') continue;
        queue.push(full);
      }
    }
  }

  if (!foundExe) {
    try { Max.post('[BSRoformer] Warning: executable not found after extraction'); } catch (e) {}
    return false;
  }

  // Pick a reasonable bundle root to move from
  let extractedRoot = path.dirname(foundExe);
  if (path.basename(extractedRoot).toLowerCase() === 'dist') {
    extractedRoot = path.dirname(extractedRoot);
  }

  try { Max.post('[BSRoformer] Normalizing extracted layout from: ' + extractedRoot); } catch (e) {}

  // Move everything from extractedRoot into bsroformerDir
  try {
    const items = fs.readdirSync(extractedRoot);
    for (const item of items) {
      const src = path.join(extractedRoot, item);
      const dst = path.join(bsroformerDir, item);
      if (fs.existsSync(dst)) continue;
      try {
        fs.renameSync(src, dst);
      } catch (e) {
        // Fallback copy for cross-device or locked files
        try {
          const st = fs.statSync(src);
          if (st.isDirectory()) {
            fs.mkdirSync(dst, { recursive: true });
            const stack = [{ s: src, d: dst }];
            while (stack.length) {
              const { s, d } = stack.pop();
              const ents = fs.readdirSync(s, { withFileTypes: true });
              for (const en of ents) {
                const ss = path.join(s, en.name);
                const dd = path.join(d, en.name);
                if (en.isDirectory()) {
                  if (!fs.existsSync(dd)) fs.mkdirSync(dd, { recursive: true });
                  stack.push({ s: ss, d: dd });
                } else if (en.isFile()) {
                  if (!fs.existsSync(dd)) fs.copyFileSync(ss, dd);
                }
              }
            }
          } else if (st.isFile()) {
            fs.copyFileSync(src, dst);
          }
        } catch (_) {}
      }
    }
  } catch (_) {}

  // Best-effort cleanup of empty parent folders up to bsroformerDir
  try {
    let cur = extractedRoot;
    while (cur && cur.startsWith(bsroformerDir) && cur !== bsroformerDir) {
      try { fs.rmdirSync(cur); } catch (e) { break; }
      cur = path.dirname(cur);
    }
  } catch (_) {}

  return fs.existsSync(directExe);
}

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

/**
 * Verify BS-RoFormer installation
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 * @param {Function} callback - Callback(isInstalled)
 */
function verifyBSRoformer(appDirs, Max, callback) {
  const { bsroformerWeightsDir } = getBSRoformerDirs(appDirs);
  const exe = getBSRoformerPath(appDirs);
  
  const exeExists = fs.existsSync(exe);
  
  const installedModels = Object.entries(BSROFORMER_MODELS).filter(([_, m]) => {
    const checkpointPath = path.join(bsroformerWeightsDir, m.checkpoint.filename);
    return isCheckpointValid(checkpointPath, m, Max);
  }).map(([name]) => name);
  
  if (exeExists && installedModels.length > 0) {
    Max.post(`BS-RoFormer verified (${installedModels.length} model(s))`);
    callback(true);
    return;
  }
  
  if (!exeExists) Max.post('BS-RoFormer executable not found');
  if (installedModels.length === 0) Max.post('No BS-RoFormer models installed');
  
  callback(false);
}

/**
 * Get list of installed models
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {Array} List of installed model names
 */
function getInstalledModels(appDirs) {
  const { bsroformerWeightsDir } = getBSRoformerDirs(appDirs);
  return Object.entries(BSROFORMER_MODELS)
    .filter(([_, m]) => {
      const checkpointPath = path.join(bsroformerWeightsDir, m.checkpoint.filename);
      return isCheckpointValid(checkpointPath, m, null);
    })
    .map(([name]) => name);
}

/**
 * Get list of missing models (based on weights only)
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {Array} List of missing model names
 */
function getMissingModels(appDirs) {
  const installed = new Set(getInstalledModels(appDirs));
  return Object.keys(BSROFORMER_MODELS).filter(name => !installed.has(name));
}

// ============================================================================
// DOWNLOAD FUNCTIONS
// ============================================================================

/**
 * Download BS-RoFormer executable
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 * @param {Function} downloadToFile - Download function from setupSW+.js
 * @param {Function} extractArchive - Extract function from setupSW+.js
 * @param {Function|Object} callbackOrOptions - Callback(error) or options object
 * @param {Function} [callback] - Callback(error) if options provided as 5th arg
 * 
 * Options:
 * - preferCuda {boolean} - If true on Windows, download CUDA build (default: false)
 */
function downloadBSRoformer(appDirs, Max, downloadToFile, extractArchive, callbackOrOptions, callback) {
  // Handle both signatures: (appDirs, Max, download, extract, callback) and (appDirs, Max, download, extract, options, callback)
  let options = {};
  let cb = callbackOrOptions;
  
  if (typeof callbackOrOptions === 'object' && typeof callback === 'function') {
    options = callbackOrOptions || {};
    cb = callback;
  }
  const { bsroformerDir } = getBSRoformerDirs(appDirs);
  ensureBSRoformerDirs(appDirs);

  // Skip download if already present
  const exe = getBSRoformerPath(appDirs);
  if (fs.existsSync(exe)) {
    Max.post('✅ BS-RoFormer executable already installed');
    cb(null);
    return;
  }
  
  // Prevent concurrent downloads - queue callbacks if download in progress
  if (bsroformerDownloadInProgress) {
    Max.post('BS-RoFormer download already in progress, queuing...');
    bsroformerDownloadCallbacks.push(cb);
    return;
  }
  
  // Mark download as in progress
  bsroformerDownloadInProgress = true;
  
  // Helper to complete download and notify all queued callbacks
  const completeDownload = (error) => {
    bsroformerDownloadInProgress = false;
    cb(error);
    // Notify all queued callbacks
    const queued = bsroformerDownloadCallbacks.splice(0);
    queued.forEach(queuedCb => queuedCb(error));
  };
  
  Max.post('==================================================');
  Max.post('DOWNLOADING BS-ROFORMER');
  Max.post('==================================================');
  Max.post('State-of-the-art music source separation');
  Max.post('');
  
  const isWin = os.platform() === 'win32';
  const isMac = os.platform() === 'darwin';
  
  let releaseUrl = null;
  let archiveExt = '.tar.gz';
  
  if (isMac) {
    // Detect Apple Silicon vs Intel
    const isArm = os.arch() === 'arm64';
    releaseUrl = isArm ? BSROFORMER_RELEASES.macos_arm : BSROFORMER_RELEASES.macos_intel;
    Max.post(`Detected macOS ${isArm ? 'ARM (Apple Silicon)' : 'Intel'}`);
  } else if (isWin) {
    // Use CUDA build if preferCuda option is set (detected by caller via detectNvidia)
    if (options.preferCuda) {
      Max.post('Detected NVIDIA GPU with CUDA support');
      Max.post('Downloading CUDA-enabled build (larger but GPU-accelerated)');
      releaseUrl = BSROFORMER_RELEASES.windows_cuda;
      archiveExt = '.7z';
    } else {
      Max.post('No NVIDIA GPU detected - using CPU build');
      Max.post('(Install NVIDIA drivers to enable GPU acceleration)');
      releaseUrl = BSROFORMER_RELEASES.windows;
      archiveExt = '.zip';
    }
  }
  
  if (!releaseUrl) {
    Max.post('BS-RoFormer not available for this platform');
    Max.post('Supported: macOS Intel, macOS ARM, Windows CPU, Windows CUDA');
    completeDownload(new Error('Platform not supported'));
    return;
  }
  
  const archivePath = path.join(bsroformerDir, 'mss-separate' + archiveExt);
  
  // Check for leftover archive from interrupted install
  if (fs.existsSync(archivePath)) {
    Max.post('Found existing BS-RoFormer archive - extracting...');
    extractArchive(archivePath, bsroformerDir, (exErr) => {
      if (exErr) {
        Max.post('Extraction failed: ' + exErr.message + ' - will re-download');
        try { fs.unlinkSync(archivePath); } catch (e) {}
        // Fall through to download
        startDownload();
        return;
      }
      
      try { fs.unlinkSync(archivePath); } catch (e) {}
      
      // Some archives extract to a 'dist/' subfolder - flatten it
      const distDir = path.join(bsroformerDir, 'dist');
      if (fs.existsSync(distDir) && fs.statSync(distDir).isDirectory()) {
        Max.post('Flattening dist/ subfolder...');
        try {
          const items = fs.readdirSync(distDir);
          for (const item of items) {
            const src = path.join(distDir, item);
            const dst = path.join(bsroformerDir, item);
            if (!fs.existsSync(dst)) {
              fs.renameSync(src, dst);
            }
          }
          try { fs.rmdirSync(distDir); } catch (e) {}
        } catch (flattenErr) {
          Max.post('Warning: Could not flatten dist/: ' + flattenErr.message);
        }
      }

      // Normalize nested parent folders (some .7z/.zip releases wrap content in a top-level folder)
      normalizeBSRoformerLayout(appDirs, Max);
      const exeAfter = getBSRoformerPath(appDirs);
      
      // Make executable and remove quarantine
      if (!isWin) {
        const { exec } = require('child_process');
        exec(`chmod +x "${exeAfter}" && xattr -dr com.apple.quarantine "${bsroformerDir}" 2>/dev/null || true`, () => {
          if (fs.existsSync(exeAfter)) {
            Max.post('✅ BS-RoFormer extracted from existing archive');
            completeDownload(null);
          } else {
            Max.post('Extraction incomplete - will re-download');
            startDownload();
          }
        });
      } else {
        if (fs.existsSync(exeAfter)) {
          Max.post('✅ BS-RoFormer extracted from existing archive');
          completeDownload(null);
        } else {
          Max.post('Extraction incomplete - will re-download');
          startDownload();
        }
      }
    });
    return;
  }
  
  startDownload();
  
  function startDownload() {
    Max.post('Downloading BS-RoFormer executable...');
    Max.post('This may take several minutes...');
    
    downloadToFile(releaseUrl, archivePath, (err1) => {
    if (err1) {
      Max.post('Download failed: ' + err1.message);
      completeDownload(err1);
      return;
    }
    
    // Verify the download actually succeeded
    if (!fs.existsSync(archivePath)) {
      Max.post('Download completed but file not found: ' + archivePath);
      completeDownload(new Error('Download failed - file not created'));
      return;
    }
    
    const fileSize = fs.statSync(archivePath).size;
    if (fileSize < 1000) {
      Max.post('Download completed but file is too small (' + fileSize + ' bytes): ' + archivePath);
      try { fs.unlinkSync(archivePath); } catch (e) {}
      completeDownload(new Error('Download failed - file too small, likely an error page'));
      return;
    }
    
    Max.post('Download verified (' + Math.round(fileSize / 1024 / 1024) + ' MB), extracting...');
    extractArchive(archivePath, bsroformerDir, (err2) => {
      try { fs.unlinkSync(archivePath); } catch (e) {}
      
      if (err2) {
        Max.post('Extraction failed: ' + err2.message);
        completeDownload(err2);
        return;
      }
      
      // Some archives extract to a 'dist/' subfolder - flatten it
      const distDir = path.join(bsroformerDir, 'dist');
      if (fs.existsSync(distDir) && fs.statSync(distDir).isDirectory()) {
        Max.post('Flattening dist/ subfolder...');
        try {
          const items = fs.readdirSync(distDir);
          for (const item of items) {
            const src = path.join(distDir, item);
            const dst = path.join(bsroformerDir, item);
            if (!fs.existsSync(dst)) {
              fs.renameSync(src, dst);
            }
          }
          // Remove empty dist folder
          try { fs.rmdirSync(distDir); } catch (e) {}
        } catch (flattenErr) {
          Max.post('Warning: Could not flatten dist/: ' + flattenErr.message);
        }
      }

      // Normalize nested parent folders
      normalizeBSRoformerLayout(appDirs, Max);
      
      // Make executable on Unix and remove quarantine attribute (macOS Gatekeeper)
      const exe = getBSRoformerPath(appDirs);
      if (os.platform() !== 'win32') {
        // Remove quarantine from entire bsroformer directory (has many bundled files)
        const { bsroformerDir } = getBSRoformerDirs(appDirs);
        exec(`chmod +x "${exe}" && xattr -dr com.apple.quarantine "${bsroformerDir}" 2>/dev/null || true`, () => {
          Max.post('✅ BS-RoFormer executable installed');
          completeDownload(null);
        });
      } else {
        Max.post('✅ BS-RoFormer executable installed');
        completeDownload(null);
      }
    });
  });
  }  // end startDownload
}

/**
 * Download a specific BS-RoFormer model
 * @param {string} modelName - Name of the model to download
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 * @param {Function} downloadToFile - Download function from setupSW+.js
 * @param {Function} callback - Callback(error)
 */
function downloadBSRoformerModel(modelName, appDirs, Max, downloadToFile, callback) {
  const { bsroformerWeightsDir, bsroformerConfigsDir } = getBSRoformerDirs(appDirs);
  const MODEL_ALIASES = {
    // Backward compatibility for older IDs
    apollo_official: 'apollo_official_msst'
  };

  const canonicalName = MODEL_ALIASES[modelName] || modelName;
  const model = BSROFORMER_MODELS[canonicalName];
  
  if (!model) {
    Max.post('Unknown model: ' + modelName);
    callback(new Error('Unknown model: ' + modelName));
    return;
  }
  
  ensureBSRoformerDirs(appDirs);
  
  Max.post(`Downloading BS-RoFormer model: ${canonicalName}`);
  Max.post(`  ${model.description}`);
  Max.post(`  Size (declared): ${model.checkpoint.size}`);
  
  const checkpointPath = path.join(bsroformerWeightsDir, model.checkpoint.filename);
  
  // Skip if already exists
  if (fs.existsSync(checkpointPath)) {
    if (!isCheckpointValid(checkpointPath, model, Max)) {
      Max.post('  Deleting and re-downloading...');
      try { fs.unlinkSync(checkpointPath); } catch (e) { Max.post('  Could not delete checkpoint: ' + e.message); }
    }
  }

  // Skip if already exists (after corruption check)
  if (fs.existsSync(checkpointPath)) {
    Max.post(`  Model already exists: ${model.checkpoint.filename}`);
    if (canonicalName === 'apollo_vocal_msst_full') {
      Max.post('  Converting full checkpoint for inference compatibility...');
      convertApolloFullCheckpoint(checkpointPath, Max, (convErr) => {
        if (convErr) {
          Max.post('  ⚠ Apollo full conversion failed. Model may not load.');
          Max.post('    ' + convErr.message);
        }
        updateModelsRegistry(canonicalName, appDirs);
        callback(null);
      });
      return;
    }
    updateModelsRegistry(canonicalName, appDirs);
    callback(null);
    return;
  }
  
  Max.post('  Downloading checkpoint...');
  downloadToFile(model.checkpoint.url, checkpointPath, (err1) => {
    if (err1) {
      Max.post('  Download failed: ' + err1.message);
      callback(err1);
      return;
    }
    const localSize = fs.existsSync(checkpointPath) ? fs.statSync(checkpointPath).size : null;
    fetchRemoteSize(model.checkpoint.url, Max, (remoteSize) => {
      if (remoteSize) {
        const cache = readSizeCache(appDirs);
        cache[canonicalName] = remoteSize;
        writeSizeCache(appDirs, cache);
      }
      if (remoteSize) {
        Max.post(`  Size (server): ${formatBytes(remoteSize)}`);
      }
      if (remoteSize && localSize !== null) {
        if (localSize !== remoteSize) {
          Max.post(`  ⚠ Size mismatch for ${path.basename(checkpointPath)}: expected ${remoteSize} bytes, got ${localSize} bytes`);
          try { fs.unlinkSync(checkpointPath); } catch (e) {}
          callback(new Error('Checkpoint size mismatch after download.'));
          return;
        }
      } else if (model.checkpoint && model.checkpoint.size) {
        if (!validateCheckpointSize(checkpointPath, model.checkpoint.size, Max)) {
          try { fs.unlinkSync(checkpointPath); } catch (e) {}
          callback(new Error('Checkpoint size mismatch after download.'));
          return;
        }
      }

      const finalizeInstall = () => {
        updateModelsRegistry(canonicalName, appDirs);
        Max.post(`  ✅ Model installed: ${canonicalName}`);
        callback(null);
      };

      const maybeConvertApolloFull = (next) => {
        if (canonicalName !== 'apollo_vocal_msst_full') {
          next();
          return;
        }
        Max.post('  Converting full checkpoint for inference compatibility...');
        convertApolloFullCheckpoint(checkpointPath, Max, (convErr) => {
          if (convErr) {
            Max.post('  ⚠ Apollo full conversion failed. Model may not load.');
            Max.post('    ' + convErr.message);
          }
          next();
        });
      };

      const afterConfig = () => {
        maybeConvertApolloFull(finalizeInstall);
      };

      // Download custom config if needed (not bundled)
      if (model.config && model.config.url) {
        const configPath = path.join(bsroformerConfigsDir, model.config.filename);
        Max.post('  Downloading config...');
        downloadToFile(model.config.url, configPath, (err2) => {
          if (err2) {
            Max.post('  Config download failed (will use bundled): ' + err2.message);
          }
          afterConfig();
        });
      } else {
        afterConfig();
      }
    });
  });
}

/**
 * Convert Apollo full training checkpoint to inference-compatible format.
 * Strips "audio_model." prefix and removes discriminator keys.
 * @param {string} checkpointPath - Path to .ckpt file
 * @param {Function} Max - Max API object
 * @param {Function} callback - Callback(error)
 */
function convertApolloFullCheckpoint(checkpointPath, Max, callback) {
  const python = process.env.PYTHON || 'python3';
  const tempPath = checkpointPath + '.tmp';
  const script = [
    'import sys',
    'import torch',
    'ckpt = torch.load(sys.argv[1], map_location="cpu")',
    'state = ckpt.get("state_dict", ckpt)',
    'new_state = {}',
    'for k, v in state.items():',
    '    if k.startswith("audio_model."):',
    '        new_state[k[len("audio_model."):]] = v',
    'if not new_state:',
    '    for k, v in state.items():',
    '        if k.startswith("model."):',
    '            new_state[k[len("model."):]] = v',
    'if not new_state:',
    '    new_state = state',
    'torch.save(new_state, sys.argv[2])',
    'print("converted", len(state), "->", len(new_state))'
  ].join('\n');

  const child = spawn(python, ['-c', script, checkpointPath, tempPath], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', (data) => { stderr += data.toString(); });
  child.on('close', (code) => {
    if (code !== 0) {
      callback(new Error(stderr || `python exited with code ${code}`));
      return;
    }
    try {
      if (fs.existsSync(tempPath)) {
        fs.renameSync(tempPath, checkpointPath);
      }
      callback(null);
    } catch (err) {
      callback(err);
    }
  });
}

// ============================================================================
// LIST FUNCTIONS
// ============================================================================

/**
 * List all available BS-RoFormer models
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 */
function listBSRoformerModels(appDirs, Max) {
  const { bsroformerWeightsDir } = getBSRoformerDirs(appDirs);
  
  const categories = {
    '4-stem': '4-STEM (bass/drums/vocals/other)',
    '6-stem': '6-STEM (+ piano/guitar)',
    'vocals': 'VOCAL EXTRACTION',
    'instrumental': 'INSTRUMENTAL EXTRACTION',
    'karaoke': 'KARAOKE (lead/backing separation)',
    'processing': 'AUDIO PROCESSING',
    'specialized': 'SPECIALIZED (de-breath, duets, crowd)',
    'drumsep': 'DRUM KIT SEPARATION',
    'instrument': 'SINGLE INSTRUMENT ISOLATION'
  };
  
  Max.post('');
  Max.post('BS-ROFORMER MODELS');
  Max.post('==================');
  
  for (const [cat, title] of Object.entries(categories)) {
    const catModels = Object.entries(BSROFORMER_MODELS).filter(([_, m]) => m.category === cat);
    if (catModels.length === 0) continue;
    
    Max.post('');
    Max.post(`${title}:`);
    
    for (const [name, info] of catModels) {
      const installed = fs.existsSync(path.join(bsroformerWeightsDir, info.checkpoint.filename));
      Max.post(`  ${installed ? '✅' : '❌'} ${name}: ${info.description}`);
    }
  }
  
  Max.post('');
}

function getModelSizeMap(appDirs) {
  const cache = readSizeCache(appDirs);
  const sizeMap = {};
  const { bsroformerWeightsDir } = getBSRoformerDirs(appDirs);
  for (const [modelName, model] of Object.entries(BSROFORMER_MODELS)) {
    let bytes = cache[modelName];
    if (!bytes && model.checkpoint && model.checkpoint.filename) {
      const localPath = path.join(bsroformerWeightsDir, model.checkpoint.filename);
      if (fs.existsSync(localPath)) {
        try {
          bytes = fs.statSync(localPath).size;
          cache[modelName] = bytes;
        } catch (e) {}
      }
    }
    if (bytes) {
      sizeMap[modelName] = formatBytesShort(bytes) || model.checkpoint.size;
    } else if (model.checkpoint && model.checkpoint.size) {
      sizeMap[modelName] = model.checkpoint.size.replace(/\s+/g, '');
    }
  }
  writeSizeCache(appDirs, cache);
  return sizeMap;
}

/**
 * Get model info
 * @param {string} modelName - Name of the model
 * @returns {Object|null} Model info or null if not found
 */
function getModelInfo(modelName) {
  return BSROFORMER_MODELS[modelName] || null;
}

/**
 * Get all model names
 * @returns {Array} List of all model names
 */
function getAllModelNames() {
  return Object.keys(BSROFORMER_MODELS);
}

/**
 * Get models by category
 * @param {string} category - Category name
 * @returns {Array} List of model names in that category
 */
function getModelsByCategory(category) {
  return Object.entries(BSROFORMER_MODELS)
    .filter(([_, m]) => m.category === category)
    .map(([name]) => name);
}

// ============================================================================
// MODELS.JSON REGISTRY MANAGEMENT
// ============================================================================

/**
 * Get path to models.json registry file
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {string} Path to models.json
 */
function getModelsJsonPath(appDirs) {
  const { bsroformerModelsDir } = getBSRoformerDirs(appDirs);
  return path.join(bsroformerModelsDir, 'models.json');
}

/**
 * Load models.json registry (or return empty object)
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {Object} Models registry
 */
function loadModelsRegistry(appDirs) {
  const jsonPath = getModelsJsonPath(appDirs);
  
  if (fs.existsSync(jsonPath)) {
    try {
      return JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    } catch (e) {
      console.error('Could not parse models.json:', e.message);
    }
  }
  return {};
}

/**
 * Save models.json registry
 * @param {Object} models - Models registry object
 * @param {Object} appDirs - The app directories from getAppDirs()
 */
function saveModelsRegistry(models, appDirs) {
  const jsonPath = getModelsJsonPath(appDirs);
  ensureBSRoformerDirs(appDirs);
  
  try {
    fs.writeFileSync(jsonPath, JSON.stringify(models, null, 2), 'utf-8');
  } catch (e) {
    console.error('Could not save models.json:', e.message);
  }
}

/**
 * Update models.json when a model is downloaded
 * Automatically adds the model entry so the executable can find it
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - The app directories from getAppDirs()
 */
function updateModelsRegistry(modelName, appDirs) {
  const model = BSROFORMER_MODELS[modelName];
  if (!model) return;
  
  const registry = loadModelsRegistry(appDirs);
  
  // Build the config path
  let configPath = model.bundledConfig;
  if (model.config && model.config.filename) {
    // Custom config was downloaded
    configPath = `configs/${model.config.filename}`;
  }
  
  // Add/update model entry
  registry[modelName] = {
    type: model.modelType,
    config: configPath,
    checkpoint: `weights/${model.checkpoint.filename}`,
    stems: model.stems,
    description: model.description
  };
  
  saveModelsRegistry(registry, appDirs);
}

/**
 * Add a custom user-trained model to the registry
 * @param {Object} modelDef - Model definition
 * @param {string} modelDef.name - Unique model name
 * @param {string} modelDef.type - Model type (bs_roformer, mel_band_roformer, htdemucs, mdx23c)
 * @param {string} modelDef.configPath - Path to config file (relative to models dir)
 * @param {string} modelDef.checkpointPath - Path to checkpoint (relative to models dir)
 * @param {Array<string>} modelDef.stems - Output stem names
 * @param {string} [modelDef.description] - Optional description
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 */
function addCustomModel(modelDef, appDirs, Max) {
  const { name, type, configPath, checkpointPath, stems, description } = modelDef;
  
  if (!name || !type || !configPath || !checkpointPath || !stems) {
    Max.post('Error: Missing required fields (name, type, configPath, checkpointPath, stems)');
    return false;
  }
  
  const validTypes = ['bs_roformer', 'mel_band_roformer', 'htdemucs', 'mdx23c'];
  if (!validTypes.includes(type)) {
    Max.post(`Error: Invalid type "${type}". Must be one of: ${validTypes.join(', ')}`);
    return false;
  }
  
  const registry = loadModelsRegistry(appDirs);
  
  registry[name] = {
    type: type,
    config: configPath,
    checkpoint: checkpointPath,
    stems: stems,
    description: description || 'Custom model'
  };
  
  saveModelsRegistry(registry, appDirs);
  Max.post(`✅ Added custom model: ${name}`);
  Max.post(`   Type: ${type}`);
  Max.post(`   Stems: ${stems.join(', ')}`);
  
  return true;
}

/**
 * Remove a model from the registry
 * @param {string} modelName - Name of the model to remove
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 */
function removeCustomModel(modelName, appDirs, Max) {
  const registry = loadModelsRegistry(appDirs);
  
  if (!registry[modelName]) {
    Max.post(`Model not found in registry: ${modelName}`);
    return false;
  }
  
  delete registry[modelName];
  saveModelsRegistry(registry, appDirs);
  Max.post(`✅ Removed model from registry: ${modelName}`);
  
  return true;
}

/**
 * List all models in the registry (including custom ones)
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 */
function listRegistryModels(appDirs, Max) {
  const registry = loadModelsRegistry(appDirs);
  const { bsroformerModelsDir } = getBSRoformerDirs(appDirs);
  
  Max.post('');
  Max.post('MODELS REGISTRY (models.json)');
  Max.post('==============================');
  
  if (Object.keys(registry).length === 0) {
    Max.post('  (empty - download a model to populate)');
    Max.post('');
    return;
  }
  
  for (const [name, info] of Object.entries(registry)) {
    const checkpointPath = path.join(bsroformerModelsDir, info.checkpoint);
    const installed = fs.existsSync(checkpointPath);
    const isBuiltIn = BSROFORMER_MODELS[name] ? '' : ' [CUSTOM]';
    
    Max.post(`  ${installed ? '✅' : '❌'} ${name}${isBuiltIn}`);
    Max.post(`      Type: ${info.type}`);
    Max.post(`      Stems: ${info.stems.join(', ')}`);
    if (info.description) Max.post(`      ${info.description}`);
  }
  
  Max.post('');
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Directory functions
  getBSRoformerDirs,
  ensureBSRoformerDirs,
  getBSRoformerPath,
  normalizeBSRoformerLayout,
  
  // Model data
  BSROFORMER_MODELS,
  BSROFORMER_RELEASES,
  
  // Verification
  verifyBSRoformer,
  getInstalledModels,
  getMissingModels,
  
  // Download
  downloadBSRoformer,
  downloadBSRoformerModel,
  getModelSizeMap,
  
  // Listing
  listBSRoformerModels,
  getModelInfo,
  getAllModelNames,
  getModelsByCategory,
  
  // Registry management (models.json)
  getModelsJsonPath,
  loadModelsRegistry,
  saveModelsRegistry,
  updateModelsRegistry,
  addCustomModel,
  removeCustomModel,
  listRegistryModels
};
