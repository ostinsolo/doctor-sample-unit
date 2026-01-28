/**
 * audio_separator_models.js - Audio Separator (VR) Model Definitions for SplitWizard
 * 
 * VR Architecture models for wind instrument and other separations.
 * Import this module into setupSW+.js for Audio Separator support.
 * 
 * Models included:
 * - 17_HP-Wind_Inst-UVR (Wind instrument separation)
 * - 2_HP-UVR (Vocals/Instrumental)
 * - 3_HP-Vocal-UVR (Vocals extraction)
 * 
 * Based on: https://github.com/Anjok07/ultimatevocalremovergui
 * Binary from: https://github.com/ostinsolo/audio-separator-cxfreeze
 * Current releases:
 * - macOS ARM: v1.2.4 (audio-separator-mac-arm.zip)
 * - Windows: v1.3.5-win (CPU/CUDA/DirectML)
 * 
 * Created by Ostin Solo
 * Website: ostinsolo.co.uk
 * 
 * @version 1.0.0
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const { isFileSizeMatch } = require('./model_integrity');

// ============================================================================
// DIRECTORY CONFIGURATION
// ============================================================================

/**
 * Get Audio Separator directories
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {Object} Audio Separator specific directories
 * 
 * NOTE: audioSeparatorDir and audioSeparatorBinaryDir are LEGACY - 
 * were used for audio-separator-cxfreeze executable.
 * DSU shared runtime now provides Audio Separator functionality.
 * Only audioSeparatorModelsDir is actively used.
 */
function getAudioSeparatorDirs(appDirs) {
  const { thirdParty, modelsDir } = appDirs;
  // LEGACY: These directories were for standalone executable (no longer used with DSU)
  const audioSeparatorDir = path.join(thirdParty, 'audio-separator');
  const audioSeparatorBinaryDir = path.join(audioSeparatorDir, 'audio-separator-cxfreeze');
  // ACTIVE: Model directory for VR/Apollo weights
  const audioSeparatorModelsDir = path.join(modelsDir, 'audio-separator');
  return { audioSeparatorDir, audioSeparatorBinaryDir, audioSeparatorModelsDir };
}

/**
 * Ensure Audio Separator model directories exist
 * Note: Only creates models directory, not legacy executable directories
 * DSU runtime is used instead of standalone executable
 * @param {Object} appDirs - The app directories from getAppDirs()
 */
function ensureAudioSeparatorDirs(appDirs) {
  const { audioSeparatorModelsDir } = getAudioSeparatorDirs(appDirs);
  // Only create models directory, not the legacy executable directories
  if (!fs.existsSync(audioSeparatorModelsDir)) {
    fs.mkdirSync(audioSeparatorModelsDir, { recursive: true });
  }
}

/**
 * Get path to Audio Separator executable
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {string} Path to executable
 */
function getAudioSeparatorPath(appDirs) {
  const { audioSeparatorBinaryDir } = getAudioSeparatorDirs(appDirs);
  return os.platform() === 'win32'
    ? path.join(audioSeparatorBinaryDir, 'audio-separator.exe')
    : path.join(audioSeparatorBinaryDir, 'audio-separator');
}

// ============================================================================
// MODEL DEFINITIONS
// ============================================================================

/**
 * All available Audio Separator (VR Architecture) models
 * 
 * Models are downloaded to ~/Documents/Max 9/SplitWizard/ThirdPartyApps/Models/audio-separator/
 */
const AUDIO_SEPARATOR_MODELS = {
  // === WIND INSTRUMENTS ===
  '17_HP-Wind_Inst-UVR': {
    category: 'wind',
    description: 'Wind Instrument Separation (flute, saxophone, clarinet, etc.)',
    stems: ['Woodwinds', 'No Woodwinds'],
    file: {
      url: 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/17_HP-Wind_Inst-UVR.pth',
      filename: '17_HP-Wind_Inst-UVR.pth',
      size: '~60MB'
    }
  },

  // === VOCALS/INSTRUMENTAL ===
  '2_HP-UVR': {
    category: 'vocals',
    description: 'Vocals/Instrumental Separation - HP v2',
    stems: ['Vocals', 'Instrumental'],
    file: {
      url: 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/2_HP-UVR.pth',
      filename: '2_HP-UVR.pth',
      size: '~60MB'
    }
  },

  '3_HP-Vocal-UVR': {
    category: 'vocals',
    description: 'Vocals Extraction - HP v3',
    stems: ['Vocals', 'Instrumental'],
    file: {
      url: 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/3_HP-Vocal-UVR.pth',
      filename: '3_HP-Vocal-UVR.pth',
      size: '~60MB'
    }
  },

  '4_HP-Vocal-UVR': {
    category: 'vocals',
    description: 'Vocals Extraction - HP v4',
    stems: ['Vocals', 'Instrumental'],
    file: {
      url: 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/4_HP-Vocal-UVR.pth',
      filename: '4_HP-Vocal-UVR.pth',
      size: '~60MB'
    }
  },

  '5_HP-Karaoke-UVR': {
    category: 'karaoke',
    description: 'Karaoke - Remove lead vocals',
    stems: ['Vocals', 'Instrumental'],
    file: {
      url: 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/5_HP-Karaoke-UVR.pth',
      filename: '5_HP-Karaoke-UVR.pth',
      size: '~60MB'
    }
  }
};

// ============================================================================
// DEFAULT MODELS (Downloaded automatically on install)
// ============================================================================

const DEFAULT_MODELS = [
  '17_HP-Wind_Inst-UVR',  // Wind instruments
  '2_HP-UVR',             // Vocals/Instrumental HP v2
  '3_HP-Vocal-UVR',       // Vocals HP v3
  '4_HP-Vocal-UVR',       // Vocals HP v4
  '5_HP-Karaoke-UVR'      // Karaoke
];

// ============================================================================
// DOWNLOAD FUNCTIONS
// ============================================================================

/**
 * Download a file with progress + retries + resume
 * @param {string} url - URL to download
 * @param {string} destPath - Destination file path
 * @param {Function} onProgress - Progress callback (optional)
 */
async function downloadFile(url, destPath, onProgress = null) {
  const https = require('https');
  const http = require('http');
  const maxRetries = 3;
  const timeoutMs = 60000;
  const stallTimeoutMs = 60000;
  const tempPath = destPath + '.part';
  
  function resolveRedirect(baseUrl, location) {
    try {
      return new URL(location, baseUrl).href;
    } catch (e) {
      return location;
    }
  }
  
  const attemptDownload = (attempt, currentUrl) => new Promise((resolve, reject) => {
    let startByte = 0;
    try {
      if (fs.existsSync(tempPath)) {
        startByte = fs.statSync(tempPath).size;
      } else if (fs.existsSync(destPath)) {
        try {
          fs.renameSync(destPath, tempPath);
          startByte = fs.statSync(tempPath).size;
        } catch (e) {
          try { fs.unlinkSync(destPath); } catch (e2) {}
          startByte = 0;
        }
      }
    } catch (e) {
      startByte = 0;
    }
    
    const protocol = currentUrl.startsWith('https') ? https : http;
    const headers = {};
    if (startByte > 0) headers.Range = `bytes=${startByte}-`;
    
    const file = fs.createWriteStream(tempPath, { flags: startByte > 0 ? 'a' : 'w' });
    let totalBytes = null;
    let downloadedBytes = startByte;
    let stallTimer = null;
    let request = null;
    
    const clearStall = () => {
      if (stallTimer) clearTimeout(stallTimer);
      stallTimer = null;
    };
    
    const armStall = () => {
      clearStall();
      stallTimer = setTimeout(() => {
        try { if (request) request.destroy(); } catch (e) {}
        reject(new Error('Download stalled'));
      }, stallTimeoutMs);
    };
    
    request = protocol.get(currentUrl, { headers }, (response) => {
      // Handle redirects (301, 302, 307, 308)
      if ([301, 302, 307, 308].includes(response.statusCode) && response.headers.location) {
        const redirectUrl = resolveRedirect(currentUrl, response.headers.location);
        response.destroy();
        file.close(() => resolve(attemptDownload(attempt, redirectUrl)));
        return;
      }
      
      if (response.statusCode !== 200 && response.statusCode !== 206) {
        response.destroy();
        reject(new Error(`HTTP ${response.statusCode}: ${currentUrl}`));
        return;
      }
      
      if (response.statusCode === 200 && startByte > 0) {
        try { file.close(); } catch (e) {}
        try { fs.unlinkSync(tempPath); } catch (e) {}
        resolve(attemptDownload(attempt, currentUrl));
        return;
      }
      
      const contentLength = parseInt(response.headers['content-length'], 10);
      const contentRange = response.headers['content-range'];
      if (contentRange) {
        const match = contentRange.match(/\/(\d+)$/);
        if (match) totalBytes = parseInt(match[1], 10);
      } else if (Number.isFinite(contentLength)) {
        totalBytes = startByte + contentLength;
      }
      
      armStall();
      
      response.on('data', (chunk) => {
        downloadedBytes += chunk.length;
        armStall();
        if (onProgress && totalBytes) {
          onProgress(downloadedBytes / totalBytes);
        }
      });
      
      response.pipe(file);
      
      file.on('finish', () => {
        file.close(() => {
          try {
            const finalSize = fs.existsSync(tempPath) ? fs.statSync(tempPath).size : 0;
            if (totalBytes && finalSize !== totalBytes) {
              try { fs.unlinkSync(tempPath); } catch (e) {}
              reject(new Error('Downloaded file size mismatch'));
              return;
            }
            try { if (fs.existsSync(destPath)) fs.unlinkSync(destPath); } catch (e) {}
            fs.renameSync(tempPath, destPath);
          } catch (e) {
            reject(e);
            return;
          }
          clearStall();
          resolve(destPath);
        });
      });
      
      file.on('error', (err) => {
        reject(err);
      });
    });
    
    request.on('error', reject);
    request.setTimeout(timeoutMs, () => {
      try { request.destroy(new Error('Download timeout')); } catch (e) {}
    });
  });
  
  let attempt = 0;
  while (attempt <= maxRetries) {
    try {
      return await attemptDownload(attempt, url);
    } catch (err) {
      attempt++;
      if (attempt > maxRetries) throw err;
      const delay = Math.min(8000, 1000 * Math.pow(2, attempt - 1));
      await new Promise((r) => setTimeout(r, delay));
    }
  }
}

/**
 * Download a specific Audio Separator model
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - App directories
 * @param {Object} callbacks - Optional callbacks { onProgress, onLog }
 */
async function downloadAudioSeparatorModel(modelName, appDirs, callbacks = {}) {
  const { onProgress, onLog } = callbacks;
  const log = onLog || console.log;
  
  const model = AUDIO_SEPARATOR_MODELS[modelName];
  if (!model) {
    throw new Error(`Unknown model: ${modelName}`);
  }
  
  const { audioSeparatorModelsDir } = getAudioSeparatorDirs(appDirs);
  ensureAudioSeparatorDirs(appDirs);
  
  const destPath = path.join(audioSeparatorModelsDir, model.file.filename);
  
  if (fs.existsSync(destPath)) {
    if (model.file && model.file.size) {
      const sizeCheck = isFileSizeMatch(destPath, model.file.size, { tolerancePercent: 0.05, minBytes: 1024 * 1024 });
      if (!sizeCheck.ok) {
        log(`[AudioSep] Size mismatch for ${model.file.filename} (expected ${model.file.size}, got ${sizeCheck.actualBytes} bytes). Re-downloading...`);
        try { fs.unlinkSync(destPath); } catch (e) {}
      } else {
        log(`[AudioSep] Model ${modelName} already exists`);
        return destPath;
      }
    } else {
      log(`[AudioSep] Model ${modelName} already exists`);
      return destPath;
    }
  }
  
  log(`[AudioSep] Downloading ${modelName}...`);
  
  await downloadFile(model.file.url, destPath, onProgress);
  
  if (model.file && model.file.size) {
    const sizeCheck = isFileSizeMatch(destPath, model.file.size, { tolerancePercent: 0.05, minBytes: 1024 * 1024 });
    if (!sizeCheck.ok) {
      try { fs.unlinkSync(destPath); } catch (e) {}
      throw new Error(`Downloaded file size mismatch for ${model.file.filename}`);
    }
  }
  
  log(`[AudioSep] Downloaded ${modelName}`);
  return destPath;
}

/**
 * Download all default Audio Separator models
 * @param {Object} appDirs - App directories
 * @param {Object} callbacks - Optional callbacks
 */
async function downloadDefaultModels(appDirs, callbacks = {}) {
  const { onLog } = callbacks;
  const log = onLog || console.log;
  
  log('[AudioSep] Downloading default models...');
  
  for (const modelName of DEFAULT_MODELS) {
    try {
      await downloadAudioSeparatorModel(modelName, appDirs, callbacks);
    } catch (err) {
      log(`[AudioSep] Failed to download ${modelName}: ${err.message}`);
    }
  }
  
  log('[AudioSep] Default models download complete');
}

/**
 * Check if model is installed
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - App directories
 */
function isModelInstalled(modelName, appDirs) {
  const model = AUDIO_SEPARATOR_MODELS[modelName];
  if (!model) return false;
  
  const { audioSeparatorModelsDir } = getAudioSeparatorDirs(appDirs);
  const modelPath = path.join(audioSeparatorModelsDir, model.file.filename);
  if (!fs.existsSync(modelPath)) return false;
  if (model.file && model.file.size) {
    const sizeCheck = isFileSizeMatch(modelPath, model.file.size, { tolerancePercent: 0.05, minBytes: 1024 * 1024 });
    return sizeCheck.ok;
  }
  return true;
}

/**
 * Get list of installed models
 * @param {Object} appDirs - App directories
 */
function getInstalledModels(appDirs) {
  return Object.keys(AUDIO_SEPARATOR_MODELS).filter(name => 
    isModelInstalled(name, appDirs)
  );
}

/**
 * List all available models by category
 * @param {Object} callbacks - Optional callbacks
 */
function listModels(callbacks = {}) {
  const { onLog } = callbacks;
  const log = onLog || console.log;
  
  const categories = {
    'wind': 'WIND INSTRUMENTS',
    'vocals': 'VOCALS/INSTRUMENTAL',
    'karaoke': 'KARAOKE'
  };
  
  log('\n=== AUDIO SEPARATOR MODELS (VR Architecture) ===\n');
  
  Object.entries(categories).forEach(([cat, title]) => {
    log(`${title}:`);
    Object.entries(AUDIO_SEPARATOR_MODELS)
      .filter(([_, info]) => info.category === cat)
      .forEach(([name, info]) => {
        log(`  - ${name}: ${info.description}`);
        log(`    Stems: ${info.stems.join(', ')}`);
      });
    log('');
  });
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Directory functions
  getAudioSeparatorDirs,
  ensureAudioSeparatorDirs,
  getAudioSeparatorPath,
  
  // Model definitions
  AUDIO_SEPARATOR_MODELS,
  DEFAULT_MODELS,
  
  // Download functions
  downloadFile,
  downloadAudioSeparatorModel,
  downloadDefaultModels,
  
  // Query functions
  isModelInstalled,
  getInstalledModels,
  listModels
};

