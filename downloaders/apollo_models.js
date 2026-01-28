/**
 * apollo_models.js - Apollo Audio Restoration Model Definitions for SplitWizard
 * 
 * Apollo converts lossy compressed audio (MP3, AAC) to higher quality
 * by reconstructing lost frequency information using Band-sequence Modeling.
 * 
 * Models included:
 * - Lew Universal (RECOMMENDED) - Best quality for any lossy files
 * - Official JusperLee - General restoration
 * - Lew V2 - Lightweight, vocal enhancement
 * - Big/EDM by essid - EDM/Electronic music
 * 
 * Based on: https://github.com/JusperLee/Apollo
 * Paper: "Apollo: Band-sequence Modeling for High-Quality Music Restoration"
 * 
 * Created by Ostin Solo
 * Website: ostinsolo.co.uk
 * 
 * @version 1.0.0
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { isFileSizeMatch } = require('./model_integrity');
const http = require('http');

// ============================================================================
// MODEL DEFINITIONS
// ============================================================================

/**
 * All available Apollo models
 * 
 * Models are downloaded to ~/Documents/Max 9/SplitWizard/ThirdPartyApps/Models/apollo/
 */
const APOLLO_MODELS = {
  // === RECOMMENDED ===
  'apollo_lew_uni': {
    category: 'recommended',
    description: 'Lew Universal - BEST quality for any lossy files',
    feature_dim: 384,
    layer: 6,
    files: [
      {
        url: 'https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/apollo_model_uni.ckpt',
        filename: 'apollo_lew_uni.ckpt'
      },
      {
        url: 'https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/uni/config_apollo_uni.yaml',
        filename: 'apollo_lew_uni.yaml'
      }
    ],
    size: '~140MB'
  },

  // === OFFICIAL ===
  'apollo_official': {
    category: 'official',
    description: 'Official JusperLee - General restoration',
    feature_dim: 256,
    layer: 6,
    files: [
      {
        url: 'https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin',
        filename: 'apollo_official.bin'
      }
    ],
    size: '~63MB'
  },

  // === LIGHTWEIGHT ===
  'apollo_lew_v2': {
    category: 'lightweight',
    description: 'Lew V2 - Lightweight, vocal enhancement',
    feature_dim: 192,
    layer: 6,
    files: [
      {
        url: 'https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/2.0/apollo_model_v2.ckpt',
        filename: 'apollo_lew_v2.ckpt'
      },
      {
        url: 'https://github.com/deton24/Lew-s-vocal-enhancer-for-Apollo-by-JusperLee/releases/download/2.0/config_apollo_vocal.yaml',
        filename: 'apollo_lew_v2.yaml'
      }
    ],
    size: '~36MB'
  },

  // === EDM/ELECTRONIC ===
  'apollo_edm_big': {
    category: 'specialized',
    description: 'Big/EDM by essid - EDM/Electronic music',
    feature_dim: 256,
    layer: 6,
    files: [
      {
        url: 'https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Apollo/apollo_edm_big_by_essid.ckpt',
        filename: 'apollo_edm_big.ckpt'
      },
      {
        url: 'https://huggingface.co/Politrees/UVR_resources/resolve/main/models/Apollo/apollo_edm_big_by_essid.yaml',
        filename: 'apollo_edm_big.yaml'
      }
    ],
    size: '~222MB'
  }
};

// Default models to download on install (all models)
const DEFAULT_MODELS = [
  'apollo_lew_uni',    // Best quality - universal
  'apollo_official',   // Official JusperLee
  'apollo_lew_v2',     // Lightweight - vocal enhancement
  'apollo_edm_big'     // EDM/Electronic music
];

// ============================================================================
// DIRECTORY FUNCTIONS
// ============================================================================

/**
 * Get Apollo directories
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {Object} Apollo specific directories
 */
function getApolloDirs(appDirs) {
  const { modelsDir } = appDirs;
  const apolloModelsDir = path.join(modelsDir, 'apollo');
  return { apolloModelsDir };
}

/**
 * Ensure Apollo directories exist
 * @param {Object} appDirs - The app directories from getAppDirs()
 */
function ensureApolloDirs(appDirs) {
  const { apolloModelsDir } = getApolloDirs(appDirs);
  if (!fs.existsSync(apolloModelsDir)) {
    fs.mkdirSync(apolloModelsDir, { recursive: true });
  }
}

// ============================================================================
// DOWNLOAD FUNCTIONS
// ============================================================================

/**
 * Download a file with redirect handling
 * @param {string} url - URL to download
 * @param {string} destPath - Destination file path
 * @param {Function} onProgress - Progress callback (optional)
 */
async function downloadFile(url, destPath, onProgress = null) {
  return new Promise((resolve, reject) => {
    let parsedUrl;
    try {
      parsedUrl = new URL(url);
    } catch (e) {
      reject(e);
      return;
    }

    const protocol = parsedUrl.protocol === 'https:' ? https : http;

    const request = protocol.get(parsedUrl, (response) => {
      // Handle redirects (301, 302, 307, 308)
      if ([301, 302, 307, 308].includes(response.statusCode)) {
        const location = response.headers.location;
        if (!location) {
          reject(new Error(`Redirect without Location header: ${url}`));
          return;
        }

        // Some hosts return relative redirects. Resolve against the current URL.
        const redirectUrl = new URL(location, parsedUrl).toString();

        downloadFile(redirectUrl, destPath, onProgress)
          .then(resolve)
          .catch(reject);
        return;
      }
      
      if (response.statusCode !== 200) {
        reject(new Error(`HTTP ${response.statusCode}: ${url}`));
        return;
      }
      
      const totalSize = parseInt(response.headers['content-length'], 10);
      let downloadedSize = 0;
      
      const file = fs.createWriteStream(destPath);
      
      response.on('data', (chunk) => {
        downloadedSize += chunk.length;
        if (onProgress && totalSize) {
          onProgress(downloadedSize / totalSize);
        }
      });
      
      response.pipe(file);
      
      file.on('finish', () => {
        file.close();
        resolve(destPath);
      });
      
      file.on('error', (err) => {
        fs.unlink(destPath, () => {});
        reject(err);
      });
    });
    
    request.on('error', reject);
    request.setTimeout(120000, () => {
      request.destroy();
      reject(new Error('Download timeout'));
    });
  });
}

/**
 * Download a specific Apollo model
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - App directories
 * @param {Object} callbacks - Optional callbacks { onProgress, onLog }
 */
async function downloadApolloModel(modelName, appDirs, callbacks = {}) {
  const { onProgress, onLog } = callbacks;
  const log = onLog || console.log;
  
  const model = APOLLO_MODELS[modelName];
  if (!model) {
    throw new Error(`Unknown Apollo model: ${modelName}`);
  }
  
  const { apolloModelsDir } = getApolloDirs(appDirs);
  ensureApolloDirs(appDirs);
  
  log(`[Apollo] Downloading ${modelName}...`);
  log(`[Apollo] Size: ${model.size}`);
  
  for (const file of model.files) {
    const destPath = path.join(apolloModelsDir, file.filename);
    const isCheckpoint = file.filename.endsWith('.ckpt') || file.filename.endsWith('.bin');
    
    if (fs.existsSync(destPath)) {
      if (isCheckpoint && model.size) {
        const sizeCheck = isFileSizeMatch(destPath, model.size, { tolerancePercent: 0.05, minBytes: 1024 * 1024 });
        if (!sizeCheck.ok) {
          log(`[Apollo] Size mismatch for ${file.filename} (expected ${model.size}, got ${sizeCheck.actualBytes} bytes). Re-downloading...`);
          try { fs.unlinkSync(destPath); } catch (e) {}
        } else {
          log(`[Apollo] ${file.filename} already exists, skipping`);
          continue;
        }
      } else {
        log(`[Apollo] ${file.filename} already exists, skipping`);
        continue;
      }
    }
    
    log(`[Apollo] Downloading ${file.filename}...`);
    await downloadFile(file.url, destPath, onProgress);
    log(`[Apollo] Downloaded ${file.filename}`);
  }
  
  // Verify all files exist after download - retry any missing ones
  const missing = getMissingFiles(modelName, appDirs);
  if (missing.length > 0) {
    log(`[Apollo] Verification found ${missing.length} missing file(s), retrying...`);
    for (const file of missing) {
      const destPath = path.join(apolloModelsDir, file.filename);
      log(`[Apollo] Retrying: ${file.filename}...`);
      try {
        await downloadFile(file.url, destPath, onProgress);
        log(`[Apollo] Downloaded ${file.filename}`);
      } catch (err) {
        log(`[Apollo] Failed to download ${file.filename}: ${err.message}`);
        throw new Error(`Failed to download ${file.filename}: ${err.message}`);
      }
    }
  }
  
  // Final verification
  if (!isModelInstalled(modelName, appDirs)) {
    const stillMissing = getMissingFiles(modelName, appDirs);
    throw new Error(`Model ${modelName} incomplete after download. Missing: ${stillMissing.map(f => f.filename).join(', ')}`);
  }
  
  log(`[Apollo] Model ${modelName} ready!`);
  return true;
}

/**
 * Download all default Apollo models
 * @param {Object} appDirs - App directories
 * @param {Object} callbacks - Optional callbacks
 */
async function downloadDefaultModels(appDirs, callbacks = {}) {
  const { onLog } = callbacks;
  const log = onLog || console.log;
  
  log('[Apollo] Downloading default models...');
  
  for (const modelName of DEFAULT_MODELS) {
    try {
      await downloadApolloModel(modelName, appDirs, callbacks);
    } catch (err) {
      log(`[Apollo] Failed to download ${modelName}: ${err.message}`);
    }
  }
  
  log('[Apollo] Default models download complete');
}

// ============================================================================
// QUERY FUNCTIONS
// ============================================================================

/**
 * Check if model is installed
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - App directories
 */
function isModelInstalled(modelName, appDirs) {
  const model = APOLLO_MODELS[modelName];
  if (!model) return false;
  
  const { apolloModelsDir } = getApolloDirs(appDirs);
  
  // Check if all files exist
  for (const file of model.files) {
    const filePath = path.join(apolloModelsDir, file.filename);
    if (!fs.existsSync(filePath)) {
      return false;
    }
  }
  
  // If size hint is provided, validate the main checkpoint file size
  if (model.size) {
    const checkpointFile = model.files.find(f => f.filename.endsWith('.ckpt') || f.filename.endsWith('.bin'));
    if (checkpointFile) {
      const checkpointPath = path.join(apolloModelsDir, checkpointFile.filename);
      const sizeCheck = isFileSizeMatch(checkpointPath, model.size, { tolerancePercent: 0.05, minBytes: 1024 * 1024 });
      if (!sizeCheck.ok) return false;
    }
  }
  return true;
}

/**
 * Get missing files for a model (for incomplete installs)
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - App directories
 * @returns {Array} List of missing file objects
 */
function getMissingFiles(modelName, appDirs) {
  const model = APOLLO_MODELS[modelName];
  if (!model) return [];
  
  const { apolloModelsDir } = getApolloDirs(appDirs);
  
  return model.files.filter(file => {
    const filePath = path.join(apolloModelsDir, file.filename);
    if (!fs.existsSync(filePath)) return true;
    if (model.size && (file.filename.endsWith('.ckpt') || file.filename.endsWith('.bin'))) {
      const sizeCheck = isFileSizeMatch(filePath, model.size, { tolerancePercent: 0.05, minBytes: 1024 * 1024 });
      return !sizeCheck.ok;
    }
    return false;
  });
}

/**
 * Check if model has partial install (some files exist, some missing)
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - App directories
 * @returns {boolean} True if model has some files but is incomplete
 */
function isModelPartiallyInstalled(modelName, appDirs) {
  const model = APOLLO_MODELS[modelName];
  if (!model) return false;
  
  const { apolloModelsDir } = getApolloDirs(appDirs);
  
  let existCount = 0;
  for (const file of model.files) {
    const filePath = path.join(apolloModelsDir, file.filename);
    if (!fs.existsSync(filePath)) continue;
    if (model.size && (file.filename.endsWith('.ckpt') || file.filename.endsWith('.bin'))) {
      const sizeCheck = isFileSizeMatch(filePath, model.size, { tolerancePercent: 0.05, minBytes: 1024 * 1024 });
      if (!sizeCheck.ok) continue;
    }
    existCount++;
  }
  
  // Partial = some files exist but not all
  return existCount > 0 && existCount < model.files.length;
}

/**
 * Get list of installed models
 * @param {Object} appDirs - App directories
 */
function getInstalledModels(appDirs) {
  return Object.keys(APOLLO_MODELS).filter(name => 
    isModelInstalled(name, appDirs)
  );
}

/**
 * Repair incomplete model installs by downloading missing files
 * @param {Object} appDirs - App directories
 * @param {Object} callbacks - Optional callbacks { onProgress, onLog }
 */
async function repairIncompleteModels(appDirs, callbacks = {}) {
  const { onLog } = callbacks;
  const log = onLog || console.log;
  
  const { apolloModelsDir } = getApolloDirs(appDirs);
  ensureApolloDirs(appDirs);
  
  let repaired = 0;
  
  for (const modelName of Object.keys(APOLLO_MODELS)) {
    if (isModelPartiallyInstalled(modelName, appDirs)) {
      const missing = getMissingFiles(modelName, appDirs);
      log(`[Apollo] Found incomplete install: ${modelName} - missing ${missing.length} file(s)`);
      
      for (const file of missing) {
        const destPath = path.join(apolloModelsDir, file.filename);
        log(`[Apollo] Downloading missing: ${file.filename}`);
        try {
          await downloadFile(file.url, destPath);
          log(`[Apollo] Repaired: ${file.filename}`);
          repaired++;
        } catch (err) {
          log(`[Apollo] Failed to repair ${file.filename}: ${err.message}`);
        }
      }
    }
  }
  
  if (repaired > 0) {
    log(`[Apollo] Repaired ${repaired} missing file(s)`);
  }
  
  return repaired;
}

/**
 * Get model checkpoint path
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - App directories
 * @returns {string|null} Path to checkpoint file or null
 */
function getModelPath(modelName, appDirs) {
  const model = APOLLO_MODELS[modelName];
  if (!model) return null;
  
  const { apolloModelsDir } = getApolloDirs(appDirs);
  
  // Find the .ckpt or .bin file
  const checkpointFile = model.files.find(f => 
    f.filename.endsWith('.ckpt') || f.filename.endsWith('.bin')
  );
  
  if (!checkpointFile) return null;
  return path.join(apolloModelsDir, checkpointFile.filename);
}

/**
 * Get model config path
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - App directories
 * @returns {string|null} Path to config file or null
 */
function getConfigPath(modelName, appDirs) {
  const model = APOLLO_MODELS[modelName];
  if (!model) return null;
  
  const { apolloModelsDir } = getApolloDirs(appDirs);
  
  // Find the .yaml file
  const configFile = model.files.find(f => f.filename.endsWith('.yaml'));
  
  if (!configFile) return null;
  return path.join(apolloModelsDir, configFile.filename);
}

/**
 * List all available models by category
 * @param {Object} callbacks - Optional callbacks
 */
function listModels(callbacks = {}) {
  const { onLog } = callbacks;
  const log = onLog || console.log;
  
  const categories = {
    'recommended': 'RECOMMENDED',
    'official': 'OFFICIAL',
    'lightweight': 'LIGHTWEIGHT',
    'specialized': 'SPECIALIZED'
  };
  
  log('');
  log('=== APOLLO AUDIO RESTORATION MODELS ===');
  log('');
  
  Object.entries(categories).forEach(([cat, title]) => {
    const models = Object.entries(APOLLO_MODELS)
      .filter(([_, info]) => info.category === cat);
    
    if (models.length === 0) return;
    
    log(`${title}:`);
    models.forEach(([name, info]) => {
      log(`  - ${name}: ${info.description}`);
      log(`    feature_dim=${info.feature_dim}, layer=${info.layer}, size=${info.size}`);
    });
    log('');
  });
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Model definitions
  APOLLO_MODELS,
  DEFAULT_MODELS,
  
  // Directory functions
  getApolloDirs,
  ensureApolloDirs,
  
  // Download functions
  downloadFile,
  downloadApolloModel,
  downloadDefaultModels,
  
  // Query functions
  isModelInstalled,
  getInstalledModels,
  getModelPath,
  getConfigPath,
  listModels,
  
  // Repair functions
  getMissingFiles,
  isModelPartiallyInstalled,
  repairIncompleteModels
};

