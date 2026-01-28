/**
 * demucs_models.js - Demucs Model Definitions for SplitWizard
 * 
 * All Demucs model definitions and download functions.
 * Import this module into setupSW+.js for complete Demucs model support.
 * 
 * Models included:
 * - htdemucs (basic hybrid transformer)
 * - htdemucs_ft (fine-tuned, bag of 4 models)
 * - htdemucs_6s (6 stems)
 * - htdemucs_mmi (more stems)
 * - mdx (single model)
 * - mdx_extra (bag of 4 models)
 * - mdx_q (quantized)
 * - mdx_extra_q (quantized, bag of 4)
 * - filosax (saxophone separation, community model)
 * 
 * Created by Ostin Solo
 * Website: ostinsolo.co.uk
 * 
 * @version 1.0.0
 */

const fs = require('fs');
const path = require('path');
const os = require('os');
const MIN_INAKI_BYTES = 50 * 1024 * 1024;

function isLikelyCompleteInaki(filePath) {
  if (!fs.existsSync(filePath)) return false;
  try {
    const size = fs.statSync(filePath).size;
    if (size < MIN_INAKI_BYTES) return false;
    return true;
  } catch (e) {
    return false;
  }
}

// ============================================================================
// DIRECTORY CONFIGURATION
// ============================================================================

/**
 * Get Demucs directories
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {Object} Demucs specific directories
 * 
 * NOTE: demucsDir and demucsBinaryDir are LEGACY - 
 * were used for demucs-cxfreeze executable.
 * DSU shared runtime now provides Demucs functionality.
 * Only demucsModelsDir, torchCacheDir, and torchCheckpointsDir are actively used.
 */
function getDemucsDirs(appDirs) {
  const { thirdParty, modelsDir } = appDirs;
  // LEGACY: These directories were for standalone executable (no longer used with DSU)
  const demucsDir = path.join(thirdParty, 'demucs');
  const demucsBinaryDir = path.join(demucsDir, 'demucs-cxfreeze');
  // ACTIVE: Model directories
  const demucsModelsDir = path.join(modelsDir, 'demucs'); // Subfolder for demucs models
  // Where official Demucs stores weights (torch hub):
  //   {TORCH_HOME}\hub\checkpoints\*.th
  const torchCacheDir = path.join(thirdParty, 'torch_cache');
  const torchCheckpointsDir = path.join(torchCacheDir, 'hub', 'checkpoints');
  return { demucsDir, demucsBinaryDir, demucsModelsDir, modelsDir, thirdParty, torchCacheDir, torchCheckpointsDir };
}

/**
 * Get path to Demucs executable
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {string} Path to executable
 */
function getDemucsPath(appDirs) {
  const { demucsBinaryDir } = getDemucsDirs(appDirs);
  return os.platform() === 'win32'
    ? path.join(demucsBinaryDir, 'demucs-cxfreeze.exe')
    : path.join(demucsBinaryDir, 'demucs-cxfreeze');
}

// ============================================================================
// MODEL DEFINITIONS
// ============================================================================

/**
 * All available Demucs models with download URLs
 * 
 * Model files are stored in ~/Documents/Max 9/SplitWizard/ThirdPartyApps/Models/
 * The DEMUCS_CACHE_DIR environment variable points to this location.
 */
const DEMUCS_MODELS = {
  // === BASIC MODELS ===
  'htdemucs': {
    category: 'basic',
    description: 'Hybrid Transformer Demucs - Fast 4-stem',
    stems: ['bass', 'drums', 'vocals', 'other'],
    files: [
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th',
        filename: '955717e8-8726e21a.th'
      },
      {
        url: 'https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/htdemucs.yaml',
        filename: 'htdemucs.yaml'
      }
    ]
  },

  // === FINE-TUNED (Bag of 4 models - BEST QUALITY) ===
  'htdemucs_ft': {
    category: 'fine-tuned',
    description: 'Fine-tuned HTDemucs - Best quality (bag of 4)',
    stems: ['bass', 'drums', 'vocals', 'other'],
    files: [
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th',
        filename: 'f7e0c4bc-ba3fe64a.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th',
        filename: 'd12395a8-e57c48e6.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th',
        filename: '92cfc3b6-ef3bcb9c.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th',
        filename: '04573f0d-f3cf25b2.th'
      },
      {
        url: 'https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/htdemucs_ft.yaml',
        filename: 'htdemucs_ft.yaml'
      }
    ]
  },

  // === 6 STEMS ===
  'htdemucs_6s': {
    category: '6-stem',
    description: '6-stem separation (+ piano, guitar)',
    stems: ['bass', 'drums', 'vocals', 'other', 'piano', 'guitar'],
    files: [
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th',
        filename: '5c90dfd2-34c22ccb.th'
      },
      {
        url: 'https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/htdemucs_6s.yaml',
        filename: 'htdemucs_6s.yaml'
      }
    ]
  },

  // === MMI (More Musical Information) ===
  'hdemucs_mmi': {
    category: 'mmi',
    description: 'Hybrid Demucs with more musical information training',
    stems: ['bass', 'drums', 'vocals', 'other'],
    files: [
      // Uses 75fc33f5 weights (NOT 955717e8 like htdemucs)
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/75fc33f5-1941ce65.th',
        filename: '75fc33f5-1941ce65.th'
      },
      {
        url: 'https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/hdemucs_mmi.yaml',
        filename: 'hdemucs_mmi.yaml'
      }
    ]
  },

  // === MDX MODELS ===
  'mdx': {
    category: 'mdx',
    description: 'MDX architecture - Bag of 4 models',
    stems: ['bass', 'drums', 'vocals', 'other'],
    files: [
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/0d19c1c6-0f06f20e.th',
        filename: '0d19c1c6-0f06f20e.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/7ecf8ec1-70f50cc9.th',
        filename: '7ecf8ec1-70f50cc9.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/c511e2ab-fe698775.th',
        filename: 'c511e2ab-fe698775.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/7d865c68-3d5dd56b.th',
        filename: '7d865c68-3d5dd56b.th'
      },
      {
        url: 'https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/mdx.yaml',
        filename: 'mdx.yaml'
      }
    ]
  },

  'mdx_extra': {
    category: 'mdx',
    description: 'MDX Extra - Bag of 4 models',
    stems: ['bass', 'drums', 'vocals', 'other'],
    files: [
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/e51eebcc-c1b80bdd.th',
        filename: 'e51eebcc-c1b80bdd.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/a1d90b5c-ae9d2452.th',
        filename: 'a1d90b5c-ae9d2452.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/5d2d6c55-db83574e.th',
        filename: '5d2d6c55-db83574e.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/cfa93e08-61801ae1.th',
        filename: 'cfa93e08-61801ae1.th'
      },
      {
        url: 'https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/mdx_extra.yaml',
        filename: 'mdx_extra.yaml'
      }
    ]
  },

  // === QUANTIZED MODELS (smaller, slightly less quality) ===
  'mdx_q': {
    category: 'quantized',
    description: 'MDX Quantized - Bag of 4 models, faster/smaller',
    stems: ['bass', 'drums', 'vocals', 'other'],
    files: [
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/6b9c2ca1-3fd82607.th',
        filename: '6b9c2ca1-3fd82607.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/b72baf4e-8778635e.th',
        filename: 'b72baf4e-8778635e.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/42e558d4-196e0e1b.th',
        filename: '42e558d4-196e0e1b.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/305bc58f-18378783.th',
        filename: '305bc58f-18378783.th'
      },
      {
        url: 'https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/mdx_q.yaml',
        filename: 'mdx_q.yaml'
      }
    ]
  },

  'mdx_extra_q': {
    category: 'quantized',
    description: 'MDX Extra Quantized - Best balance of speed/quality',
    stems: ['bass', 'drums', 'vocals', 'other'],
    files: [
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/83fc094f-4a16d450.th',
        filename: '83fc094f-4a16d450.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/464b36d7-e5a9386e.th',
        filename: '464b36d7-e5a9386e.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/14fc6a69-a89dd0ee.th',
        filename: '14fc6a69-a89dd0ee.th'
      },
      {
        url: 'https://dl.fbaipublicfiles.com/demucs/mdx_final/7fd6ef75-a905dd85.th',
        filename: '7fd6ef75-a905dd85.th'
      },
      {
        url: 'https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/mdx_extra_q.yaml',
        filename: 'mdx_extra_q.yaml'
      }
    ]
  },

  // === SPECIALTY MODELS (Community) ===
  'filosax': {
    category: 'specialty',
    description: 'Saxophone Separation - Isolates sax from backing track',
    stems: ['Sax', 'BackingMono'],
    source: 'https://huggingface.co/xavriley/demucs_v3_saxophone_separation',
    paper: 'https://arxiv.org/abs/2405.16687',
    files: [
      {
        url: 'https://huggingface.co/xavriley/demucs_v3_saxophone_separation/resolve/main/filosax_demucs_v3_14.22_SDR.th',
        filename: 'filosax_demucs_v3_14.22_SDR.th'
      }
    ]
  },

  // === DRUM KIT SEPARATION ===
  'inaki': {
    category: 'specialty',
    description: 'Drum Kit Separation - Splits drums into kick, snare, toms, cymbals',
    stems: ['kick', 'snare', 'toms', 'cymbals'],
    source: 'https://drive.google.com/file/d/1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC',
    // Note: Google Drive URL with confirm bypass for large files
    files: [
      {
        url: 'https://drive.usercontent.google.com/download?id=1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC&export=download&confirm=t',
        filename: '49469ca8.th',
        isGoogleDrive: true  // Flag for special download handling
      }
    ]
  }
};

// ============================================================================
// VERIFICATION FUNCTIONS
// ============================================================================

/**
 * Check if a specific model is installed
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {boolean} True if all model files exist
 */
function isModelInstalled(modelName, appDirs) {
  const { demucsModelsDir, modelsDir, thirdParty, torchCheckpointsDir } = getDemucsDirs(appDirs);
  const model = DEMUCS_MODELS[modelName];
  
  if (!model) return false;
  
  // Special handling for Inaki - check model folder (moved there after download)
  // Also check old locations for backward compatibility
  if (modelName === 'inaki') {
    const modelFolder = path.join(thirdParty, 'model');
    const inakiRepoPath = path.join(modelFolder, 'inaki.th');
    // Check new location first, then old locations for compatibility
    if (isLikelyCompleteInaki(inakiRepoPath)) return true;
    return model.files.every(file => 
      isLikelyCompleteInaki(path.join(demucsModelsDir, file.filename)) ||
      isLikelyCompleteInaki(path.join(modelsDir, file.filename)) ||
      isLikelyCompleteInaki(path.join(torchCheckpointsDir, file.filename))
    );
  }
  
  return model.files.every(file => 
    fs.existsSync(path.join(demucsModelsDir, file.filename)) ||
    fs.existsSync(path.join(torchCheckpointsDir, file.filename))
  );
}

/**
 * Get list of installed models
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {Array} List of installed model names
 */
function getInstalledModels(appDirs) {
  return Object.keys(DEMUCS_MODELS).filter(name => isModelInstalled(name, appDirs));
}

/**
 * Get list of missing files for a model
 * @param {string} modelName - Name of the model
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @returns {Array} List of missing file objects
 */
function getMissingFiles(modelName, appDirs) {
  const { demucsModelsDir, modelsDir, thirdParty, torchCheckpointsDir } = getDemucsDirs(appDirs);
  const model = DEMUCS_MODELS[modelName];
  
  if (!model) return [];
  
  // Special handling for Inaki - check model folder (moved there after download)
  // Also check old locations for backward compatibility
  if (modelName === 'inaki') {
    const modelFolder = path.join(thirdParty, 'model');
    const inakiRepoPath = path.join(modelFolder, 'inaki.th');
    // If exists in new location and looks complete, nothing is missing
    if (isLikelyCompleteInaki(inakiRepoPath)) {
      return [];
    }
    // Otherwise check old locations
    return model.files.filter(file => 
      !isLikelyCompleteInaki(path.join(demucsModelsDir, file.filename)) &&
      !isLikelyCompleteInaki(path.join(modelsDir, file.filename)) &&
      !isLikelyCompleteInaki(path.join(torchCheckpointsDir, file.filename))
    );
  }
  
  return model.files.filter(file => 
    !fs.existsSync(path.join(demucsModelsDir, file.filename)) &&
    !fs.existsSync(path.join(torchCheckpointsDir, file.filename))
  );
}

/**
 * Verify Demucs installation and models
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 * @param {Function} callback - Callback(isReady)
 */
function verifyDemucs(appDirs, Max, callback) {
  const exe = getDemucsPath(appDirs);
  const exeExists = fs.existsSync(exe);
  const installedModels = getInstalledModels(appDirs);
  
  if (exeExists && installedModels.length > 0) {
    Max.post(`Demucs verified (${installedModels.length} model(s) installed)`);
    Max.post(`  Installed: ${installedModels.join(', ')}`);
    callback(true);
    return;
  }
  
  if (!exeExists) Max.post('Demucs executable not found');
  if (installedModels.length === 0) Max.post('No Demucs models installed');
  
  callback(false);
}

// ============================================================================
// DOWNLOAD FUNCTIONS
// ============================================================================

/**
 * Download a specific Demucs model
 * @param {string} modelName - Name of the model to download
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 * @param {Function} downloadToFile - Download function from setupSW+.js
 * @param {Function} callback - Callback(error)
 */
function downloadDemucsModel(modelName, appDirs, Max, downloadToFile, callback) {
  const { demucsModelsDir, modelsDir, torchCheckpointsDir } = getDemucsDirs(appDirs);
  const model = DEMUCS_MODELS[modelName];
  
  if (!model) {
    Max.post(`Unknown Demucs model: ${modelName}`);
    Max.post('Available models: ' + Object.keys(DEMUCS_MODELS).join(', '));
    callback(new Error('Unknown model: ' + modelName));
    return;
  }
  
  // Ensure models directory exists
  if (!fs.existsSync(demucsModelsDir)) {
    fs.mkdirSync(demucsModelsDir, { recursive: true });
  }
  if (!fs.existsSync(torchCheckpointsDir)) {
    fs.mkdirSync(torchCheckpointsDir, { recursive: true });
  }
  
  // Inaki goes to main models directory (for backward compatibility)
  const isWin = os.platform() === 'win32';
  const getTargetDirForFile = (file) => {
    if (modelName === 'inaki') return modelsDir;
    // Community models like filosax must live in Models\demucs for --repo usage (NOT torch hub).
    if (modelName === 'filosax') return demucsModelsDir;
    // On Windows, store large .th weights where Demucs/torch hub expects them.
    if (isWin && /\.th$/i.test(file.filename)) return torchCheckpointsDir;
    return demucsModelsDir;
  };
  
  Max.post('');
  Max.post(`Downloading Demucs model: ${modelName}`);
  Max.post(`  ${model.description}`);
  Max.post(`  Stems: ${model.stems.join(', ')}`);
  Max.post(`  Files: ${model.files.length}`);
  
  const missingFiles = getMissingFiles(modelName, appDirs);
  
  if (missingFiles.length === 0) {
    Max.post(`  ✅ Model already installed: ${modelName}`);
    callback(null);
    return;
  }
  
  Max.post(`  Downloading ${missingFiles.length} file(s)...`);
  if (modelName === 'inaki') {
    Max.post(`  Note: Inaki is a large file (~85MB), this may take a few minutes...`);
  }
  
  let index = 0;
  const downloadNext = () => {
    if (index >= missingFiles.length) {
      // For Inaki, verify file size and move to model folder for --repo usage
      if (modelName === 'inaki') {
        // Inaki is special: we download a single .th and then move it into ThirdPartyApps\model\inaki.th
        // so Demucs can use it via --repo "<Models\demucs>" + internal repo logic.
        const inakiFile = (model.files || []).find(f => /\.th$/i.test(f.filename)) || { filename: '49469ca8.th' };
        const inakiDownloadedPath = path.join(getTargetDirForFile(inakiFile), inakiFile.filename);
        if (fs.existsSync(inakiDownloadedPath)) {
          const stats = fs.statSync(inakiDownloadedPath);
          if (stats.size < 10 * 1024 * 1024) { // Less than 10MB is suspicious
            Max.post(`  ⚠️ Inaki file seems too small (${(stats.size / 1024 / 1024).toFixed(2)}MB), may be incomplete`);
            try { fs.unlinkSync(inakiDownloadedPath); } catch (e) {}
            callback(new Error('Downloaded file is too small, may be incomplete'));
            return;
          }
          // Move to model folder for --repo "model" usage with demucs
          const { thirdParty } = appDirs;
          const modelFolder = path.join(thirdParty, 'model');
          const inakiRepoPath = path.join(modelFolder, 'inaki.th');
          try {
            if (!fs.existsSync(modelFolder)) {
              fs.mkdirSync(modelFolder, { recursive: true });
            }
            // Use rename (move) instead of copy to avoid duplicate files
            fs.renameSync(inakiDownloadedPath, inakiRepoPath);
            Max.post(`  ✅ Inaki model moved to model folder for --repo usage`);
          } catch (moveErr) {
            // Fallback to copy+delete if rename fails (e.g., cross-device)
            try {
              fs.copyFileSync(inakiDownloadedPath, inakiRepoPath);
              fs.unlinkSync(inakiDownloadedPath);
              Max.post(`  ✅ Inaki model moved to model folder for --repo usage`);
            } catch (copyErr) {
              Max.post(`  ⚠️ Could not move to model folder: ${copyErr.message}`);
            }
          }
        }
      }
      Max.post(`  ✅ Model installed: ${modelName}`);
      callback(null);
      return;
    }
    
    const file = missingFiles[index];
    const destPath = path.join(getTargetDirForFile(file), file.filename);
    
    Max.post(`  [${index + 1}/${missingFiles.length}] ${file.filename}`);
    
    // Aggregate progress across all missing files so UI gets a single smooth 0→100 per model.
    // Strategy: overall = ((fileIndex + filePercent/100) / totalFiles) * 100
    const totalFiles = Math.max(1, missingFiles.length);
    const baseIndex = index; // capture current file index
    const reportOverall = (filePercent) => {
      const fp = Math.max(0, Math.min(100, Number(filePercent) || 0));
      const overall = ((baseIndex + (fp / 100)) / totalFiles) * 100;
      try { Max.outlet('installProgress', 'demucs', modelName, Math.round(overall)); } catch (e) {}
    };
    reportOverall(0);

    downloadToFile(file.url, destPath, (err) => {
      if (err) {
        Max.post(`  ❌ Failed to download ${file.filename}: ${err.message}`);
        callback(err);
        return;
      }
      
      index++;
      // Ensure we end this file at its final progress position
      reportOverall(100);
      downloadNext();
    }, {
      onProgress: (percent) => reportOverall(percent)
    });
  };
  
  downloadNext();
}

/**
 * Download all Demucs models
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 * @param {Function} downloadToFile - Download function from setupSW+.js
 * @param {Function} callback - Callback(error)
 */
function downloadAllDemucsModels(appDirs, Max, downloadToFile, callback) {
  const modelNames = Object.keys(DEMUCS_MODELS);
  
  Max.post('');
  Max.post('==================================================');
  Max.post('DOWNLOADING ALL DEMUCS MODELS');
  Max.post('==================================================');
  Max.post(`Total models: ${modelNames.length}`);
  Max.post('');
  
  let index = 0;
  let successCount = 0;
  let failCount = 0;
  
  const downloadNext = () => {
    if (index >= modelNames.length) {
      Max.post('');
      Max.post('==================================================');
      Max.post('DEMUCS MODELS DOWNLOAD COMPLETE');
      Max.post(`  Success: ${successCount}`);
      Max.post(`  Failed: ${failCount}`);
      Max.post('==================================================');
      callback(failCount > 0 ? new Error(`${failCount} models failed`) : null);
      return;
    }
    
    const modelName = modelNames[index];
    
    downloadDemucsModel(modelName, appDirs, Max, downloadToFile, (err) => {
      if (err) {
        failCount++;
      } else {
        successCount++;
      }
      index++;
      downloadNext();
    });
  };
  
  downloadNext();
}

/**
 * Download essential Demucs models (htdemucs_ft only - already default)
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 * @param {Function} downloadToFile - Download function from setupSW+.js
 * @param {Function} callback - Callback(error)
 */
function downloadEssentialModels(appDirs, Max, downloadToFile, callback) {
  Max.post('');
  Max.post('Downloading essential Demucs models...');
  downloadDemucsModel('htdemucs_ft', appDirs, Max, downloadToFile, callback);
}

/**
 * Download recommended Demucs models (htdemucs_ft, htdemucs, mdx_extra_q)
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 * @param {Function} downloadToFile - Download function from setupSW+.js
 * @param {Function} callback - Callback(error)
 */
function downloadRecommendedModels(appDirs, Max, downloadToFile, callback) {
  const recommended = ['htdemucs_ft', 'htdemucs', 'mdx_extra_q', 'htdemucs_6s'];
  
  Max.post('');
  Max.post('==================================================');
  Max.post('DOWNLOADING RECOMMENDED DEMUCS MODELS');
  Max.post('==================================================');
  Max.post('Models: ' + recommended.join(', '));
  Max.post('');
  
  let index = 0;
  let successCount = 0;
  let failCount = 0;
  
  const downloadNext = () => {
    if (index >= recommended.length) {
      Max.post('');
      Max.post('==================================================');
      Max.post('RECOMMENDED MODELS DOWNLOAD COMPLETE');
      Max.post(`  Success: ${successCount}`);
      Max.post(`  Failed: ${failCount}`);
      Max.post('==================================================');
      callback(failCount > 0 ? new Error(`${failCount} models failed`) : null);
      return;
    }
    
    const modelName = recommended[index];
    
    downloadDemucsModel(modelName, appDirs, Max, downloadToFile, (err) => {
      if (err) {
        failCount++;
      } else {
        successCount++;
      }
      index++;
      downloadNext();
    });
  };
  
  downloadNext();
}

// ============================================================================
// LIST FUNCTIONS
// ============================================================================

/**
 * List all available Demucs models
 * @param {Object} appDirs - The app directories from getAppDirs()
 * @param {Function} Max - Max API object
 */
function listDemucsModels(appDirs, Max) {
  const categories = {
    'basic': 'BASIC (fast)',
    'fine-tuned': 'FINE-TUNED (best quality)',
    '6-stem': '6-STEM (+ piano/guitar)',
    'mmi': 'MMI (more musical info)',
    'mdx': 'MDX ARCHITECTURE',
    'quantized': 'QUANTIZED (faster, smaller)'
  };
  
  Max.post('');
  Max.post('==================================================');
  Max.post('DEMUCS MODELS');
  Max.post('==================================================');
  
  for (const [cat, title] of Object.entries(categories)) {
    const catModels = Object.entries(DEMUCS_MODELS).filter(([_, m]) => m.category === cat);
    if (catModels.length === 0) continue;
    
    Max.post('');
    Max.post(`${title}:`);
    
    for (const [name, info] of catModels) {
      const installed = isModelInstalled(name, appDirs);
      const fileCount = info.files.length;
      Max.post(`  ${installed ? '✅' : '❌'} ${name} (${fileCount} files)`);
      Max.post(`      ${info.description}`);
      Max.post(`      Stems: ${info.stems.join(', ')}`);
    }
  }
  
  // Add note about Inaki
  Max.post('');
  Max.post('DRUM SEPARATION:');
  Max.post('  ℹ️  inaki - Drum kit separation (downloaded separately)');
  
  Max.post('');
  Max.post('==================================================');
  Max.post('');
}

/**
 * Get model info
 * @param {string} modelName - Name of the model
 * @returns {Object|null} Model info or null if not found
 */
function getModelInfo(modelName) {
  return DEMUCS_MODELS[modelName] || null;
}

/**
 * Get all model names
 * @returns {Array} List of all model names
 */
function getAllModelNames() {
  return Object.keys(DEMUCS_MODELS);
}

/**
 * Get models by category
 * @param {string} category - Category name
 * @returns {Array} List of model names in that category
 */
function getModelsByCategory(category) {
  return Object.entries(DEMUCS_MODELS)
    .filter(([_, m]) => m.category === category)
    .map(([name]) => name);
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Directory functions
  getDemucsDirs,
  getDemucsPath,
  
  // Model data
  DEMUCS_MODELS,
  
  // Verification
  verifyDemucs,
  isModelInstalled,
  getInstalledModels,
  getMissingFiles,
  
  // Download
  downloadDemucsModel,
  downloadAllDemucsModels,
  downloadEssentialModels,
  downloadRecommendedModels,
  
  // Listing
  listDemucsModels,
  getModelInfo,
  getAllModelNames,
  getModelsByCategory
};

