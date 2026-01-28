const fs = require('fs');

/**
 * Parse human-readable size to bytes (supports "~60MB", "3.5 GB").
 * Uses decimal units (MB = 1,000,000 bytes) to match release sizes.
 * Returns null if parsing fails.
 * @param {string} sizeStr
 * @returns {number|null}
 */
function parseSizeToBytes(sizeStr) {
  if (!sizeStr || typeof sizeStr !== 'string') return null;
  const match = sizeStr.trim().match(/^~?\s*([\d.]+)\s*([KMG]B)\s*$/i);
  if (!match) return null;
  const value = parseFloat(match[1]);
  if (!Number.isFinite(value)) return null;
  const unit = match[2].toUpperCase();
  const multipliers = { KB: 1000, MB: 1000 ** 2, GB: 1000 ** 3 };
  return Math.round(value * multipliers[unit]);
}

/**
 * Check file size against expected size string.
 * Returns details for logging or validation.
 * @param {string} filePath
 * @param {string} expectedSizeStr
 * @param {Object} [options]
 * @param {number} [options.tolerancePercent=0.02]
 * @param {number} [options.minBytes=1048576]
 * @param {boolean} [options.enforceUpperBound=false]
 * @returns {{ok:boolean, actualBytes:number|null, expectedBytes:number|null, delta:number|null, tolerance:number|null}}
 */
function isFileSizeMatch(filePath, expectedSizeStr, options = {}) {
  const expectedBytes = parseSizeToBytes(expectedSizeStr);
  if (!expectedBytes) {
    return { ok: true, actualBytes: null, expectedBytes: null, delta: null, tolerance: null };
  }
  if (!fs.existsSync(filePath)) {
    return { ok: false, actualBytes: null, expectedBytes, delta: null, tolerance: null };
  }
  const actualBytes = fs.statSync(filePath).size;
  const tolerancePercent = options.tolerancePercent !== undefined ? options.tolerancePercent : 0.02;
  const minBytes = options.minBytes !== undefined ? options.minBytes : 1024 * 1024;
  const tolerance = Math.max(expectedBytes * tolerancePercent, minBytes);
  const delta = Math.abs(actualBytes - expectedBytes);
  const lowerBound = expectedBytes - tolerance;
  const enforceUpperBound = options.enforceUpperBound === true;
  const ok = enforceUpperBound
    ? delta <= tolerance
    : actualBytes >= lowerBound;
  return { ok, actualBytes, expectedBytes, delta, tolerance };
}

module.exports = {
  parseSizeToBytes,
  isFileSizeMatch
};
