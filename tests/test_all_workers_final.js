/**
 * DSU Final Performance Test - All Workers
 * Tests optimized performance with model caching
 * 
 * Key findings from this test:
 * - Cold start: 6-25s (includes model loading)
 * - Cached: 0.2-1.5s (actual processing!)
 * - Speedup: 6-26x faster when cached
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const readline = require('readline');

// Configuration
const DSU_DIR = path.join(__dirname, '..', 'dsu');
const MODELS_DIR = 'C:\\Users\\soloo\\Documents\\DSU-VSTOPIA\\ThirdPartyApps\\Models';
const TEST_AUDIO = 'C:\\Users\\soloo\\Documents\\0_20_56_1_27_2026_.wav';
const OUTPUT_DIR = path.join(__dirname, 'final_test_output');
const CONFIGS_DIR = path.join(__dirname, '..', 'configs');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

class DSUWorker {
  constructor(exePath, args = []) {
    this.exePath = exePath;
    this.args = args;
    this.proc = null;
    this.rl = null;
    this.responseQueue = [];
    this.waitingResolve = null;
  }

  async start() {
    return new Promise((resolve, reject) => {
      this.proc = spawn(this.exePath, this.args, { stdio: ['pipe', 'pipe', 'pipe'] });
      this.rl = readline.createInterface({ input: this.proc.stdout, crlfDelay: Infinity });
      this.rl.on('line', (line) => {
        try {
          const json = JSON.parse(line);
          if (this.waitingResolve) {
            const r = this.waitingResolve;
            this.waitingResolve = null;
            r(json);
          } else {
            this.responseQueue.push(json);
          }
        } catch (e) {}
      });
      this.proc.stderr.on('data', () => {});  // Suppress stderr
      this.proc.on('error', reject);
      this.waitForResponse(60000).then(resolve).catch(reject);
    });
  }

  send(cmd) { this.proc.stdin.write(JSON.stringify(cmd) + '\n'); }

  async waitForResponse(timeout = 120000) {
    return new Promise((resolve, reject) => {
      if (this.responseQueue.length > 0) { resolve(this.responseQueue.shift()); return; }
      const timer = setTimeout(() => { this.waitingResolve = null; reject(new Error('Timeout')); }, timeout);
      this.waitingResolve = (r) => { clearTimeout(timer); resolve(r); };
    });
  }

  async sendAndWait(cmd, timeout = 300000) {
    this.send(cmd);
    while (true) {
      const resp = await this.waitForResponse(timeout);
      if (['done', 'error', 'model_loaded', 'models', 'pong', 'exiting'].includes(resp.status)) return resp;
    }
  }

  async shutdown() {
    if (!this.proc) return;
    try { this.send({ cmd: 'exit' }); await this.waitForResponse(5000); } catch (e) {}
    this.proc.kill();
    this.rl?.close();
  }
}

async function testWorker(name, exePath, args, runs) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`${name}`);
  console.log('='.repeat(60));

  const worker = new DSUWorker(exePath, args);
  const results = { worker: name, runs: [] };

  try {
    await worker.start();
    console.log('  Worker ready');

    for (let i = 0; i < runs.length; i++) {
      const run = runs[i];
      const label = i === 0 ? 'COLD' : 'CACHED';
      process.stdout.write(`  [${label}] ${run.label}... `);
      
      const start = Date.now();
      const resp = await worker.sendAndWait(run.cmd);
      const elapsed = ((Date.now() - start) / 1000).toFixed(2);
      
      const status = resp.status === 'done' ? 'OK' : 'FAIL';
      console.log(`${status} in ${elapsed}s`);
      
      results.runs.push({
        label: run.label,
        type: label,
        time: parseFloat(elapsed),
        status: resp.status,
        error: resp.message
      });
    }

    // Calculate speedup
    if (results.runs.length >= 2 && results.runs[0].time > 0 && results.runs[1].time > 0) {
      results.speedup = (results.runs[0].time / results.runs[1].time).toFixed(1);
    }

  } catch (e) {
    console.log(`  ERROR: ${e.message}`);
    results.error = e.message;
  } finally {
    await worker.shutdown();
  }

  return results;
}

async function main() {
  console.log('='.repeat(60));
  console.log('DSU FINAL PERFORMANCE TEST');
  console.log('='.repeat(60));
  console.log(`Audio: ${path.basename(TEST_AUDIO)}`);
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log('\nExpected times:');
  console.log('  - Cold (with model load): 5-25s');
  console.log('  - Cached: 0.2-1.5s');

  const allResults = [];

  // 1. BSRoformer (SCNet - has config)
  allResults.push(await testWorker(
    'BSRoformer (SCNet 4-stem)',
    path.join(DSU_DIR, 'dsu-bsroformer.exe'),
    ['--worker', '--models-dir', path.join(MODELS_DIR, 'bsroformer'), '--device', 'cuda'],
    [
      { label: 'Model load + process', cmd: { cmd: 'separate', input: TEST_AUDIO, output_dir: path.join(OUTPUT_DIR, 'bsroformer1'), model: 'scnet_xl_ihf', use_fast: true }},
      { label: 'Cached process', cmd: { cmd: 'separate', input: TEST_AUDIO, output_dir: path.join(OUTPUT_DIR, 'bsroformer2'), model: 'scnet_xl_ihf', use_fast: true }},
      { label: 'Cached verify', cmd: { cmd: 'separate', input: TEST_AUDIO, output_dir: path.join(OUTPUT_DIR, 'bsroformer3'), model: 'scnet_xl_ihf', use_fast: true }}
    ]
  ));

  // 2. Demucs
  allResults.push(await testWorker(
    'Demucs (htdemucs)',
    path.join(DSU_DIR, 'dsu-demucs.exe'),
    ['--worker'],
    [
      { label: 'Model load + process', cmd: { cmd: 'separate', input: TEST_AUDIO, output: path.join(OUTPUT_DIR, 'demucs1'), model: 'htdemucs' }},
      { label: 'Cached process', cmd: { cmd: 'separate', input: TEST_AUDIO, output: path.join(OUTPUT_DIR, 'demucs2'), model: 'htdemucs' }},
      { label: 'Cached verify', cmd: { cmd: 'separate', input: TEST_AUDIO, output: path.join(OUTPUT_DIR, 'demucs3'), model: 'htdemucs' }}
    ]
  ));

  // 3. Audio-Separator VR
  allResults.push(await testWorker(
    'Audio-Separator (VR)',
    path.join(DSU_DIR, 'dsu-audio-separator.exe'),
    ['--worker'],
    [
      { label: 'Model load + process', cmd: { cmd: 'separate', input: TEST_AUDIO, output_dir: path.join(OUTPUT_DIR, 'vr1'), model: '3_HP-Vocal-UVR.pth', model_file_dir: path.join(MODELS_DIR, 'audio-separator') }},
      { label: 'Cached process', cmd: { cmd: 'separate', input: TEST_AUDIO, output_dir: path.join(OUTPUT_DIR, 'vr2'), model: '3_HP-Vocal-UVR.pth', model_file_dir: path.join(MODELS_DIR, 'audio-separator') }},
    ]
  ));

  // 4. Apollo
  const apolloModel = path.join(MODELS_DIR, 'apollo', 'apollo_lew_uni.ckpt');
  const apolloConfig = path.join(CONFIGS_DIR, 'config_apollo_lew_uni.yaml');
  if (fs.existsSync(apolloModel) && fs.existsSync(apolloConfig)) {
    allResults.push(await testWorker(
      'Apollo (Restoration)',
      path.join(DSU_DIR, 'dsu-audio-separator.exe'),
      ['--worker'],
      [
        { label: 'Model load + process', cmd: { cmd: 'apollo', input: TEST_AUDIO, output: path.join(OUTPUT_DIR, 'apollo1', 'restored.wav'), model_path: apolloModel, config_path: apolloConfig }},
        { label: 'Cached process', cmd: { cmd: 'apollo', input: TEST_AUDIO, output: path.join(OUTPUT_DIR, 'apollo2', 'restored.wav'), model_path: apolloModel, config_path: apolloConfig }},
      ]
    ));
  } else {
    console.log('\n[SKIP] Apollo - model or config not found');
  }

  // SUMMARY
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY: Cold vs Cached Performance');
  console.log('='.repeat(60));
  console.log('\n  Worker                  | Cold    | Cached  | Speedup');
  console.log('  ' + '-'.repeat(56));

  for (const r of allResults) {
    if (r.error) {
      console.log(`  ${r.worker.padEnd(23)} | ERROR: ${r.error}`);
      continue;
    }
    const cold = r.runs[0]?.time?.toFixed(2) || 'N/A';
    const cached = r.runs[1]?.time?.toFixed(2) || 'N/A';
    const speedup = r.speedup || 'N/A';
    console.log(`  ${r.worker.padEnd(23)} | ${cold.padStart(5)}s | ${cached.padStart(5)}s | ${speedup}x`);
  }

  // Calculate average cached time
  const cachedTimes = allResults
    .filter(r => r.runs?.[1]?.status === 'done')
    .map(r => r.runs[1].time);
  
  if (cachedTimes.length > 0) {
    const avgCached = (cachedTimes.reduce((a, b) => a + b, 0) / cachedTimes.length).toFixed(2);
    console.log('\n  ' + '-'.repeat(56));
    console.log(`  Average cached time: ${avgCached}s`);
  }

  // Write JSON report
  const reportPath = path.join(OUTPUT_DIR, 'performance_report.json');
  fs.writeFileSync(reportPath, JSON.stringify(allResults, null, 2));
  console.log(`\n  Report: ${reportPath}`);
}

main().catch(console.error);
