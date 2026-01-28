/**
 * DSU Full Integration Test - Node.js
 * Tests all workers with CUDA: BSRoformer, Demucs, Audio-Separator, Apollo
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const readline = require('readline');

// Configuration
const DSU_DIR = path.join(__dirname, '..', 'dsu');
const MODELS_DIR = 'C:\\Users\\soloo\\Documents\\DSU-VSTOPIA\\ThirdPartyApps\\Models';
const TEST_AUDIO = 'C:\\Users\\soloo\\Documents\\0_20_56_1_27_2026_.wav';
const OUTPUT_DIR = path.join(__dirname, 'node_test_output');
const CONFIGS_DIR = path.join(__dirname, '..', 'configs');

// Ensure output directory exists
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

class DSUWorker {
  constructor(exeName, args = []) {
    this.exePath = path.join(DSU_DIR, exeName);
    this.args = args;
    this.proc = null;
    this.rl = null;
    this.responseQueue = [];
    this.waitingResolve = null;
  }

  async start() {
    return new Promise((resolve, reject) => {
      console.log(`Starting ${path.basename(this.exePath)}...`);
      
      this.proc = spawn(this.exePath, this.args, {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.rl = readline.createInterface({
        input: this.proc.stdout,
        crlfDelay: Infinity
      });

      this.rl.on('line', (line) => {
        try {
          const json = JSON.parse(line);
          if (this.waitingResolve) {
            const resolve = this.waitingResolve;
            this.waitingResolve = null;
            resolve(json);
          } else {
            this.responseQueue.push(json);
          }
        } catch (e) {
          console.log(`  [stdout] ${line}`);
        }
      });

      this.proc.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg && !msg.includes('FutureWarning') && !msg.includes('UserWarning')) {
          console.log(`  [stderr] ${msg}`);
        }
      });

      this.proc.on('error', reject);
      
      // Wait for ready
      this.waitForResponse(30000).then(resolve).catch(reject);
    });
  }

  send(cmd) {
    const json = JSON.stringify(cmd);
    this.proc.stdin.write(json + '\n');
  }

  async waitForResponse(timeout = 60000) {
    return new Promise((resolve, reject) => {
      // Check queue first
      if (this.responseQueue.length > 0) {
        resolve(this.responseQueue.shift());
        return;
      }

      // Wait for next response
      const timer = setTimeout(() => {
        this.waitingResolve = null;
        reject(new Error('Timeout waiting for response'));
      }, timeout);

      this.waitingResolve = (response) => {
        clearTimeout(timer);
        resolve(response);
      };
    });
  }

  async sendAndWait(cmd, timeout = 300000) {
    this.send(cmd);
    
    // Collect responses until we get a terminal status
    const responses = [];
    while (true) {
      const resp = await this.waitForResponse(timeout);
      responses.push(resp);
      
      if (['done', 'error', 'model_loaded', 'models', 'pong', 'exiting'].includes(resp.status)) {
        return resp;
      }
      
      // Log progress
      if (resp.status === 'loading_model') {
        console.log(`    Loading model: ${resp.model}`);
      } else if (resp.status === 'separating' || resp.status === 'restoring') {
        console.log(`    Processing: ${resp.input}`);
      }
    }
  }

  async shutdown() {
    if (!this.proc) return;
    
    try {
      this.send({ cmd: 'exit' });
      await this.waitForResponse(5000);
    } catch (e) {
      // Force kill if graceful shutdown fails
    }
    
    this.proc.kill();
    this.rl.close();
  }
}

async function testBSRoformer() {
  console.log('\n' + '='.repeat(60));
  console.log('TEST: BSRoformer Worker (CUDA)');
  console.log('='.repeat(60));

  const worker = new DSUWorker('dsu-bsroformer.exe', [
    '--worker',
    '--models-dir', path.join(MODELS_DIR, 'bsroformer'),
    '--device', 'cuda'
  ]);

  try {
    const ready = await worker.start();
    console.log(`  Ready: device=${ready.device}, threads=${ready.threads}`);

    // Test ping
    const pong = await worker.sendAndWait({ cmd: 'ping' });
    console.log(`  Ping: ${pong.status}`);

    // Test list models
    const models = await worker.sendAndWait({ cmd: 'list_models' });
    console.log(`  Available models: ${models.models?.slice(0, 5).join(', ')}...`);

    // Test separation with a model
    const modelName = models.models?.[0] || 'bsrofo_sw';
    console.log(`\n  Running separation with model: ${modelName}`);
    
    const startTime = Date.now();
    const result = await worker.sendAndWait({
      cmd: 'separate',
      input: TEST_AUDIO,
      output_dir: path.join(OUTPUT_DIR, 'bsroformer'),
      model: modelName
    });
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

    if (result.status === 'done') {
      console.log(`  SUCCESS: ${result.files?.length || 0} files in ${elapsed}s`);
      console.log(`    Files: ${result.files?.join(', ')}`);
      return { success: true, elapsed, files: result.files };
    } else {
      console.log(`  FAILED: ${result.message}`);
      return { success: false, error: result.message };
    }
  } catch (e) {
    console.log(`  ERROR: ${e.message}`);
    return { success: false, error: e.message };
  } finally {
    await worker.shutdown();
  }
}

async function testDemucs() {
  console.log('\n' + '='.repeat(60));
  console.log('TEST: Demucs Worker (CUDA)');
  console.log('='.repeat(60));

  const worker = new DSUWorker('dsu-demucs.exe', ['--worker']);

  try {
    const ready = await worker.start();
    console.log(`  Ready: device=${ready.device}`);

    // Test separation
    console.log(`\n  Running separation with model: htdemucs`);
    
    const startTime = Date.now();
    const result = await worker.sendAndWait({
      cmd: 'separate',
      input: TEST_AUDIO,
      output: path.join(OUTPUT_DIR, 'demucs'),
      model: 'htdemucs'
    });
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

    if (result.status === 'done') {
      console.log(`  SUCCESS: ${result.files?.length || 0} files in ${elapsed}s`);
      console.log(`    Files: ${result.files?.join(', ')}`);
      return { success: true, elapsed, files: result.files };
    } else {
      console.log(`  FAILED: ${result.message}`);
      return { success: false, error: result.message };
    }
  } catch (e) {
    console.log(`  ERROR: ${e.message}`);
    return { success: false, error: e.message };
  } finally {
    await worker.shutdown();
  }
}

async function testAudioSeparator() {
  console.log('\n' + '='.repeat(60));
  console.log('TEST: Audio-Separator Worker (VR Model)');
  console.log('='.repeat(60));

  const worker = new DSUWorker('dsu-audio-separator.exe', ['--worker']);

  try {
    const ready = await worker.start();
    console.log(`  Ready: ${ready.message}`);

    // Test VR separation
    console.log(`\n  Running VR separation`);
    
    const startTime = Date.now();
    const result = await worker.sendAndWait({
      cmd: 'separate',
      input: TEST_AUDIO,
      output_dir: path.join(OUTPUT_DIR, 'audio_separator'),
      model: '3_HP-Vocal-UVR.pth',
      model_file_dir: path.join(MODELS_DIR, 'audio-separator')
    });
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

    if (result.status === 'done') {
      console.log(`  SUCCESS: ${result.files?.length || 0} files in ${elapsed}s`);
      console.log(`    Files: ${result.files?.join(', ')}`);
      return { success: true, elapsed, files: result.files };
    } else {
      console.log(`  FAILED: ${result.message}`);
      return { success: false, error: result.message };
    }
  } catch (e) {
    console.log(`  ERROR: ${e.message}`);
    return { success: false, error: e.message };
  } finally {
    await worker.shutdown();
  }
}

async function testApollo() {
  console.log('\n' + '='.repeat(60));
  console.log('TEST: Apollo Worker (Restoration)');
  console.log('='.repeat(60));

  const worker = new DSUWorker('dsu-audio-separator.exe', ['--worker']);

  try {
    const ready = await worker.start();
    console.log(`  Ready: ${ready.message}`);

    // Test Apollo restoration
    const apolloModelPath = path.join(MODELS_DIR, 'apollo', 'apollo_lew_uni.ckpt');
    const apolloConfigPath = path.join(CONFIGS_DIR, 'config_apollo_lew_uni.yaml');
    
    if (!fs.existsSync(apolloModelPath)) {
      console.log(`  SKIPPED: Apollo model not found at ${apolloModelPath}`);
      return { success: false, error: 'Model not found', skipped: true };
    }

    console.log(`\n  Running Apollo restoration`);
    
    const startTime = Date.now();
    const result = await worker.sendAndWait({
      cmd: 'apollo',
      input: TEST_AUDIO,
      output: path.join(OUTPUT_DIR, 'apollo', 'restored.wav'),
      model_path: apolloModelPath,
      config_path: apolloConfigPath
    });
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

    if (result.status === 'done') {
      console.log(`  SUCCESS: Restored in ${elapsed}s`);
      console.log(`    Files: ${result.files?.join(', ')}`);
      return { success: true, elapsed, files: result.files };
    } else {
      console.log(`  FAILED: ${result.message}`);
      return { success: false, error: result.message };
    }
  } catch (e) {
    console.log(`  ERROR: ${e.message}`);
    return { success: false, error: e.message };
  } finally {
    await worker.shutdown();
  }
}

async function main() {
  console.log('='.repeat(60));
  console.log('DSU Full Integration Test');
  console.log('='.repeat(60));
  console.log(`DSU Directory: ${DSU_DIR}`);
  console.log(`Models Directory: ${MODELS_DIR}`);
  console.log(`Test Audio: ${TEST_AUDIO}`);
  console.log(`Output Directory: ${OUTPUT_DIR}`);

  // Check prerequisites
  if (!fs.existsSync(TEST_AUDIO)) {
    console.error(`ERROR: Test audio not found: ${TEST_AUDIO}`);
    process.exit(1);
  }

  const results = {};

  // Run all tests
  results.bsroformer = await testBSRoformer();
  results.demucs = await testDemucs();
  results.audioSeparator = await testAudioSeparator();
  results.apollo = await testApollo();

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('SUMMARY');
  console.log('='.repeat(60));

  const summary = [
    ['BSRoformer', results.bsroformer],
    ['Demucs', results.demucs],
    ['Audio-Separator', results.audioSeparator],
    ['Apollo', results.apollo]
  ];

  for (const [name, result] of summary) {
    const status = result.success ? 'PASS' : (result.skipped ? 'SKIP' : 'FAIL');
    const time = result.elapsed ? ` (${result.elapsed}s)` : '';
    console.log(`  ${name}: ${status}${time}`);
  }

  const passCount = summary.filter(([, r]) => r.success).length;
  const skipCount = summary.filter(([, r]) => r.skipped).length;
  console.log(`\nTotal: ${passCount}/${summary.length - skipCount} passed`);

  // Write results
  const reportPath = path.join(OUTPUT_DIR, 'test_report.json');
  fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
  console.log(`\nReport saved to: ${reportPath}`);
}

main().catch(console.error);
