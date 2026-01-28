/**
 * Test BSRoformer with explicit config path
 * This tests passing model_path + config_path directly (not via models.json)
 */

const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');

const DSU_DIR = path.join(__dirname, '..', 'dsu');
const MODELS_DIR = 'C:\\Users\\soloo\\Documents\\DSU-VSTOPIA\\ThirdPartyApps\\Models\\bsroformer';
const CONFIGS_DIR = path.join(__dirname, '..', 'configs');  // Use bundled configs
const TEST_AUDIO = 'C:\\Users\\soloo\\Documents\\0_20_56_1_27_2026_.wav';
const OUTPUT_DIR = path.join(__dirname, 'config_path_test');

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
      console.log(`Starting ${path.basename(this.exePath)}...`);
      this.proc = spawn(this.exePath, this.args, { stdio: ['pipe', 'pipe', 'pipe'] });
      this.rl = readline.createInterface({ input: this.proc.stdout, crlfDelay: Infinity });
      this.rl.on('line', (line) => {
        try {
          const json = JSON.parse(line);
          console.log(`  [JSON] ${JSON.stringify(json).substring(0, 100)}`);
          if (this.waitingResolve) {
            const r = this.waitingResolve;
            this.waitingResolve = null;
            r(json);
          } else {
            this.responseQueue.push(json);
          }
        } catch (e) {
          console.log(`  [stdout] ${line}`);
        }
      });
      this.proc.stderr.on('data', (data) => {
        const msg = data.toString().trim();
        if (msg && !msg.includes('Warning')) {
          console.log(`  [stderr] ${msg.substring(0, 80)}`);
        }
      });
      this.proc.on('error', reject);
      this.waitForResponse(60000).then(resolve).catch(reject);
    });
  }

  send(cmd) { 
    console.log(`  [SEND] ${JSON.stringify(cmd).substring(0, 100)}`);
    this.proc.stdin.write(JSON.stringify(cmd) + '\n'); 
  }

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

async function main() {
  console.log('='.repeat(60));
  console.log('Test: BSRoformer with explicit config_path');
  console.log('='.repeat(60));
  console.log(`Models dir: ${MODELS_DIR}`);
  console.log(`Configs dir: ${CONFIGS_DIR}`);
  console.log(`Test audio: ${TEST_AUDIO}\n`);

  // Test 1: Using model_path + config_path directly (bypass models.json)
  const worker = new DSUWorker(path.join(DSU_DIR, 'dsu-bsroformer.exe'), [
    '--worker',
    '--device', 'cuda'
    // Note: NOT passing --models-dir, we'll use direct paths
  ]);

  try {
    await worker.start();
    console.log('\nWorker ready!\n');

    // Test with direct paths (model_path + config_path)
    console.log('TEST 1: Direct paths (model_path + config_path)');
    console.log('-----------------------------------------------');
    
    const modelPath = path.join(MODELS_DIR, 'weights', 'bsrofo_sw_fixed.ckpt');
    const configPath = path.join(CONFIGS_DIR, 'config_bsrofo_sw_fixed.yaml');
    
    console.log(`  Model: ${modelPath}`);
    console.log(`  Config: ${configPath}\n`);

    let start = Date.now();
    let result = await worker.sendAndWait({
      cmd: 'separate',
      input: TEST_AUDIO,
      output_dir: path.join(OUTPUT_DIR, 'direct_path'),
      model_path: modelPath,
      config_path: configPath,
      model_type: 'bs_roformer',
      use_fast: true
    });
    let elapsed = ((Date.now() - start) / 1000).toFixed(2);
    
    console.log(`\nResult: ${result.status} in ${elapsed}s`);
    if (result.status === 'done') {
      console.log(`Files: ${result.files?.join(', ')}`);
    } else {
      console.log(`Error: ${result.message}`);
    }

    // Test 2: Cached run (same model)
    if (result.status === 'done') {
      console.log('\nTEST 2: Cached run (same model)');
      console.log('--------------------------------');
      
      start = Date.now();
      result = await worker.sendAndWait({
        cmd: 'separate',
        input: TEST_AUDIO,
        output_dir: path.join(OUTPUT_DIR, 'cached'),
        model_path: modelPath,
        config_path: configPath,
        model_type: 'bs_roformer',
        use_fast: true
      });
      elapsed = ((Date.now() - start) / 1000).toFixed(2);
      
      console.log(`Result: ${result.status} in ${elapsed}s`);
      if (result.files) console.log(`Files: ${result.files?.join(', ')}`);
    }

  } catch (e) {
    console.error(`ERROR: ${e.message}`);
  } finally {
    await worker.shutdown();
  }
}

main().catch(console.error);
