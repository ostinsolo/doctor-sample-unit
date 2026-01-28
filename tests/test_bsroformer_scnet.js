/**
 * Test BSRoformer with SCNet model (config exists!)
 */

const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');

const DSU_DIR = path.join(__dirname, '..', 'dsu');
const MODELS_DIR = 'C:\\Users\\soloo\\Documents\\DSU-VSTOPIA\\ThirdPartyApps\\Models';
const TEST_AUDIO = 'C:\\Users\\soloo\\Documents\\0_20_56_1_27_2026_.wav';
const OUTPUT_DIR = path.join(__dirname, 'scnet_output');

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
        } catch (e) {}
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
    this.proc.stdin.write(JSON.stringify(cmd) + '\n');
  }

  async waitForResponse(timeout = 120000) {
    return new Promise((resolve, reject) => {
      if (this.responseQueue.length > 0) {
        resolve(this.responseQueue.shift());
        return;
      }
      const timer = setTimeout(() => {
        this.waitingResolve = null;
        reject(new Error('Timeout'));
      }, timeout);
      this.waitingResolve = (response) => {
        clearTimeout(timer);
        resolve(response);
      };
    });
  }

  async sendAndWait(cmd, timeout = 300000) {
    this.send(cmd);
    while (true) {
      const resp = await this.waitForResponse(timeout);
      if (['done', 'error', 'model_loaded', 'models', 'pong', 'exiting'].includes(resp.status)) {
        return resp;
      }
      if (resp.status === 'loading_model') {
        console.log(`    Loading model: ${resp.model}`);
      } else if (resp.status === 'separating') {
        console.log(`    Separating...`);
      }
    }
  }

  async shutdown() {
    if (!this.proc) return;
    try {
      this.send({ cmd: 'exit' });
      await this.waitForResponse(5000);
    } catch (e) {}
    this.proc.kill();
    this.rl.close();
  }
}

async function main() {
  console.log('='.repeat(60));
  console.log('BSRoformer SCNet Performance Test');
  console.log('='.repeat(60));
  console.log(`Using model: scnet_xl_ihf (config exists!)\n`);

  const worker = new DSUWorker(path.join(DSU_DIR, 'dsu-bsroformer.exe'), [
    '--worker',
    '--models-dir', path.join(MODELS_DIR, 'bsroformer'),
    '--device', 'cuda'
  ]);

  try {
    await worker.start();
    console.log('Worker ready!\n');

    // COLD start (model loading)
    console.log('[RUN 1] COLD START (model loading)...');
    let start = Date.now();
    let result = await worker.sendAndWait({
      cmd: 'separate',
      input: TEST_AUDIO,
      output_dir: path.join(OUTPUT_DIR, 'run1'),
      model: 'scnet_xl_ihf',
      use_fast: true
    });
    const coldTime = ((Date.now() - start) / 1000).toFixed(2);
    console.log(`  Result: ${result.status} in ${coldTime}s`);
    if (result.status === 'error') {
      console.log(`  Error: ${result.message}`);
    } else {
      console.log(`  Files: ${result.files?.join(', ')}`);
    }

    // CACHED (same model)
    console.log('\n[RUN 2] CACHED (same model)...');
    start = Date.now();
    result = await worker.sendAndWait({
      cmd: 'separate',
      input: TEST_AUDIO,
      output_dir: path.join(OUTPUT_DIR, 'run2'),
      model: 'scnet_xl_ihf',
      use_fast: true
    });
    const cachedTime = ((Date.now() - start) / 1000).toFixed(2);
    console.log(`  Result: ${result.status} in ${cachedTime}s`);
    if (result.files) console.log(`  Files: ${result.files?.join(', ')}`);

    // CACHED again
    console.log('\n[RUN 3] CACHED (verification)...');
    start = Date.now();
    result = await worker.sendAndWait({
      cmd: 'separate',
      input: TEST_AUDIO,
      output_dir: path.join(OUTPUT_DIR, 'run3'),
      model: 'scnet_xl_ihf',
      use_fast: true
    });
    const cached2Time = ((Date.now() - start) / 1000).toFixed(2);
    console.log(`  Result: ${result.status} in ${cached2Time}s`);
    if (result.files) console.log(`  Files: ${result.files?.join(', ')}`);

    console.log('\n' + '='.repeat(60));
    console.log('RESULTS');
    console.log('='.repeat(60));
    console.log(`  Cold (model load + process): ${coldTime}s`);
    console.log(`  Cached (process only):       ${cachedTime}s`);
    console.log(`  Cached (verification):       ${cached2Time}s`);
    console.log(`  Speedup: ${(coldTime / cachedTime).toFixed(1)}x`);

  } catch (e) {
    console.error(`ERROR: ${e.message}`);
  } finally {
    await worker.shutdown();
  }
}

main().catch(console.error);
