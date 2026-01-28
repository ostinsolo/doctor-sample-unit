const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {};
  for (let i = 0; i < args.length; i += 1) {
    const key = args[i];
    if (!key.startsWith("--")) continue;
    const value = args[i + 1];
    if (value && !value.startsWith("--")) {
      out[key.slice(2)] = value;
      i += 1;
    } else {
      out[key.slice(2)] = true;
    }
  }
  return out;
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function main() {
  const args = parseArgs();
  const input = args.input;
  const modelsDir = args["models-dir"];
  const outputDir = args["output-dir"];
  const pythonExe = args.python;
  const limit = args.limit ? parseInt(args.limit, 10) : 0;

  if (!input || !modelsDir || !outputDir || !pythonExe) {
    console.error("Usage: node run_bsroformer_models.js --input <wav> --models-dir <dir> --output-dir <dir> --python <python.exe> [--limit N] [--overlap N] [--batch-size N] [--use-tta] [--fast]");
    process.exit(2);
  }

  const registryPath = path.join(modelsDir, "models.json");
  if (!fs.existsSync(registryPath)) {
    console.error(`models.json not found: ${registryPath}`);
    process.exit(2);
  }

  const models = JSON.parse(fs.readFileSync(registryPath, "utf-8"));
  const modelNames = Object.keys(models);
  const selected = limit > 0 ? modelNames.slice(0, limit) : modelNames;

  ensureDir(outputDir);
  const reportPath = path.join(outputDir, "report.json");
  const results = [];

  for (const modelName of selected) {
    const modelOutDir = path.join(outputDir, modelName);
    ensureDir(modelOutDir);

    const cmdArgs = [
      path.join(__dirname, "..", "workers", "bsroformer_worker.py"),
      "--model",
      modelName,
      "--models-dir",
      modelsDir,
      "--input_folder",
      input,
      "--store_dir",
      modelOutDir,
    ];

    if (args.overlap) {
      cmdArgs.push("--overlap", String(args.overlap));
    }
    if (args["batch-size"]) {
      cmdArgs.push("--batch-size", String(args["batch-size"]));
    }
    if (args["use-tta"]) {
      cmdArgs.push("--use_tta");
    }
    if (args.fast) {
      cmdArgs.push("--fast");
    }

    const start = Date.now();
    const result = spawnSync(pythonExe, cmdArgs, { stdio: "inherit" });
    const elapsed = Math.round((Date.now() - start) / 1000);

    results.push({
      model: modelName,
      status: result.status === 0 ? "ok" : "error",
      elapsed_s: elapsed,
      exit_code: result.status,
    });

    fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
  }

  console.log(`Report written to: ${reportPath}`);
}

main();
