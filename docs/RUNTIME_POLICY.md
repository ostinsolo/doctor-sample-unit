## Runtime Policy (optional)

Both the shared runtime and standalone workers can read a single policy file
to keep cache and performance settings consistent without coupling processes.

Default path:
- `~/.dsu/runtime_policy.json`

Override path:
- `DSU_RUNTIME_POLICY_PATH=/abs/path/runtime_policy.json`

Example:
```json
{
  "global": {
    "max_cached_models": 1
  },
  "worker": {
    "disable_conflict_stopping": false
  },
  "demucs": {
    "max_cached_models": 2
  },
  "bsroformer": {
    "max_cached_models": 1
  },
  "audiosep": {
    "max_cached_models": 1,
    "use_torch_stft": "auto",
    "auto_stft_seconds": 60,
    "mmap": true
  }
}
```

### worker.disable_conflict_stopping (Node.js WorkerManager)

On GPU (MPS/CUDA), the WorkerManager stops conflicting workers before starting a new one to avoid OOM on 8GB VRAM (e.g. BSRoformer + AudioSep = ~6–9GB). That causes a full process restart when switching architectures (bsroformer ↔ audio_separator), adding 5–10+ seconds per switch.

For **16GB+ unified memory (M1 Pro/Max, M2/M3 Pro/Max)** or systems with more VRAM, set:

```json
{
  "worker": {
    "disable_conflict_stopping": true
  }
}
```

Workers will then stay alive when you switch between models/architectures. Switching becomes much faster (no stop + restart). **Risk:** If you run multiple heavy workers at once, you may hit OOM. Use only if you have sufficient memory.

## Resource-Aware System (Orchestrator)

The orchestrator detects system resources and updates `runtime_policy.json` with `system_report` and `resource_tier`. Implementation: `orchestrator/system_report.py`. Run: `cd orchestrator && python -m system_report` or `python -m system_report --force`. See [RESOURCE_AWARE_SYSTEM_SPEC.md](RESOURCE_AWARE_SYSTEM_SPEC.md).
