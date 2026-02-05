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

## Resource-Aware System (Orchestrator)

The orchestrator detects system resources and updates `runtime_policy.json` with `system_report` and `resource_tier`. Implementation: `orchestrator/system_report.py`. Run: `cd orchestrator && python -m system_report` or `python -m system_report --force`. See [RESOURCE_AWARE_SYSTEM_SPEC.md](RESOURCE_AWARE_SYSTEM_SPEC.md).
