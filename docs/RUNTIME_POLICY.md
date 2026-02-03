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
