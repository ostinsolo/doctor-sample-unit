import json
import os


def _policy_path():
    return os.path.expanduser(
        os.environ.get("DSU_RUNTIME_POLICY_PATH", "~/.dsu/runtime_policy.json")
    )


def load_runtime_policy():
    path = _policy_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def get_policy_value(policy, section, key, default=None):
    if not isinstance(policy, dict):
        return default
    section_values = policy.get(section, {})
    if isinstance(section_values, dict) and key in section_values:
        return section_values[key]
    global_values = policy.get("global", {})
    if isinstance(global_values, dict) and key in global_values:
        return global_values[key]
    return default
