import os
import sys
import traceback


def main() -> int:
    dsu_torch_home = os.environ.get("DSU_TORCH_HOME") or os.environ.get("TORCH_HOME")
    if not dsu_torch_home:
        user_home = os.path.expanduser("~")
        dsu_torch_home = os.path.join(user_home, "Documents", "DSU-VSTOPIA", "ThirdPartyApps", "torch_cache")
    if dsu_torch_home and os.path.isdir(dsu_torch_home):
        os.environ.setdefault("TORCH_HOME", dsu_torch_home)

    import torch

    ckpt_dir = os.path.join(os.environ.get("TORCH_HOME", torch.hub.get_dir()), "hub", "checkpoints")
    print("checkpoints_dir:", ckpt_dir)

    files = [
        "f7e0c4bc-ba3fe64a.th",
        "d12395a8-e57c48e6.th",
        "92cfc3b6-ef3bcb9c.th",
        "04573f0d-f3cf25b2.th",
    ]

    failed = 0
    for fn in files:
        path = os.path.join(ckpt_dir, fn)
        print("\n---", fn, "---")
        if not os.path.exists(path):
            print("MISSING")
            failed += 1
            continue
        try:
            obj = torch.load(path, map_location="cpu")
            # minimal sanity: should be dict-like
            print("OK type:", type(obj).__name__, "keys:", list(obj)[:5] if isinstance(obj, dict) else "<n/a>")
        except Exception as e:
            print("FAIL:", type(e).__name__, e)
            print(traceback.format_exc())
            failed += 1

    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())

