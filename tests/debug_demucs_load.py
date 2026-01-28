import os
import traceback


def main() -> int:
    # Mirror worker behavior: derived default + env override
    dsu_torch_home = os.environ.get("DSU_TORCH_HOME") or os.environ.get("TORCH_HOME")
    if not dsu_torch_home:
        user_home = os.path.expanduser("~")
        dsu_torch_home = os.path.join(user_home, "Documents", "DSU-VSTOPIA", "ThirdPartyApps", "torch_cache")
    if dsu_torch_home and os.path.isdir(dsu_torch_home):
        os.environ.setdefault("TORCH_HOME", dsu_torch_home)

    import torch

    print("TORCH_HOME:", os.environ.get("TORCH_HOME"))
    try:
        hub_dir = torch.hub.get_dir()
    except Exception as e:
        hub_dir = f"<error: {type(e).__name__}: {e}>"
    print("torch.hub.get_dir():", hub_dir)

    # Patch download to show destination path (if it re-downloads)
    try:
        orig = torch.hub.download_url_to_file

        def _wrapped(url, dst, hash_prefix=None, progress=True):
            print("download_url_to_file:")
            print("  url:", url)
            print("  dst:", dst)
            return orig(url, dst, hash_prefix=hash_prefix, progress=progress)

        torch.hub.download_url_to_file = _wrapped  # type: ignore[assignment]
    except Exception:
        pass

    from demucs.pretrained import get_model

    try:
        print("Loading demucs model: htdemucs_ft")
        m = get_model("htdemucs_ft")
        print("OK:", type(m).__name__)
        return 0
    except Exception as e:
        print("FAILED:", type(e).__name__, e)
        print(traceback.format_exc())
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

