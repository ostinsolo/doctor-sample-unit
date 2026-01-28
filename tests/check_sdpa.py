import torch

cuda_b = torch.backends.cuda

def _call(name):
    fn = getattr(cuda_b, name, None)
    if fn is None:
        return "n/a"
    return fn()

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("flash available", _call("is_flash_sdp_available"))
print("mem_efficient available", _call("is_mem_efficient_sdp_available"))
print("math available", _call("is_math_sdp_available"))
print("flash enabled", _call("flash_sdp_enabled"))
print("mem_efficient enabled", _call("mem_efficient_sdp_enabled"))
print("math enabled", _call("math_sdp_enabled"))
print("has enable_math_sdp", hasattr(cuda_b, "enable_math_sdp"))
