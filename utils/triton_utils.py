import torch
import triton.language as tl
from triton.runtime import driver

_device_properties = None
_kernel_cache = {}

def get_precompiled_kernel(cache_key, kernel_func, num_warps, *args):
    global _kernel_cache
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]

    kernel = kernel_func.warmup(*args, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    num_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    _kernel_cache[cache_key] = (kernel, num_regs, size_smem)
    return kernel, num_regs, size_smem


def get_num_programs(device, thread_regs, size_smem, num_warps):
    global _device_properties
    if _device_properties is None:
        _device_properties = driver.active.utils.get_device_properties(device.index)

    NUM_SM = _device_properties["multiprocessor_count"]
    SIZE_REGS = _device_properties["max_num_regs"]
    SIZE_SMEM = _device_properties["max_shared_mem"]
    SIZE_WARP = _device_properties["warpSize"]

    occupancy = SIZE_REGS // (SIZE_WARP * num_warps * thread_regs)
    if size_smem > 0:
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    return num_programs