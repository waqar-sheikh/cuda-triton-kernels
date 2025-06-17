import triton.language as tl
from triton.runtime import driver

def get_kernel_resources(kernel_func, num_warps, *args):
    kernel = kernel_func.warmup(*args, num_warps=num_warps, grid=(1, ))
    kernel._init_handles()
    num_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    return num_regs, size_smem


def get_num_programs(device, thread_regs, size_smem, num_warps):
    properties = driver.active.utils.get_device_properties(device.index)
    NUM_SM = properties["multiprocessor_count"]
    SIZE_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    SIZE_WARP = properties["warpSize"]
    occupancy = SIZE_REGS // (SIZE_WARP * num_warps * thread_regs)
    if size_smem > 0:
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    return num_programs