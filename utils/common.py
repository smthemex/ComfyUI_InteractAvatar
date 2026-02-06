from contextlib import contextmanager
import gc
import time

import torch
import deepspeed.comm.comm as dist
# import imageio
from safetensors import safe_open


DTYPE_MAP = {'float32': torch.float32, 
    'float16': torch.float16, 
    'bfloat16': torch.bfloat16, 
    'float8': torch.float8_e4m3fn
}
# VIDEO_EXTENSIONS = set(x.extension for x in imageio.config.video_extensions)
AUTOCAST_DTYPE = None


def get_rank():
    return dist.get_rank()

# is_main_process: check if current process is the main process
def is_main_process():
    return get_rank() == 0

# zero_first: zero first in distributed training
@contextmanager
def zero_first():
    if not is_main_process():
        dist.barrier()
    yield
    if is_main_process():
        dist.barrier()

# empty_cuda_cache: empty cuda cache
def empty_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def log_duration(name):
    start = time.time()
    try:
        yield
    finally:
        print(f'{name}: {time.time()-start:.3f}')

# load_safetensors: load safetensors file
def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors
