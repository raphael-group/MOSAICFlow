import torch
from pynvml import *


def get_free_gpu() -> int:
    """
    Get index of GPU with least memory usage
    
    Ref
    ----------
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    """
    nvmlInit()
    index = 0
    max = 0
    for i in range(torch.cuda.device_count()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        index = i if info.free > max else index
        max = info.free if info.free > max else max
        
    # seed = np.random.randint(1000)
    # os.system(f'nvidia-smi -q -d Memory |grep Used > gpu-{str(seed)}.tmp')
    # memory_available = [int(x.split()[2]) for x in open('gpu.tmp', 'r').readlines()]
    # os.system(f'rm gpu-{str(seed)}.tmp')
    # print(memory_available)
    return index
