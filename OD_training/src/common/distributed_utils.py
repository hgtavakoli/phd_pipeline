# cl_src/common/distributed_utils.py

import os
import torch
import torch.distributed as dist
import random
import numpy as np

def is_master_process():
    """
    Returns True if this is the main process.
    Works for both single-GPU (rank -1) and multi-GPU (rank 0) setups.
    """
    return get_rank() in [-1, 0]

def get_world_size():
    """Returns the total number of processes in the distributed group."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def get_rank():
    """Returns the rank of the current process, or -1 if not in a distributed setup."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return -1

def set_random_seeds(seed=42):
    """Sets random seeds for reproducibility across all relevant libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if is_master_process():
        print(f"Random seeds set to {seed}")