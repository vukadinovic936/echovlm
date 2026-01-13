import os
import torch
import torch.distributed as dist
import logging
import pathlib
import re

logger = logging.getLogger(__name__)
def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)
def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device) # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device
def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type
def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp():
        dist.destroy_process_group()

def get_latest_checkpoint(base_dir="assets"):
    base_path = pathlib.Path(base_dir)
    
    # Find all directories starting with 'echovlm'
    folders = []
    for f in base_path.iterdir():
        if f.is_dir() and f.name.startswith("echovlm"):
            # Extract number: 'echovlm12' -> 12. If no number (just 'echovlm'), use 0.
            match = re.search(r'echovlm(\d+)', f.name)
            version = int(match.group(1)) if match and match.group(1) else 0
            folders.append((version, f))

    if not folders:
        return None

    # Sort by version number and pick the last one
    latest_folder = sorted(folders, key=lambda x: x[0])[-1][1]

    # Find the first (and only) .pt file inside
    checkpoint_files = list(latest_folder.glob("*.pt"))
    
    return str(checkpoint_files[0]) if checkpoint_files else None