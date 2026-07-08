"""
utils/system_detection.py — Hardware detection for system capabilities (CPUs, RAM, GPUs, and VRAM)
"""

import os
import shutil
import subprocess
import platform
import ctypes
from typing import List, Tuple

def get_cpu_cores() -> int:
    """Return the total number of CPU logical cores available on the system."""
    cores = os.cpu_count()
    return cores if cores else 1

def get_available_system_ram_gb() -> float:
    """Query available system RAM in gigabytes."""
    # 1. Try psutil if available
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass

    # 2. Platform-specific fallbacks
    system = platform.system()
    if system == "Windows":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_uint64),
                    ('ullAvailPhys', ctypes.c_uint64),
                    ('ullTotalPageFile', ctypes.c_uint64),
                    ('ullAvailPageFile', ctypes.c_uint64),
                    ('ullTotalVirtual', ctypes.c_uint64),
                    ('ullAvailVirtual', ctypes.c_uint64),
                    ('ullAvailExtendedVirtual', ctypes.c_uint64),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullAvailPhys / (1024 ** 3)
        except Exception:
            pass
    elif system == "Linux":
        # Parse /proc/meminfo
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        mem_kb = int(line.split()[1])
                        return mem_kb / (1024 * 1024)
        except Exception:
            pass

    # Safe fallback
    return 8.0

def get_available_gpus_and_vram() -> List[Tuple[int, float]]:
    """
    Detect available CUDA GPUs and query their available (free) VRAM in GB.
    Returns a list of (gpu_id, available_vram_gb) tuples.
    """
    # 1. Try torch.cuda if PyTorch is loaded
    try:
        import torch
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                gpus.append((i, free_bytes / (1024 ** 3)))
            return gpus
    except (ImportError, Exception):
        pass

    # 2. Try nvidia-smi CLI
    if shutil.which("nvidia-smi") is not None:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
            )
            gpus = []
            for line in res.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split(",")
                gpu_id = int(parts[0].strip())
                free_vram_mb = float(parts[1].strip())
                gpus.append((gpu_id, free_vram_mb / 1024.0))
            return gpus
        except Exception:
            pass

    return []
