#!/usr/bin/env python3
"""
NEXUS-LLM Converter — System Hardware Detection Module
Detects RAM, CPU, GPU, VRAM, disk space and evaluates capability.
"""
import platform
import os
import shutil
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import psutil
import subprocess
import json


def get_cpu_info():
    """Detect CPU details."""
    info = {
        "name": platform.processor() or "Unknown",
        "cores_physical": psutil.cpu_count(logical=False) or 0,
        "cores_logical": psutil.cpu_count(logical=True) or 0,
        "freq_mhz": 0,
        "arch": platform.machine(),
        "has_avx2": False,
        "has_avx512": False,
    }
    try:
        freq = psutil.cpu_freq()
        if freq:
            info["freq_mhz"] = int(freq.max or freq.current or 0)
    except Exception:
        pass

    # Detect AVX support on x86
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            info["name"] = cpu_name.strip()
            winreg.CloseKey(key)
        except Exception:
            pass
        # Check AVX via wmic or assume based on gen
        try:
            r = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=5)
            if r.stdout:
                name_line = [l.strip() for l in r.stdout.strip().split("\n") if l.strip() and "Name" not in l]
                if name_line:
                    info["name"] = name_line[0]
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    info["name"] = line.split(":")[1].strip()
                    break
            if "avx2" in cpuinfo.lower():
                info["has_avx2"] = True
            if "avx512" in cpuinfo.lower():
                info["has_avx512"] = True
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True, timeout=5)
            if r.stdout:
                info["name"] = r.stdout.strip()
        except Exception:
            pass

    # Heuristic AVX detection for Windows (based on CPU name)
    cpu_lower = info["name"].lower()
    if any(g in cpu_lower for g in ["11th gen", "12th gen", "13th gen", "14th gen", "core ultra"]):
        info["has_avx2"] = True
    if any(g in cpu_lower for g in ["11th gen", "12th gen", "xeon w-3", "xeon w-2"]):
        info["has_avx512"] = True
    if "zen" in cpu_lower or "ryzen" in cpu_lower or "epyc" in cpu_lower:
        info["has_avx2"] = True
    if "zen4" in cpu_lower or "zen 4" in cpu_lower or "7x4" in cpu_lower or "9x5" in cpu_lower:
        info["has_avx512"] = True

    return info


def get_ram_info():
    """Detect total and available RAM."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 1),
        "available_gb": round(mem.available / (1024**3), 1),
        "used_gb": round(mem.used / (1024**3), 1),
        "percent_used": mem.percent,
    }


def get_gpu_info():
    """Detect NVIDIA GPU(s) via pynvml."""
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "index": i,
                "name": name,
                "vram_total_gb": round(mem_info.total / (1024**3), 1),
                "vram_free_gb": round(mem_info.free / (1024**3), 1),
                "vram_used_gb": round(mem_info.used / (1024**3), 1),
            })
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # Fallback: try nvidia-smi
    if not gpus:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0 and r.stdout.strip():
                for i, line in enumerate(r.stdout.strip().split("\n")):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append({
                            "index": i,
                            "name": parts[0],
                            "vram_total_gb": round(float(parts[1]) / 1024, 1),
                            "vram_free_gb": round(float(parts[2]) / 1024, 1),
                            "vram_used_gb": round((float(parts[1]) - float(parts[2])) / 1024, 1),
                        })
        except Exception:
            pass

    return gpus


def get_disk_info(path=None):
    """Detect disk space at the given path or current drive."""
    if path is None:
        path = os.path.expanduser("~")
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gb": round(usage.total / (1024**3), 1),
        "free_gb": round(usage.free / (1024**3), 1),
        "used_gb": round(usage.used / (1024**3), 1),
    }


def get_full_system_info():
    """Return complete system profile."""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "gpus": get_gpu_info(),
        "disk": get_disk_info(),
    }


def evaluate_capability(system_info, model_size_gb_fp16, model_params_b):
    """
    Evaluate what this system can do with a given model.
    Returns a dict of capabilities.
    """
    ram = system_info["ram"]["available_gb"]
    total_ram = system_info["ram"]["total_gb"]
    gpus = system_info["gpus"]
    gpu_vram = max((g["vram_total_gb"] for g in gpus), default=0)
    disk_free = system_info["disk"]["free_gb"]

    q4km_size = model_size_gb_fp16 * 0.28  # Q4_K_M ~28% of FP16
    q8_size = model_size_gb_fp16 * 0.53    # Q8 ~53% of FP16
    residuals_size = model_size_gb_fp16     # DecDEC residuals = FP16 size
    eagle3_head_size = 0.3                  # ~300 MB
    total_download = model_size_gb_fp16     # FP16 download
    total_nexus_disk = q4km_size + residuals_size + eagle3_head_size + 1.5  # +config etc

    # RAM available for model (after OS)
    os_overhead = 2.5 if "Windows" in system_info["os"] else 1.2
    usable_ram = total_ram - os_overhead
    kv_budget = min(2.5, usable_ram * 0.15)  # 15% of RAM for KV, max 2.5 GB

    # Can run FP16?
    can_fp16_ram = usable_ram >= (model_size_gb_fp16 + kv_budget + 1.0)
    can_fp16_gpu = gpu_vram >= (model_size_gb_fp16 + 1.0)

    # Can run Q4_K_M?
    can_q4km_ram = usable_ram >= (q4km_size + kv_budget + 2.0)  # +2 for DecDEC + EAGLE-3
    can_q4km_gpu = gpu_vram >= q4km_size
    can_q4km_split = (gpu_vram + usable_ram) >= (q4km_size + kv_budget + 2.0)

    # Can run Q8?
    can_q8_ram = usable_ram >= (q8_size + kv_budget + 1.5)

    # Disk space check
    has_disk_for_download = disk_free >= total_download * 1.2  # 20% margin
    has_disk_for_nexus = disk_free >= total_nexus_disk * 1.2

    # Training capability (needs more VRAM)
    can_train_eagle3 = gpu_vram >= 8  # Minimum 8 GB VRAM for EAGLE-3 training
    can_train_local = gpu_vram >= 16  # Comfortable training

    # Estimate TPS
    ram_bw = 25 if "ddr4" in system_info["cpu"]["name"].lower() or total_ram <= 16 else 40  # rough heuristic
    if any("apple" in system_info["cpu"]["name"].lower() or "m1" in system_info["cpu"]["name"].lower()
           or "m2" in system_info["cpu"]["name"].lower() for _ in [1]):
        ram_bw = 100

    est_tps_cpu = round(ram_bw / max(q4km_size * 0.75, 0.1), 1) if can_q4km_ram else 0
    est_tps_gpu = round(min(gpu_vram * 30, 80), 1) if can_q4km_gpu else 0  # rough estimate

    # Determine best config
    if can_fp16_gpu:
        best = "FP16_GPU"
        best_quality = "100%"
        best_tps = f"{min(gpu_vram * 30, 120):.0f}"
    elif can_fp16_ram:
        best = "FP16_CPU"
        best_quality = "100%"
        best_tps = f"{ram_bw / max(model_size_gb_fp16 * 0.75, 0.1):.0f}"
    elif can_q4km_gpu:
        best = "Q4_K_M_GPU"
        best_quality = "98-99%"
        best_tps = f"{est_tps_gpu:.0f}"
    elif can_q4km_split and gpu_vram > 0:
        best = "Q4_K_M_SPLIT"
        best_quality = "98-99%"
        best_tps = f"{max(est_tps_cpu, est_tps_gpu * 0.5):.0f}"
    elif can_q4km_ram:
        best = "Q4_K_M_CPU"
        best_quality = "98-99%"
        best_tps = f"{est_tps_cpu:.0f}"
    else:
        best = "TOO_LARGE"
        best_quality = "N/A"
        best_tps = "0"

    return {
        "model_fp16_gb": round(model_size_gb_fp16, 1),
        "model_q4km_gb": round(q4km_size, 1),
        "model_q8_gb": round(q8_size, 1),
        "model_params_b": model_params_b,
        "residuals_gb": round(residuals_size, 1),
        "total_nexus_disk_gb": round(total_nexus_disk, 1),
        "total_download_gb": round(total_download, 1),
        "usable_ram_gb": round(usable_ram, 1),
        "gpu_vram_gb": gpu_vram,
        "can_run": {
            "fp16_gpu": can_fp16_gpu,
            "fp16_ram": can_fp16_ram,
            "q8_ram": can_q8_ram,
            "q4km_gpu": can_q4km_gpu,
            "q4km_ram": can_q4km_ram,
            "q4km_split": can_q4km_split,
        },
        "can_train": {
            "eagle3_local": can_train_eagle3,
            "full_local": can_train_local,
            "kaggle_t4": True,  # Always possible on Kaggle
        },
        "disk_ok": {
            "download": has_disk_for_download,
            "nexus_full": has_disk_for_nexus,
        },
        "best_config": best,
        "best_quality": best_quality,
        "est_tps": best_tps,
        "est_tps_with_eagle3": f"{float(best_tps) * 2.5:.0f}" if best_tps != "0" else "0",
    }
