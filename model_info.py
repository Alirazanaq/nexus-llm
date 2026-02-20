#!/usr/bin/env python3
"""
NEXUS-LLM Converter — HuggingFace Model Info Module
Fetches model metadata, estimates sizes, and validates model links.
"""
import re
import requests
import json


HF_API = "https://huggingface.co/api/models"


def parse_model_id(user_input):
    """
    Parse a HuggingFace model ID from various input formats:
    - "meta-llama/Llama-3.1-8B"
    - "https://huggingface.co/meta-llama/Llama-3.1-8B"
    - "huggingface.co/meta-llama/Llama-3.1-8B"
    """
    user_input = user_input.strip()

    # Remove URL prefix
    patterns = [
        r"https?://huggingface\.co/([^/]+/[^/\s?#]+)",
        r"huggingface\.co/([^/]+/[^/\s?#]+)",
    ]
    for pat in patterns:
        m = re.search(pat, user_input)
        if m:
            return m.group(1).rstrip("/")

    # Assume direct model ID format: "org/model"
    if "/" in user_input and len(user_input.split("/")) == 2:
        return user_input

    return None


def get_model_info(model_id):
    """
    Fetch model info from HuggingFace API.
    Returns dict with model metadata or None on failure.
    """
    try:
        resp = requests.get(f"{HF_API}/{model_id}", timeout=15)
        if resp.status_code == 401:
            return {"error": "Model is gated. You may need a HuggingFace token."}
        if resp.status_code == 404:
            return {"error": f"Model '{model_id}' not found on HuggingFace."}
        if resp.status_code != 200:
            return {"error": f"HuggingFace API error: {resp.status_code}"}
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "HuggingFace API request timed out. Check your internet."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot reach HuggingFace. Check your internet connection."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def estimate_model_size(model_info):
    """
    Estimate model size from HuggingFace metadata.
    Returns dict with estimated parameters and sizes.
    """
    result = {
        "model_id": model_info.get("modelId", model_info.get("id", "unknown")),
        "author": model_info.get("author", "unknown"),
        "pipeline_tag": model_info.get("pipeline_tag", "unknown"),
        "tags": model_info.get("tags", []),
        "license": model_info.get("cardData", {}).get("license", "unknown") if model_info.get("cardData") else "unknown",
        "params_billion": None,
        "fp16_size_gb": None,
        "q4km_size_gb": None,
        "q8_size_gb": None,
        "is_quantized": False,
        "is_gguf": False,
        "quant_method": None,
        "downloads": model_info.get("downloads", 0),
        "likes": model_info.get("likes", 0),
    }

    # Check if it's already quantized
    tags = [t.lower() for t in result["tags"]]
    model_id_lower = result["model_id"].lower()

    if any(t in tags for t in ["gguf", "ggml"]) or "gguf" in model_id_lower:
        result["is_gguf"] = True
        result["is_quantized"] = True
    if any(t in tags for t in ["gptq", "awq", "bnb", "int4", "int8"]):
        result["is_quantized"] = True
        for t in tags:
            if t in ["gptq", "awq"]:
                result["quant_method"] = t.upper()

    # Try to extract param count from safetensors metadata
    safetensors = model_info.get("safetensors")
    if safetensors and isinstance(safetensors, dict):
        params_info = safetensors.get("parameters")
        if params_info and isinstance(params_info, dict):
            total_params = sum(params_info.values())
            result["params_billion"] = round(total_params / 1e9, 2)

    # Fallback: try to infer from model name
    if result["params_billion"] is None:
        name = result["model_id"].lower()
        param_patterns = [
            (r"(\d+\.?\d*)b", 1.0),    # "8B", "70B", "1.5B"
            (r"(\d+\.?\d*)m", 0.001),   # "350M"
        ]
        for pat, multiplier in param_patterns:
            m = re.search(pat, name)
            if m:
                val = float(m.group(1)) * multiplier
                if 0.1 <= val <= 1000:  # sanity range
                    result["params_billion"] = round(val, 2)
                    break

    # Calculate sizes
    if result["params_billion"]:
        p = result["params_billion"]
        result["fp16_size_gb"] = round(p * 2, 1)        # 2 bytes per param
        result["q4km_size_gb"] = round(p * 0.5625, 1)   # ~4.5 bits per param
        result["q8_size_gb"] = round(p * 1.0625, 1)     # ~8.5 bits per param

    return result


def get_model_files(model_id, token=None):
    """
    List files in a HuggingFace repo to estimate actual download size.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(
            f"{HF_API}/{model_id}",
            params={"blobs": True},
            headers=headers,
            timeout=15
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        siblings = data.get("siblings", [])
        total_size = 0
        files = []
        for f in siblings:
            size = f.get("size", 0) or 0
            total_size += size
            files.append({
                "filename": f.get("rfilename", "unknown"),
                "size_mb": round(size / (1024**2), 1)
            })

        return {
            "total_size_gb": round(total_size / (1024**3), 2),
            "file_count": len(files),
            "files": sorted(files, key=lambda x: x["size_mb"], reverse=True)[:20],  # top 20 largest
        }
    except Exception:
        return None


def estimate_download_time(size_gb, speed_mbps=50):
    """Estimate download time given size in GB and internet speed in Mbps."""
    size_mb = size_gb * 1024
    speed_mb_per_sec = speed_mbps / 8  # Convert Mbps to MB/s
    seconds = size_mb / max(speed_mb_per_sec, 0.1)

    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.0f} minutes"
    else:
        hours = seconds / 3600
        mins = (seconds % 3600) / 60
        return f"{hours:.0f}h {mins:.0f}m"
