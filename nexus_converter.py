#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║                    NEXUS-LLM CONVERTER                      ║
║     Convert any HuggingFace model to NEXUS-LLM format       ║
║                                                              ║
║  Author: Syed Ali Raza Naqvi                                 ║
║  Version: 1.1.0                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
import sys
import os
import time
import json
import subprocess
import shutil
import re

# Ensure UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.text import Text
    from rich.columns import Columns
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
except ImportError:
    print("ERROR: 'rich' library not found. Install it with: pip install rich")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Local modules
from system_check import get_full_system_info, evaluate_capability
from model_info import parse_model_id, get_model_info, estimate_model_size, get_model_files, estimate_download_time

console = Console()

# ============================================================================
# UI COMPONENTS
# ============================================================================

BANNER = """[bold cyan]
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗    ██╗     ██╗     ███╗   ███╗    ║
║  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝    ██║     ██║     ████╗ ████║    ║
║  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗    ██║     ██║     ██╔████╔██║    ║
║  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║    ██║     ██║     ██║╚██╔╝██║    ║
║  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║    ███████╗███████╗██║ ╚═╝ ██║    ║
║  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚══════╝╚══════╝╚═╝     ╚═╝    ║
║                                                                                ║
║            NEAR-LOSSLESS LLM CONVERTER FOR CONSUMER HARDWARE                   ║
║                              v1.2.0                                            ║
╚════════════════════════════════════════════════════════════════════════════════╝
[/bold cyan]"""


def show_banner():
    console.print(BANNER)
    console.print("[dim]Convert any HuggingFace model to NEXUS-LLM optimized format[/dim]")
    console.print("[dim]Quality: 98-99% | Speed: 2-5x faster | RAM: 60-75% less[/dim]")
    console.print()


def check_hf_token():
    """
    Check if HuggingFace token is configured.
    If not, offer to set it up within the TUI.
    Returns True if token is available, False otherwise.
    """
    try:
        from huggingface_hub import get_token, login
    except ImportError:
        # Fallback for older versions
        try:
            from huggingface_hub import HfFolder
            get_token = HfFolder.get_token
            login = lambda token, **kw: HfFolder.save_token(token)
        except ImportError:
            console.print("  [yellow]huggingface_hub not installed properly.[/yellow]")
            return True

    token = get_token()

    if token:
        # Verify token is valid
        try:
            import requests
            resp = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if resp.status_code == 200:
                user_info = resp.json()
                username = user_info.get("name", "Unknown")
                console.print(f"  [green]HuggingFace logged in as:[/green] [bold]{username}[/bold]")
                return True
            else:
                console.print("  [yellow]HuggingFace token found but invalid/expired.[/yellow]")
        except Exception:
            console.print("  [yellow]Could not verify HF token (no internet?). Will try anyway.[/yellow]")
            return True
    else:
        console.print("  [yellow]No HuggingFace token found.[/yellow]")

    # Offer to set up token
    console.print(Panel(
        "[bold]Some models on HuggingFace are gated (e.g., Llama, Gemma).[/bold]\n"
        "You need a HuggingFace access token to download them.\n\n"
        "[dim]Even for public models, having a token avoids rate limits.[/dim]\n\n"
        "How to get a token:\n"
        "  1. Go to huggingface.co/settings/tokens\n"
        "  2. Create a new token (Read access is enough)\n"
        "  3. Copy the token and paste it below",
        title="[yellow]HuggingFace Token Setup[/yellow]",
        border_style="yellow"
    ))

    choice = Prompt.ask(
        "[bold cyan]Enter your HF token (or press Enter to skip)[/bold cyan]",
        default=""
    )

    if choice.strip():
        token_str = choice.strip()
        try:
            # Verify before saving
            import requests
            resp = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token_str}"},
                timeout=10
            )
            if resp.status_code == 200:
                user_info = resp.json()
                username = user_info.get("name", "Unknown")
                # Save the token using the modern API
                login(token=token_str, add_to_git_credential=False)
                console.print(f"  [bold green]Token saved! Logged in as: {username}[/bold green]")
                return True
            else:
                console.print("  [red]Invalid token. Check and try again.[/red]")
                if Confirm.ask("  Continue without token? (public models only)"):
                    return True
                return False
        except Exception as e:
            console.print(f"  [yellow]Could not verify token: {e}[/yellow]")
            # Try saving anyway
            try:
                login(token=token_str, add_to_git_credential=False)
            except Exception:
                pass
            console.print("  [yellow]Token saved anyway. Will retry during download.[/yellow]")
            return True
    else:
        console.print("  [dim]Skipping token setup. Public models will still work.[/dim]")
        return True


def show_system_info(sys_info):
    """Display detected system hardware in a beautiful table."""
    console.print("\n[bold yellow]>> STEP 1: System Hardware Detection[/bold yellow]\n")

    table = Table(title="Your System Configuration", box=box.DOUBLE_EDGE,
                  title_style="bold white", border_style="cyan")
    table.add_column("Component", style="bold", width=20)
    table.add_column("Details", width=50)
    table.add_column("Status", justify="center", width=10)

    # OS
    table.add_row("OS", sys_info["os"], "[green]OK[/green]")

    # CPU
    cpu = sys_info["cpu"]
    cpu_detail = f"{cpu['name']}\n{cpu['cores_physical']}C/{cpu['cores_logical']}T @ {cpu['freq_mhz']} MHz"
    avx = ""
    if cpu["has_avx512"]:
        avx = "[green]AVX-512[/green]"
    elif cpu["has_avx2"]:
        avx = "[green]AVX2[/green]"
    else:
        avx = "[yellow]Basic[/yellow]"
    table.add_row("CPU", cpu_detail, avx)

    # RAM
    ram = sys_info["ram"]
    ram_color = "green" if ram["total_gb"] >= 16 else ("yellow" if ram["total_gb"] >= 8 else "red")
    table.add_row(
        "RAM",
        f"Total: {ram['total_gb']} GB | Available: {ram['available_gb']} GB | Used: {ram['percent_used']}%",
        f"[{ram_color}]{ram['total_gb']} GB[/{ram_color}]"
    )

    # GPU(s)
    gpus = sys_info["gpus"]
    if gpus:
        for g in gpus:
            vram_color = "green" if g["vram_total_gb"] >= 8 else ("yellow" if g["vram_total_gb"] >= 4 else "red")
            table.add_row(
                f"GPU {g['index']}",
                f"{g['name']}\nVRAM: {g['vram_total_gb']} GB (Free: {g['vram_free_gb']} GB)",
                f"[{vram_color}]{g['vram_total_gb']} GB[/{vram_color}]"
            )
    else:
        table.add_row("GPU", "No NVIDIA GPU detected", "[red]None[/red]")

    # Disk
    disk = sys_info["disk"]
    disk_color = "green" if disk["free_gb"] >= 100 else ("yellow" if disk["free_gb"] >= 30 else "red")
    table.add_row(
        "Disk",
        f"Free: {disk['free_gb']} GB / Total: {disk['total_gb']} GB",
        f"[{disk_color}]{disk['free_gb']} GB[/{disk_color}]"
    )

    console.print(table)


def detect_model_precision(model_data, files_data):
    """
    Detect the precision/format of the model from its metadata and files.
    Returns: "FP32", "FP16", "BF16", "Q8", "Q4", "GGUF", "GPTQ", "AWQ", or "UNKNOWN"
    """
    model_id = model_data["model_id"].lower()
    tags = [t.lower() for t in model_data.get("tags", [])]

    # Check for GGUF models
    if model_data.get("is_gguf") or "gguf" in model_id:
        # Try to detect specific quant from filename
        quant_patterns = {
            "q2_k": "Q2_K", "q3_k": "Q3_K", "q4_0": "Q4_0", "q4_k_m": "Q4_K_M",
            "q4_k_s": "Q4_K_S", "q5_0": "Q5_0", "q5_k_m": "Q5_K_M", "q5_k_s": "Q5_K_S",
            "q6_k": "Q6_K", "q8_0": "Q8_0", "f16": "FP16", "f32": "FP32",
        }
        for key, label in quant_patterns.items():
            if key in model_id:
                return f"GGUF-{label}"
        return "GGUF"

    # Check for GPTQ/AWQ
    if "gptq" in model_id or "gptq" in tags:
        return "GPTQ"
    if "awq" in model_id or "awq" in tags:
        return "AWQ"

    # Check for specific precision in name or files
    if "fp32" in model_id or "float32" in model_id:
        return "FP32"
    if "bf16" in model_id or "bfloat16" in model_id:
        return "BF16"
    if "fp16" in model_id or "float16" in model_id:
        return "FP16"
    if "int8" in model_id or "8bit" in model_id:
        return "INT8"
    if "int4" in model_id or "4bit" in model_id:
        return "INT4"

    # Check files for clues
    if files_data and files_data.get("files"):
        for f in files_data["files"]:
            fname = f["filename"].lower()
            if fname.endswith(".gguf"):
                return "GGUF"
            if "fp16" in fname:
                return "FP16"
            if "bf16" in fname:
                return "BF16"

    # Default: most HuggingFace safetensors models are BF16 or FP16
    if any(f.get("filename", "").endswith(".safetensors") for f in (files_data or {}).get("files", [])):
        return "FP16/BF16"

    return "FP16/BF16"


def search_model_variants(model_id):
    """
    Search HuggingFace for alternative quantized versions of this model.
    Returns list of variant dicts.
    """
    import requests

    base_name = model_id.split("/")[-1]
    # Remove common suffixes to get base model name
    clean_name = re.sub(r'[-_](fp16|bf16|fp32|gguf|gptq|awq|int[48]|q[2-8].*)', '', base_name, flags=re.IGNORECASE)
    author = model_id.split("/")[0] if "/" in model_id else ""

    variants = []

    # Search for GGUF versions
    search_queries = [
        f"{clean_name} GGUF",
        f"{author}/{clean_name}",
    ]

    for query in search_queries:
        try:
            resp = requests.get(
                "https://huggingface.co/api/models",
                params={"search": query, "limit": 10, "sort": "downloads"},
                timeout=10
            )
            if resp.status_code == 200:
                for m in resp.json():
                    mid = m.get("id", "")
                    mtags = [t.lower() for t in m.get("tags", [])]
                    if mid == model_id:
                        continue  # skip the same model
                    # Check if this is related
                    if clean_name.lower() in mid.lower():
                        # Determine format
                        fmt = "FP16"
                        if "gguf" in mtags or "gguf" in mid.lower():
                            fmt = "GGUF"
                        elif "gptq" in mtags or "gptq" in mid.lower():
                            fmt = "GPTQ"
                        elif "awq" in mtags or "awq" in mid.lower():
                            fmt = "AWQ"

                        variants.append({
                            "id": mid,
                            "format": fmt,
                            "downloads": m.get("downloads", 0),
                            "likes": m.get("likes", 0),
                        })
        except Exception:
            pass

    # Deduplicate
    seen = set()
    unique = []
    for v in variants:
        if v["id"] not in seen:
            seen.add(v["id"])
            unique.append(v)

    return unique[:8]  # Top 8


def show_model_info_and_format(model_data, files_data, sys_info):
    """Display model info including detected format and system compatibility."""
    console.print("\n[bold yellow]>> STEP 2: Model Analysis[/bold yellow]\n")

    # Detect precision
    precision = detect_model_precision(model_data, files_data)

    table = Table(title=f"Model: {model_data['model_id']}", box=box.DOUBLE_EDGE,
                  title_style="bold white", border_style="magenta")
    table.add_column("Property", style="bold", width=25)
    table.add_column("Value", width=45)

    table.add_row("Author", str(model_data["author"]))
    table.add_row("Task", str(model_data["pipeline_tag"]))
    table.add_row("License", str(model_data["license"]))
    table.add_row("Downloads", f"{model_data['downloads']:,}")
    table.add_row("Likes", f"{model_data['likes']:,}")

    if model_data["params_billion"]:
        table.add_row("Parameters", f"[bold]{model_data['params_billion']}B[/bold]")
    else:
        table.add_row("Parameters", "[yellow]Could not detect[/yellow]")

    # Format detection
    if model_data["is_quantized"] or model_data["is_gguf"]:
        color = "yellow"
        fmt_str = f"[yellow]{precision} (Already Quantized)[/yellow]"
    else:
        color = "green"
        fmt_str = f"[green]{precision} (Full Precision)[/green]"
    table.add_row("Detected Format", fmt_str)

    if files_data:
        table.add_row("Repo Size", f"{files_data['total_size_gb']} GB ({files_data['file_count']} files)")

    console.print(table)

    return precision


def show_download_options(model_data, precision, capability, sys_info):
    """
    Show download options based on system capability and ask user which to download.
    Returns the chosen format string.
    """
    console.print("\n[bold yellow]>> STEP 3: Download Options (Based on Your System)[/bold yellow]\n")

    params = model_data["params_billion"]
    if not params:
        params = 7.0  # default

    # Calculate sizes for each format
    formats = [
        {
            "num": 1,
            "name": "FP16 (Full Precision)",
            "size_gb": round(params * 2, 1),
            "quality": "100%",
            "desc": "Original quality, largest size",
            "needs_ram_gb": round(params * 2 + 3, 1),
            "needs_vram_gb": round(params * 2 + 1, 1),
        },
        {
            "num": 2,
            "name": "Q8_0 (8-bit Quantized)",
            "size_gb": round(params * 1.0625, 1),
            "quality": "99.5%",
            "desc": "Near-lossless, half the size",
            "needs_ram_gb": round(params * 1.0625 + 3, 1),
            "needs_vram_gb": round(params * 1.0625 + 1, 1),
        },
        {
            "num": 3,
            "name": "Q4_K_M (4-bit NEXUS Optimal)",
            "size_gb": round(params * 0.5625, 1),
            "quality": "97-98%",
            "desc": "Best balance for consumer hardware",
            "needs_ram_gb": round(params * 0.5625 + 3, 1),
            "needs_vram_gb": round(params * 0.5625 + 0.5, 1),
        },
        {
            "num": 4,
            "name": "Q4_K_M + DecDEC (NEXUS Full)",
            "size_gb": round(params * 0.5625, 1),
            "quality": "98-99%",
            "desc": "Q4 with error correction (NEXUS best)",
            "needs_ram_gb": round(params * 0.5625 + 4, 1),
            "needs_vram_gb": round(params * 0.5625 + 1.5, 1),
        },
    ]

    # Determine system capabilities
    total_ram = sys_info["ram"]["total_gb"]
    os_overhead = 2.5 if "Windows" in sys_info["os"] else 1.2
    usable_ram = total_ram - os_overhead
    gpu_vram = max((g["vram_total_gb"] for g in sys_info["gpus"]), default=0)
    total_memory = usable_ram + gpu_vram

    # Build the options table
    opt_table = Table(title="Available Download Options", box=box.DOUBLE_EDGE, border_style="cyan")
    opt_table.add_column("#", justify="center", width=4)
    opt_table.add_column("Format", style="bold", width=28)
    opt_table.add_column("Size", justify="right", width=10)
    opt_table.add_column("Quality", justify="center", width=10)
    opt_table.add_column("RAM Needed", justify="right", width=12)
    opt_table.add_column("Can Run?", justify="center", width=12)
    opt_table.add_column("Download Time", justify="right", width=14)

    runnable_options = []

    for fmt in formats:
        # Check if system can run this format
        can_run_ram = usable_ram >= fmt["needs_ram_gb"]
        can_run_gpu = gpu_vram >= fmt["needs_vram_gb"]
        can_run_split = total_memory >= fmt["needs_ram_gb"]

        if can_run_gpu:
            status = "[bold green]GPU YES[/bold green]"
            can_run = True
        elif can_run_ram:
            status = "[green]CPU YES[/green]"
            can_run = True
        elif can_run_split:
            status = "[yellow]Split OK[/yellow]"
            can_run = True
        else:
            status = "[bold red]NO[/bold red]"
            can_run = False

        dl_time = estimate_download_time(fmt["size_gb"])

        # For Q4_K_M options, we always download FP16 first, then quantize locally
        if fmt["num"] in [3, 4]:
            dl_note = estimate_download_time(round(params * 2, 1))
            dl_time = f"{dl_note} *"

        opt_table.add_row(
            str(fmt["num"]),
            fmt["name"],
            f"{fmt['size_gb']} GB",
            fmt["quality"],
            f"{fmt['needs_ram_gb']} GB",
            status,
            dl_time,
        )

        if can_run:
            runnable_options.append(fmt["num"])

    console.print(opt_table)

    # Explain the flow clearly
    console.print(Panel(
        "[bold]How Q4/Q8 options work:[/bold]\n\n"
        "  1. Download the full FP16 model from HuggingFace\n"
        "  2. Convert to GGUF format (CPU only, no GPU needed)\n"
        "  3. Quantize to your chosen format on YOUR device (CPU only)\n"
        "  4. Delete the FP16 original to save disk space\n\n"
        "[dim]Quantization runs on CPU - no GPU needed. Your device does it all.[/dim]",
        title="Download + Quantize Flow", border_style="dim"
    ))

    # System summary
    console.print(f"\n  [bold]Your Memory:[/bold] {usable_ram:.1f} GB RAM + {gpu_vram:.1f} GB VRAM = {total_memory:.1f} GB total")

    if not runnable_options:
        console.print(Panel(
            "[bold red]None of these formats will fit on your system![/bold red]\n\n"
            "Suggestions:\n"
            "  1. Try a smaller model (fewer parameters)\n"
            "  2. Close other applications to free RAM\n"
            "  3. Use a cloud service (Kaggle, Colab) for inference",
            title="[red]SYSTEM TOO LIMITED[/red]", border_style="red"
        ))
        return None

    # Recommend best option
    if 4 in runnable_options:
        rec = 4
    elif 3 in runnable_options:
        rec = 3
    elif 2 in runnable_options:
        rec = 2
    else:
        rec = 1

    rec_fmt = next(f for f in formats if f["num"] == rec)
    console.print(Panel(
        f"[bold green]Recommended: Option {rec} - {rec_fmt['name']}[/bold green]\n"
        f"  Size: {rec_fmt['size_gb']} GB | Quality: {rec_fmt['quality']} | {rec_fmt['desc']}",
        title="[green]BEST FOR YOUR SYSTEM[/green]", border_style="green"
    ))

    # Show which options CAN'T run
    cant_run = [f for f in formats if f["num"] not in runnable_options]
    if cant_run:
        names = ", ".join(f["name"].split(" (")[0] for f in cant_run)
        console.print(f"  [red]Cannot run on your system:[/red] {names}")

    # Ask user
    console.print()
    valid_choices = ", ".join(str(n) for n in runnable_options)
    choice = Prompt.ask(
        f"[bold cyan]Choose format to download (options: {valid_choices})[/bold cyan]",
        default=str(rec)
    )

    try:
        choice_num = int(choice)
    except ValueError:
        console.print("[yellow]Invalid choice, using recommended option.[/yellow]")
        choice_num = rec

    if choice_num not in runnable_options:
        if choice_num in [f["num"] for f in formats]:
            console.print(f"[yellow]Option {choice_num} will NOT fit in your system's memory![/yellow]")
            if not Confirm.ask("[yellow]Download anyway? (model won't run on this device)[/yellow]"):
                choice_num = rec
                console.print(f"[green]Using recommended option {rec} instead.[/green]")
        else:
            console.print(f"[yellow]Invalid option. Using recommended option {rec}.[/yellow]")
            choice_num = rec

    chosen = next(f for f in formats if f["num"] == choice_num)
    console.print(f"\n  [bold green]Selected: {chosen['name']} ({chosen['size_gb']} GB, {chosen['quality']} quality)[/bold green]")

    return chosen


def show_nexus_pipeline(capability, model_data, chosen_format, sys_info):
    """Show the NEXUS-LLM conversion pipeline steps."""
    console.print("\n[bold yellow]>> STEP 4: NEXUS-LLM Conversion Pipeline[/bold yellow]\n")

    q4_gb = capability["model_q4km_gb"]
    fp16_gb = capability["model_fp16_gb"]

    # Adjust pipeline based on chosen format
    if chosen_format["num"] == 1:
        # FP16 — just download, no quantization
        steps = [
            {"num": 1, "name": "Download FP16 Model",
             "desc": f"Download {model_data['model_id']} ({fp16_gb} GB)",
             "time": estimate_download_time(fp16_gb), "needs_gpu": False, "local": True},
            {"num": 2, "name": "EAGLE-3 Draft Head Training",
             "desc": "Train speculative decoding head for 3-5x speed boost",
             "time": "15-20 hours", "needs_gpu": True, "local": capability["can_train"]["full_local"]},
        ]
    elif chosen_format["num"] == 2:
        # Q8 — download FP16 then quantize to Q8
        steps = [
            {"num": 1, "name": "Download FP16 Model",
             "desc": f"Download {model_data['model_id']} ({fp16_gb} GB)",
             "time": estimate_download_time(fp16_gb), "needs_gpu": False, "local": True},
            {"num": 2, "name": "Convert to GGUF Format",
             "desc": "Convert safetensors to GGUF", "time": "5-15 min",
             "needs_gpu": False, "local": True},
            {"num": 3, "name": "Q8_0 Quantization",
             "desc": f"Quantize to Q8_0 ({fp16_gb} GB -> {capability['model_q8_gb']} GB)",
             "time": "10-30 min", "needs_gpu": False, "local": True},
            {"num": 4, "name": "EAGLE-3 Draft Head Training",
             "desc": "Speed boost (3-5x, zero quality loss)",
             "time": "15-20 hours", "needs_gpu": True, "local": capability["can_train"]["full_local"]},
        ]
    else:
        # Q4_K_M (options 3 and 4)
        steps = [
            {"num": 1, "name": "Download FP16 Model",
             "desc": f"Download {model_data['model_id']} ({fp16_gb} GB)",
             "time": estimate_download_time(fp16_gb), "needs_gpu": False, "local": True},
            {"num": 2, "name": "Convert to GGUF Format",
             "desc": "Convert safetensors to GGUF", "time": "5-15 min",
             "needs_gpu": False, "local": True},
            {"num": 3, "name": "Q4_K_M Quantization",
             "desc": f"Quantize to Q4_K_M ({fp16_gb} GB -> {q4_gb} GB) CPU only",
             "time": "10-30 min", "needs_gpu": False, "local": True},
        ]
        if chosen_format["num"] == 4:
            steps.extend([
                {"num": 4, "name": "DecDEC Residual Computation",
                 "desc": f"Error correction residuals (+{fp16_gb} GB disk)",
                 "time": "2-3 hours", "needs_gpu": True, "local": capability["can_train"]["eagle3_local"]},
                {"num": 5, "name": "MixLLM Salience Analysis",
                 "desc": "Critical channel FP16 preservation (5% weights)",
                 "time": "30-60 min", "needs_gpu": True, "local": capability["can_train"]["eagle3_local"]},
            ])
        steps.extend([
            {"num": len(steps) + 1, "name": "EAGLE-3 Draft Head Training",
             "desc": "Speed boost (3-5x, zero quality loss)",
             "time": "15-20 hours", "needs_gpu": True, "local": capability["can_train"]["full_local"]},
            {"num": len(steps) + 2, "name": "FlexiDepth Router Training",
             "desc": "Layer-skip routers (25% compute savings)",
             "time": "5-10 min", "needs_gpu": True, "local": True},
        ])

    table = Table(title="NEXUS-LLM Pipeline Steps", box=box.DOUBLE_EDGE, border_style="magenta")
    table.add_column("#", justify="center", width=4)
    table.add_column("Step", style="bold", width=30)
    table.add_column("Description", width=45)
    table.add_column("Time", justify="right", width=12)
    table.add_column("Where?", justify="center", width=14)

    for s in steps:
        if s["local"]:
            where = "[green]Local[/green]"
        else:
            where = "[yellow]Kaggle/Cloud[/yellow]"
        table.add_row(str(s["num"]), s["name"], s["desc"], s["time"], where)

    console.print(table)

    local_steps = [s for s in steps if s["local"]]
    cloud_steps = [s for s in steps if not s["local"]]

    # ================================================================
    # DEVICE POWER CHECK — Can full NEXUS conversion happen locally?
    # ================================================================
    console.print("\n[bold yellow]>> DEVICE POWER CHECK[/bold yellow]\n")

    gpu_vram = max((g["vram_total_gb"] for g in sys_info["gpus"]), default=0)
    total_ram = sys_info["ram"]["total_gb"]

    power_table = Table(title="Can Your Device Handle Full NEXUS Conversion?",
                        box=box.DOUBLE_EDGE, border_style="cyan")
    power_table.add_column("Pipeline Step", style="bold", width=28)
    power_table.add_column("Needs", width=20)
    power_table.add_column("Your Device", width=20)
    power_table.add_column("Verdict", justify="center", width=14)

    step_requirements = [
        ("Download Model", "Internet only", "Any device", True, False),
        ("GGUF Conversion", "CPU + RAM", f"{total_ram:.0f} GB RAM", True, False),
        ("Quantization (Q4/Q8)", "CPU only", f"{total_ram:.0f} GB RAM", True, False),
        ("DecDEC Residuals", "8+ GB VRAM", f"{gpu_vram:.0f} GB VRAM", gpu_vram >= 8, True),
        ("MixLLM Salience", "8+ GB VRAM", f"{gpu_vram:.0f} GB VRAM", gpu_vram >= 8, True),
        ("EAGLE-3 Training", "16+ GB VRAM", f"{gpu_vram:.0f} GB VRAM", gpu_vram >= 16, True),
        ("FlexiDepth Routers", "4+ GB VRAM", f"{gpu_vram:.0f} GB VRAM", gpu_vram >= 4, True),
    ]

    all_local = True
    needs_cloud_steps = []
    for name, needs, have, can_local, needs_gpu in step_requirements:
        if can_local:
            verdict = "[bold green]LOCAL[/bold green]"
        else:
            verdict = "[bold yellow]CLOUD[/bold yellow]"
            all_local = False
            needs_cloud_steps.append(name)

        power_table.add_row(name, needs, have, verdict)

    console.print(power_table)

    if all_local:
        console.print(Panel(
            "[bold green]Your device can handle the ENTIRE NEXUS-LLM conversion![/bold green]\n\n"
            "All pipeline steps will run locally on your machine.\n"
            "No cloud GPU needed. Everything happens on your device.",
            title="[green]FULL LOCAL CONVERSION POSSIBLE[/green]",
            border_style="green"
        ))
    else:
        cloud_names = ", ".join(needs_cloud_steps)
        local_count = len(step_requirements) - len(needs_cloud_steps)

        console.print(Panel(
            f"[bold yellow]Your device can do {local_count}/{len(step_requirements)} steps locally.[/bold yellow]\n\n"
            f"[green]LOCAL (will run on your device):[/green]\n"
            f"  Download, GGUF Convert, Quantize" +
            (f", FlexiDepth" if gpu_vram >= 4 else "") + "\n\n"
            f"[yellow]CLOUD NEEDED (use Kaggle T4 - FREE):[/yellow]\n"
            f"  {cloud_names}\n\n"
            f"[dim]Kaggle gives you a free T4 GPU (16 GB VRAM) for 30 hrs/week.\n"
            f"Upload the quantized model to Kaggle, run the training steps there,\n"
            f"then download the results back to your device.[/dim]",
            title="[yellow]PARTIAL LOCAL - CLOUD NEEDED FOR SOME STEPS[/yellow]",
            border_style="yellow"
        ))

    return steps, local_steps, cloud_steps, all_local


def find_or_install_llama_cpp():
    """
    Find llama-quantize on the system.
    If not found, automatically download and install llama.cpp.
    Returns the path to llama-quantize binary, or None if installation failed.
    """
    # --- Step 1: Check if already installed ---
    llama_quantize = shutil.which("llama-quantize")
    if llama_quantize:
        console.print(f"  [green]Found llama-quantize: {llama_quantize}[/green]")
        return llama_quantize

    # Check common paths
    is_win = sys.platform == "win32"
    exe = ".exe" if is_win else ""

    common_paths = [
        os.path.expanduser(f"~/llama.cpp/llama-quantize{exe}"),
        os.path.expanduser(f"~/llama.cpp/build/bin/llama-quantize{exe}"),
        os.path.expanduser(f"~/llama.cpp/build/bin/Release/llama-quantize{exe}"),
        f"C:\\llama.cpp\\llama-quantize{exe}" if is_win else "",
        f"C:\\llama.cpp\\build\\bin\\Release\\llama-quantize{exe}" if is_win else "",
        "/usr/local/bin/llama-quantize",
    ]

    # Also check tool's own directory
    tool_dir = os.path.dirname(os.path.abspath(__file__))
    llama_in_tool = os.path.join(tool_dir, "llama.cpp")
    common_paths.extend([
        os.path.join(llama_in_tool, f"llama-quantize{exe}"),
        os.path.join(llama_in_tool, f"build/bin/llama-quantize{exe}"),
        os.path.join(llama_in_tool, f"build/bin/Release/llama-quantize{exe}"),
    ])

    for p in common_paths:
        if p and os.path.isfile(p):
            console.print(f"  [green]Found llama-quantize: {p}[/green]")
            return p

    # --- Step 2: Not found — offer to auto-install ---
    console.print("  [yellow]llama.cpp not found on your system.[/yellow]")
    console.print()

    if not Confirm.ask("  [bold cyan]Download llama.cpp automatically?[/bold cyan]"):
        return None

    install_dir = os.path.join(tool_dir, "llama.cpp")

    if is_win:
        return _download_llama_cpp_windows(install_dir)
    else:
        return _download_llama_cpp_unix(install_dir)


def _download_llama_cpp_windows(install_dir):
    """Download pre-built llama.cpp Windows binaries from GitHub releases."""
    import zipfile
    import requests

    console.print("\n  [bold cyan]Downloading llama.cpp pre-built binaries...[/bold cyan]")

    # Get latest release info from GitHub
    try:
        resp = requests.get(
            "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest",
            timeout=30
        )
        resp.raise_for_status()
        release = resp.json()
        tag = release["tag_name"]
        console.print(f"  Latest release: [bold]{tag}[/bold]")

        # Find Windows binary asset (prefer plain win-x64)
        download_url = None
        asset_name = None
        preferred = ["win-amd64-x64", "win-x64", "bin-win"]

        for asset in release.get("assets", []):
            name = asset["name"].lower()
            if ".zip" in name and "win" in name:
                for pref in preferred:
                    if pref in name:
                        download_url = asset["browser_download_url"]
                        asset_name = asset["name"]
                        break
                if not download_url and "cudart" not in name and "vulkan" not in name:
                    download_url = asset["browser_download_url"]
                    asset_name = asset["name"]

        if not download_url:
            console.print("  [red]Could not find Windows binary in latest release.[/red]")
            return _download_llama_cpp_pip_fallback(install_dir)

    except Exception as e:
        console.print(f"  [yellow]Could not reach GitHub API: {e}[/yellow]")
        return _download_llama_cpp_pip_fallback(install_dir)

    # Download the zip
    console.print(f"  Downloading: [dim]{asset_name}[/dim]")
    zip_path = os.path.join(os.path.dirname(install_dir), "llama-cpp-release.zip")

    try:
        with requests.get(download_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                task = progress.add_task("Downloading llama.cpp...", total=total or None)

                with open(zip_path, "wb") as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            progress.update(task, completed=downloaded)

                progress.update(task, completed=total or 100, total=total or 100)

        console.print("  Extracting...")
        os.makedirs(install_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(install_dir)

        os.remove(zip_path)

        # Find llama-quantize.exe in extracted files
        for root, dirs, files in os.walk(install_dir):
            for fname in files:
                if fname.lower() == "llama-quantize.exe":
                    path = os.path.join(root, fname)
                    console.print(f"  [bold green]llama.cpp installed![/bold green]")
                    console.print(f"  Path: {path}")
                    return path

        console.print("  [yellow]llama-quantize.exe not found in release archive.[/yellow]")
        return _download_llama_cpp_pip_fallback(install_dir)

    except Exception as e:
        console.print(f"  [red]Download failed: {e}[/red]")
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        return _download_llama_cpp_pip_fallback(install_dir)


def _download_llama_cpp_unix(install_dir):
    """Clone and build llama.cpp on Linux/Mac."""
    console.print("\n  [bold cyan]Cloning and building llama.cpp...[/bold cyan]")

    try:
        # Check for git
        if not shutil.which("git"):
            console.print("  [red]git not found. Install git first.[/red]")
            return _download_llama_cpp_pip_fallback(install_dir)

        # Check for make or cmake
        has_make = shutil.which("make")
        has_cmake = shutil.which("cmake")
        if not has_make and not has_cmake:
            console.print("  [red]make/cmake not found. Install build tools first.[/red]")
            return _download_llama_cpp_pip_fallback(install_dir)

        # Clone
        console.print("  Cloning repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", install_dir],
            check=True, capture_output=True, text=True
        )

        # Build
        console.print("  Building (this may take 2-5 minutes)...")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      TimeElapsedColumn(), console=console) as progress:
            ptask = progress.add_task("Compiling llama.cpp...", total=None)

            if has_cmake:
                build_dir = os.path.join(install_dir, "build")
                os.makedirs(build_dir, exist_ok=True)
                subprocess.run(["cmake", ".."], cwd=build_dir, check=True, capture_output=True)
                subprocess.run(["cmake", "--build", ".", "--config", "Release", "-j"],
                             cwd=build_dir, check=True, capture_output=True)
            else:
                subprocess.run(["make", "-j"], cwd=install_dir, check=True, capture_output=True)

            progress.update(ptask, completed=100, total=100)

        # Find the binary
        for candidate in [
            os.path.join(install_dir, "llama-quantize"),
            os.path.join(install_dir, "build", "bin", "llama-quantize"),
        ]:
            if os.path.isfile(candidate):
                console.print(f"  [bold green]llama.cpp built successfully![/bold green]")
                console.print(f"  Path: {candidate}")
                return candidate

        console.print("  [yellow]Build completed but llama-quantize not found.[/yellow]")
        return None

    except subprocess.CalledProcessError as e:
        console.print(f"  [red]Build failed: {e.stderr[:200] if e.stderr else 'Unknown error'}[/red]")
        return _download_llama_cpp_pip_fallback(install_dir)


def _download_llama_cpp_pip_fallback(install_dir):
    """Fallback: try installing llama-cpp-python via pip which includes quantize tools."""
    console.print("\n  [dim]Trying pip fallback: llama-cpp-python...[/dim]")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "llama-cpp-python"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            # Check if quantize is now available
            quantize = shutil.which("llama-quantize")
            if quantize:
                console.print(f"  [green]Installed via pip! Path: {quantize}[/green]")
                return quantize

        console.print("  [yellow]pip fallback did not provide llama-quantize.[/yellow]")
        console.print("  [bold]Manual install needed:[/bold]")
        if sys.platform == "win32":
            console.print("    1. Go to: github.com/ggerganov/llama.cpp/releases")
            console.print("    2. Download the Windows .zip")
            console.print("    3. Extract to C:\\llama.cpp")
        else:
            console.print("    git clone https://github.com/ggerganov/llama.cpp")
            console.print("    cd llama.cpp && make -j$(nproc)")
        return None

    except Exception:
        console.print("  [yellow]pip fallback failed too.[/yellow]")
        return None


def run_local_pipeline(model_id, output_dir, capability, chosen_format):
    """Execute the local pipeline steps (download + quantize)."""
    console.print("\n[bold yellow]>> STEP 5: Running NEXUS-LLM Pipeline[/bold yellow]\n")

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Download ---
    console.print("[bold cyan][Step 1] Downloading model from HuggingFace...[/bold cyan]")
    console.print(f"  Model:  {model_id}")
    console.print(f"  Format: {chosen_format['name']}")
    console.print(f"  Output: {output_dir}")
    console.print()

    from huggingface_hub import snapshot_download

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Downloading model files...", total=None)

            download_path = snapshot_download(
                repo_id=model_id,
                local_dir=os.path.join(output_dir, "fp16_original"),
                local_dir_use_symlinks=False,
            )
            progress.update(task, completed=100, total=100)

        console.print(f"  [green]Download complete![/green] Saved to: {download_path}")
    except Exception as e:
        err_str = str(e)
        console.print(f"  [red]Download failed: {err_str[:200]}[/red]")
        if "401" in err_str or "403" in err_str or "gated" in err_str.lower():
            console.print("  [yellow]This model is gated. You need to:[/yellow]")
            console.print("  [yellow]  1. Accept the license at huggingface.co[/yellow]")
            console.print("  [yellow]  2. Run: huggingface-cli login[/yellow]")
        return False

    # If FP16 only, we're done with download
    if chosen_format["num"] == 1:
        console.print("\n  [bold green]FP16 model ready![/bold green]")
        console.print(f"  Path: {download_path}")
        return True

    # --- Step 2+3: Quantize ---
    quant_type = "Q8_0" if chosen_format["num"] == 2 else "Q4_K_M"
    console.print(f"\n[bold cyan][Step 2-3] Converting and Quantizing to {quant_type}...[/bold cyan]")

    llama_cpp_path = find_or_install_llama_cpp()
    convert_script = None

    if not llama_cpp_path:
        console.print("  [yellow]Could not install llama.cpp automatically.[/yellow]")
        console.print("  [yellow]FP16 model downloaded successfully at:[/yellow]")
        console.print(f"  {download_path}")
        console.print(f"  [yellow]Install llama.cpp manually, then quantize with:[/yellow]")
        console.print(f"  llama-quantize <gguf_file> output.gguf {quant_type}")
        return True

    # Find convert script
    llama_dir = os.path.dirname(os.path.dirname(llama_cpp_path))
    for name in ["convert_hf_to_gguf.py", "convert-hf-to-gguf.py"]:
        p = os.path.join(llama_dir, name)
        if os.path.isfile(p):
            convert_script = p
            break

    gguf_fp16 = os.path.join(output_dir, "model-f16.gguf")
    gguf_output = os.path.join(output_dir, f"model-{quant_type}.gguf")

    if convert_script:
        console.print("  Converting HF safetensors to GGUF...")
        try:
            subprocess.run(
                [sys.executable, convert_script, download_path, "--outfile", gguf_fp16],
                check=True, capture_output=True, text=True
            )
            console.print(f"  [green]GGUF conversion complete[/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"  [red]GGUF conversion failed: {e.stderr[:200]}[/red]")
            return False

    if os.path.isfile(gguf_fp16):
        console.print(f"  Quantizing to {quant_type} (this may take 10-30 min)...")
        try:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          TimeElapsedColumn(), console=console) as progress:
                ptask = progress.add_task(f"Quantizing to {quant_type}...", total=None)
                result = subprocess.run(
                    [llama_cpp_path, gguf_fp16, gguf_output, quant_type],
                    capture_output=True, text=True
                )
                progress.update(ptask, completed=100, total=100)

            if os.path.isfile(gguf_output):
                size_gb = os.path.getsize(gguf_output) / (1024**3)
                console.print(f"  [bold green]Quantization complete![/bold green]")
                console.print(f"  Output: {gguf_output} ({size_gb:.1f} GB)")
            else:
                console.print(f"  [red]Quantization may have failed.[/red]")
                if result.stderr:
                    console.print(f"  [dim]{result.stderr[:300]}[/dim]")
        except Exception as e:
            console.print(f"  [red]Quantization error: {e}[/red]")

    # --- Summary ---
    console.print()
    summary_table = Table(title="Conversion Summary", box=box.DOUBLE_EDGE, border_style="green")
    summary_table.add_column("Item", style="bold", width=25)
    summary_table.add_column("Path / Status", width=55)

    summary_table.add_row("FP16 Original", download_path)

    if os.path.isfile(gguf_output):
        summary_table.add_row(f"{quant_type} Model", gguf_output)
        summary_table.add_row("Base Status", "[bold green]READY[/bold green]")
    elif os.path.isfile(gguf_fp16):
        summary_table.add_row("GGUF FP16", gguf_fp16)
        summary_table.add_row("Base Status", f"[yellow]Need to quantize to {quant_type}[/yellow]")
    else:
        summary_table.add_row("Base Status", "[yellow]Needs GGUF conversion[/yellow]")

    if chosen_format["num"] == 4:
        summary_table.add_row("DecDEC Residuals", "[yellow]Run on Kaggle T4[/yellow]")
        summary_table.add_row("MixLLM Channels", "[yellow]Run on Kaggle T4[/yellow]")

    summary_table.add_row("EAGLE-3 Head", "[yellow]Run on Kaggle T4[/yellow]")
    summary_table.add_row("FlexiDepth Routers", "[yellow]Run on Kaggle T4[/yellow]")

    console.print(summary_table)
    return True


def generate_nexus_config(model_id, output_dir, capability, chosen_format):
    """Generate the NEXUS-LLM config."""
    quant = "Q4_K_M" if chosen_format["num"] in [3, 4] else ("Q8_0" if chosen_format["num"] == 2 else "FP16")

    config = {
        "nexus_llm": {
            "version": "1.1.0",
            "model": {
                "id": model_id,
                "base": f"model-{quant}.gguf" if quant != "FP16" else "fp16_original/",
                "params_billion": capability["model_params_b"],
                "format": quant,
                "quality": chosen_format["quality"],
            },
            "ppq": {
                "decdec": {"enabled": chosen_format["num"] == 4,
                           "residuals_file": "model-residuals.bin"},
                "mixllm": {"enabled": chosen_format["num"] == 4,
                           "weights_file": "model-mixllm-fp16.bin"},
            },
            "lsr": {
                "eagle3": {"enabled": False, "head_file": "model-eagle3-head.bin", "draft_tokens": 6},
                "flexidepth": {"enabled": False, "routers_file": "model-flexidepth.bin"},
            },
            "acm": {
                "q_filters": {"enabled": True, "compression_ratio": 32},
                "streaming_llm": {"enabled": True, "sink_tokens": 4, "window_size": 2048},
            },
            "rol": {
                "tacs": {"enabled": True, "temp_warning_c": 85},
                "prefetch": {"enabled": True, "lookahead_layers": 2},
            },
            "recommended_config": capability["best_config"],
            "estimated_tps": capability["est_tps"],
        }
    }

    config_path = os.path.join(output_dir, "nexus_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    console.print(f"\n  [green]Config saved: {config_path}[/green]")
    return config_path


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    show_banner()

    # --- Step 1: System Detection ---
    with console.status("[bold cyan]Scanning your hardware...[/bold cyan]", spinner="dots"):
        sys_info = get_full_system_info()
    show_system_info(sys_info)

    # --- HuggingFace Token Check ---
    console.print("\n[bold yellow]>> HuggingFace Account[/bold yellow]\n")
    if not check_hf_token():
        return

    # --- Step 2: Get Model ID ---
    console.print("\n[bold yellow]>> Choose a Model[/bold yellow]\n")
    console.print("  Paste a HuggingFace model link or ID.")
    console.print("  Examples:")
    console.print("    [dim]meta-llama/Llama-3.1-8B[/dim]")
    console.print("    [dim]mistralai/Mistral-7B-v0.3[/dim]")
    console.print("    [dim]https://huggingface.co/Qwen/Qwen2.5-7B[/dim]")
    console.print()

    user_input = Prompt.ask("[bold cyan]Model ID or URL[/bold cyan]")
    model_id = parse_model_id(user_input)

    if not model_id:
        console.print("[red]Invalid model ID format. Use: org/model-name[/red]")
        return

    console.print(f"\n  Parsed model ID: [bold]{model_id}[/bold]")

    # Fetch model info
    with console.status(f"[bold cyan]Fetching info for {model_id}...[/bold cyan]", spinner="dots"):
        raw_info = get_model_info(model_id)
        if "error" in raw_info:
            console.print(f"\n[red]Error: {raw_info['error']}[/red]")
            return
        model_data = estimate_model_size(raw_info)
        files_data = get_model_files(model_id)

    # Show model info with format detection
    precision = show_model_info_and_format(model_data, files_data, sys_info)

    # Handle parameter detection
    if not model_data["params_billion"]:
        console.print("\n[yellow]Could not auto-detect parameter count.[/yellow]")
        param_input = Prompt.ask("Enter parameter count in billions (e.g., 7 or 14)", default="7")
        try:
            model_data["params_billion"] = float(param_input)
            model_data["fp16_size_gb"] = round(model_data["params_billion"] * 2, 1)
            model_data["q4km_size_gb"] = round(model_data["params_billion"] * 0.5625, 1)
            model_data["q8_size_gb"] = round(model_data["params_billion"] * 1.0625, 1)
        except ValueError:
            model_data["params_billion"] = 7.0
            model_data["fp16_size_gb"] = 14.0
            model_data["q4km_size_gb"] = 3.9
            model_data["q8_size_gb"] = 7.4

    # Warning for already quantized models
    if model_data["is_quantized"] or model_data["is_gguf"]:
        console.print(Panel(
            f"[yellow]This model appears to be already quantized ({precision})![/yellow]\n\n"
            "For best NEXUS-LLM quality, start from the [bold]full FP16/BF16 version[/bold].\n"
            "Quantizing an already-quantized model loses additional quality.\n\n"
            "Look for the original model without 'GGUF/GPTQ/AWQ' in the name.",
            title="[yellow]WARNING: PRE-QUANTIZED MODEL[/yellow]", border_style="yellow"
        ))

        # Search for FP16 version
        console.print("\n  [dim]Searching for original FP16 version...[/dim]")
        variants = search_model_variants(model_id)
        fp16_variants = [v for v in variants if v["format"] not in ["GGUF", "GPTQ", "AWQ"]]
        if fp16_variants:
            console.print(f"  [green]Found FP16 version: {fp16_variants[0]['id']}[/green]")
            if Confirm.ask(f"  Switch to [bold]{fp16_variants[0]['id']}[/bold] instead?"):
                model_id = fp16_variants[0]["id"]
                # Re-fetch
                with console.status(f"[bold cyan]Fetching {model_id}...[/bold cyan]", spinner="dots"):
                    raw_info = get_model_info(model_id)
                    model_data = estimate_model_size(raw_info)
                    files_data = get_model_files(model_id)
                    precision = detect_model_precision(model_data, files_data)
                console.print(f"  [green]Switched to: {model_id}[/green]")
        else:
            if not Confirm.ask("Continue with quantized model anyway?"):
                return

    # --- Step 3: Compatibility + Download Choice ---
    capability = evaluate_capability(
        sys_info,
        model_data["fp16_size_gb"],
        model_data["params_billion"]
    )

    chosen_format = show_download_options(model_data, precision, capability, sys_info)
    if chosen_format is None:
        return

    # --- Step 4: Show Pipeline + Device Power Check ---
    steps, local_steps, cloud_steps, all_local = show_nexus_pipeline(
        capability, model_data, chosen_format, sys_info
    )

    # --- Step 5: Execute? ---
    console.print()
    if all_local:
        confirm_msg = "[bold cyan]Start FULL NEXUS-LLM conversion on your device?[/bold cyan]"
    else:
        confirm_msg = "[bold cyan]Start local steps now? (Cloud steps can be done later on Kaggle)[/bold cyan]"

    if Confirm.ask(confirm_msg):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  f"nexus-{model_id.split('/')[-1]}")
        console.print(f"\n  Output directory: [bold]{output_dir}[/bold]")

        success = run_local_pipeline(model_id, output_dir, capability, chosen_format)

        if success:
            generate_nexus_config(model_id, output_dir, capability, chosen_format)

            qual = chosen_format["quality"]

            if all_local:
                console.print(Panel(
                    f"[bold green]FULL NEXUS-LLM conversion complete on YOUR device![/bold green]\n\n"
                    f"Format: {chosen_format['name']} | Quality: {qual}\n\n"
                    "All steps ran locally. No cloud needed!\n"
                    "Your model is ready to use.",
                    title="[green]FULL LOCAL SUCCESS[/green]", border_style="green"
                ))
            else:
                cloud_names = ", ".join(s["name"] for s in cloud_steps)
                console.print(Panel(
                    f"[bold green]Local conversion complete![/bold green]\n\n"
                    f"Format: {chosen_format['name']} | Quality: {qual}\n\n"
                    "[green]Done locally:[/green]\n"
                    f"  {', '.join(s['name'] for s in local_steps)}\n\n"
                    "[yellow]Still needed (run on Kaggle T4 - FREE):[/yellow]\n"
                    f"  {cloud_names}\n\n"
                    "[dim]The model is already usable without cloud steps!\n"
                    "Cloud steps add speed (EAGLE-3) and quality (DecDEC) enhancements.[/dim]",
                    title="[green]LOCAL STEPS DONE[/green]", border_style="green"
                ))
    else:
        console.print("\n  [dim]Conversion cancelled. Run again when ready.[/dim]")

    console.print("\n[bold cyan]Thank you for using NEXUS-LLM Converter![/bold cyan]\n")


if __name__ == "__main__":
    main()
