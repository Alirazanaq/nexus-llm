#!/usr/bin/env python3
"""
NEXUS-LLM Converter Setup Script (Cross-Platform)

Works on: Windows, Linux, macOS

Run this once to:
  1. Install all required Python packages
  2. Register 'NEXUS-LLM' as a global command

After setup, just type 'NEXUS-LLM' or 'nexus-llm' in any terminal.
"""
import subprocess
import sys
import os
import shutil
import platform

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║           NEXUS-LLM CONVERTER - SETUP                    ║
║                                                          ║
║  This will:                                              ║
║    1. Install all Python dependencies                    ║
║    2. Create 'NEXUS-LLM' global command                  ║
║                                                          ║
║  Platform: {:<45s}                                       ║
╚══════════════════════════════════════════════════════════╝
""".format(f"{platform.system()} {platform.machine()}"))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    is_win = sys.platform == "win32"

    # ── Step 1: Install requirements ──
    print("[1/3] Installing Python dependencies...")
    req_file = os.path.join(script_dir, "requirements.txt")
    if os.path.isfile(req_file):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_file],
            capture_output=False
        )
        if result.returncode != 0:
            print("\n[ERROR] pip install failed. Try running manually:")
            print(f"  pip install -r \"{req_file}\"")
            return False
        print("[OK] All dependencies installed.\n")
    else:
        print(f"[WARN] requirements.txt not found at {req_file}")
        print("       Installing packages individually...")
        packages = ["huggingface_hub>=0.20.0", "rich>=13.0.0", "psutil>=5.9.0",
                     "requests>=2.31.0"]
        # pynvml is optional on systems without NVIDIA GPUs
        packages.append("pynvml>=11.5.0")
        for pkg in packages:
            subprocess.run([sys.executable, "-m", "pip", "install", pkg])
        print("[OK] Packages installed.\n")

    # ── Step 2: Create pip-installable package ──
    print("[2/3] Creating installable package...")

    pyproject_path = os.path.join(script_dir, "pyproject.toml")
    pyproject_content = '''[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "nexus-llm-converter"
version = "1.2.0"
description = "Convert any HuggingFace model to NEXUS-LLM format"
requires-python = ">=3.9"
dependencies = [
    "huggingface_hub>=0.20.0",
    "rich>=13.0.0",
    "psutil>=5.9.0",
    "requests>=2.31.0",
    "pynvml>=11.5.0",
]

[project.scripts]
NEXUS-LLM = "nexus_converter:main"
nexus-llm = "nexus_converter:main"
'''

    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(pyproject_content)
    print(f"  Created {pyproject_path}")

    # ── Step 3: pip install in editable mode ──
    print("\n[3/3] Registering 'NEXUS-LLM' / 'nexus-llm' as global command...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", script_dir],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print("\n[WARN] pip install -e failed. Using alternative method...")
        create_manual_scripts(script_dir)
    else:
        print("[OK] Global command registered.\n")

    # ── Verify ──
    print("=" * 56)
    print()

    nexus_cmd = shutil.which("NEXUS-LLM") or shutil.which("nexus-llm")
    if nexus_cmd:
        print("  [SUCCESS] Setup complete!")
        print()
        print("  You can now run NEXUS-LLM from ANY terminal:")
        print()
        if is_win:
            print("    > NEXUS-LLM")
            print("    or")
            print("    > nexus-llm")
        else:
            print("    $ nexus-llm")
            print("    or")
            print("    $ NEXUS-LLM")
        print()
    else:
        print("  [PARTIAL] Packages installed, but global command")
        print("  may not be on PATH yet.")
        print()
        create_manual_scripts(script_dir)

        if is_win:
            print(f"  Try: Close and reopen terminal, then type: nexus-llm")
        else:
            print(f"  Try: source ~/.bashrc  (or ~/.zshrc), then type: nexus-llm")
            print(f"  Or:  python3 \"{os.path.join(script_dir, 'nexus_converter.py')}\"")
        print()

    print("=" * 56)
    return True


def create_manual_scripts(script_dir):
    """Create batch/shell scripts as fallback for global command."""
    converter_path = os.path.join(script_dir, "nexus_converter.py")

    if sys.platform == "win32":
        # Find Python Scripts directory
        scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
        user_scripts = os.path.join(os.path.expanduser("~"),
                                     "AppData", "Roaming", "Python",
                                     f"Python{sys.version_info.major}{sys.version_info.minor}",
                                     "Scripts")

        target_dir = user_scripts if os.path.isdir(user_scripts) else scripts_dir
        bat_content = f'@echo off\n"{sys.executable}" "{converter_path}" %*\n'

        try:
            os.makedirs(target_dir, exist_ok=True)
            for name in ["NEXUS-LLM.bat", "nexus-llm.bat"]:
                bat_path = os.path.join(target_dir, name)
                with open(bat_path, "w") as f:
                    f.write(bat_content)
                print(f"  Created shortcut: {bat_path}")
        except PermissionError:
            print(f"  [WARN] No permission to write to {target_dir}")
            print(f"  Run as administrator or manually create the shortcut.")

    else:
        # Linux / macOS: create shell scripts in ~/.local/bin
        local_bin = os.path.expanduser("~/.local/bin")
        os.makedirs(local_bin, exist_ok=True)

        sh_content = f'#!/bin/bash\n"{sys.executable}" "{converter_path}" "$@"\n'

        for name in ["NEXUS-LLM", "nexus-llm"]:
            sh_path = os.path.join(local_bin, name)
            with open(sh_path, "w") as f:
                f.write(sh_content)
            os.chmod(sh_path, 0o755)
            print(f"  Created: {sh_path}")

        # Ensure ~/.local/bin is in PATH
        shell_rc = None
        shell = os.environ.get("SHELL", "/bin/bash")
        if "zsh" in shell:
            shell_rc = os.path.expanduser("~/.zshrc")
        elif "bash" in shell:
            shell_rc = os.path.expanduser("~/.bashrc")
        elif "fish" in shell:
            shell_rc = os.path.expanduser("~/.config/fish/config.fish")

        if shell_rc and os.path.isfile(shell_rc):
            with open(shell_rc, "r") as f:
                rc_content = f.read()
            if ".local/bin" not in rc_content:
                with open(shell_rc, "a") as f:
                    f.write('\n# Added by NEXUS-LLM setup\nexport PATH="$HOME/.local/bin:$PATH"\n')
                print(f"  Added ~/.local/bin to PATH in {shell_rc}")
                print(f"  Run: source {shell_rc}")
        else:
            print(f"  Make sure ~/.local/bin is in your PATH:")
            print(f"  export PATH=\"$HOME/.local/bin:$PATH\"")


if __name__ == "__main__":
    success = main()
    if success:
        input("\nPress Enter to exit...")
    else:
        input("\nSetup had issues. Press Enter to exit...")
