#!/usr/bin/env python3
# Wrapper script - runs with venv Python if available, otherwise system Python
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(script_dir, "venv", "bin", "python")

# If venv Python exists, use it
if os.path.exists(venv_python):
    os.execv(venv_python, [venv_python, os.path.join(script_dir, "src", "main.py")])
else:
    # Fall back to system Python
    from src.main import run_interactive
    run_interactive()
