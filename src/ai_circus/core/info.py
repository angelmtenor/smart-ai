"""
Title:    Info
Author:   Angel Martinez-tenor, 2025. Adapted from https://github.com/angelmtenor/ds-template
"""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, TypeVar

import cpuinfo
import psutil

from ai_circus.core import logger

F = TypeVar("F", bound=Callable[..., Any])
log = logger.get_logger(__name__)

# Cached installed packages
INSTALLED_PACKAGES = {dist.metadata["Name"]: dist.version for dist in distributions()}

# Default modules to log
DEFAULT_MODULES = ["httpx"]


def info_os() -> None:
    """Log operating system version and architecture."""
    log.info(f"{'OS':<25}{platform.platform()}")


def info_software(modules: list[str] | None = None) -> None:
    """
    Log Python version and versions of specified modules.

    Args:
        modules (list[str] | None): List of module names to log. If None, uses DEFAULT_MODULES.
    """
    log.info(f"{'ENV':<25}{sys.prefix}")
    log.info(f"{'PYTHON':<25}{sys.version.split('(', 1)[0].strip()}")

    for module in modules or DEFAULT_MODULES:
        version = "--N/A--" if module == "pickle" else INSTALLED_PACKAGES.get(module, "--NO--")
        log.info(f" - {module:<22}{version}")


def info_hardware() -> None:
    """Log CPU model, core count, and RAM size."""
    cpu = cpuinfo.get_cpu_info().get("brand_raw", "Unknown CPU")
    cores = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024**3))
    log.info(f"{'MACHINE':<25}{cpu} ({cores} cores, {ram_gb} GB RAM)")


def info_gpu() -> None:
    """Log GPU details using PyTorch, if available."""
    try:
        import torch  # noqa: F401 # type: ignore[import] # PyTorch is optional

        if torch.cuda.is_available():
            log.info(f"{'GPU':<25}{torch.cuda.get_device_name(0)}")
        else:
            log.info(f"{'GPU':<25}No GPU available")
    except ImportError:
        log.info(f"{'GPU':<25}PyTorch not installed")


def info_system(hardware: bool = True, modules: list[str] | None = None) -> None:
    """
    Log full system information including OS, hardware, and software.

    Args:
        hardware (bool): Whether to include hardware info (CPU, RAM, GPU). Defaults to True.
        modules (list[str] | None): List of module names to log versions for. Defaults to DEFAULT_MODULES.
    """
    if hardware:
        info_hardware()
        info_gpu()
    info_os()
    info_software(modules)
    log.info(f"{'EXECUTION PATH':<25}{Path().absolute()}")
    log.info(f"{'EXECUTION DATE':<25}{time.ctime()}")


def get_memory_usage(obj: object) -> float:
    """
    Calculate and return memory usage of an object in megabytes.

    Args:
        obj (object): The object to analyze.

    Returns:
        float: Approximate memory usage in MB.
    """
    return round(sys.getsizeof(obj) / 1024**2, 3)
