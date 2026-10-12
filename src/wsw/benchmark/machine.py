"""Facts about the machine and checkout a result was produced on, stored with every result."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "scipy", "scikit-learn", "nicegui", "snorkel", "torch")


def _windows_cpu_name() -> str | None:
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        return None


def _windows_physical_cores() -> int | None:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).NumberOfCores",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return sum(int(x) for x in out.split())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _windows_ram_bytes() -> int | None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return int(status.ullTotalPhys)


def _linux_cpuinfo() -> tuple[str | None, int | None]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return None, None
    name = next(
        (ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("model name")),
        None,
    )
    cores = {
        (block.get("physical id"), block.get("core id"))
        for block in (
            dict(
                (k.strip(), v.strip())
                for k, v in (ln.split(":", 1) for ln in chunk.splitlines() if ":" in ln)
            )
            for chunk in text.split("\n\n")
        )
        if "core id" in block
    }
    return name, len(cores) or None


def _linux_ram_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def machine_info() -> dict[str, object]:
    if sys.platform == "win32":
        cpu, cores, ram = _windows_cpu_name(), _windows_physical_cores(), _windows_ram_bytes()
    else:
        cpu, cores = _linux_cpuinfo()
        ram = _linux_ram_bytes()
    versions: dict[str, str] = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return {
        "cpu": cpu or platform.processor() or "unknown",
        "physical_cores": cores,
        "logical_cpus": os.cpu_count(),
        "ram_gb": None if ram is None else round(ram / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "accelerator": "none (CPU only)",
    }


def git_state(cwd: Path | None = None) -> dict[str, object]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_sha": sha or "unknown", "git_dirty": bool(status) if status is not None else None}


def local_now() -> datetime:
    """Local wall-clock time with its UTC offset (results are dated in local time)."""
    return datetime.now().astimezone()
