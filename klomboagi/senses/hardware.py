"""
Hardware Sense — KlomboAGI's awareness of its own physical substrate.

The first step toward an OS-level cognitive entity: know what machine
you're running on. CPU, RAM, GPU, storage, network, peripherals.

Not always-on polling. Called on boot and on-demand.
macOS-first (Mac Studio M1 Max), falls back gracefully on other platforms.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field


@dataclass
class CPUInfo:
    model: str = "unknown"
    cores_physical: int = 0
    cores_logical: int = 0
    architecture: str = "unknown"
    usage_percent: float = 0.0


@dataclass
class RAMInfo:
    total_gb: float = 0.0
    available_gb: float = 0.0
    usage_percent: float = 0.0


@dataclass
class GPUInfo:
    model: str = "unknown"
    cores: int = 0
    vram_gb: float = 0.0
    metal_support: bool = False


@dataclass
class StorageInfo:
    total_gb: float = 0.0
    free_gb: float = 0.0
    usage_percent: float = 0.0
    mount_point: str = "/"


@dataclass
class NetworkInterface:
    name: str = ""
    ip: str = ""
    mac: str = ""
    is_up: bool = False


@dataclass
class DisplayInfo:
    name: str = "unknown"
    resolution: str = "unknown"


@dataclass
class PeripheralInfo:
    name: str = ""
    vendor: str = ""
    bus: str = ""  # USB, Thunderbolt, etc.


@dataclass
class HardwareState:
    """Complete snapshot of the machine's hardware."""
    hostname: str = "unknown"
    os_name: str = "unknown"
    os_version: str = "unknown"
    cpu: CPUInfo = field(default_factory=CPUInfo)
    ram: RAMInfo = field(default_factory=RAMInfo)
    gpu: GPUInfo = field(default_factory=GPUInfo)
    storage: list[StorageInfo] = field(default_factory=list)
    network: list[NetworkInterface] = field(default_factory=list)
    displays: list[DisplayInfo] = field(default_factory=list)
    peripherals: list[PeripheralInfo] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Host: {self.hostname} ({self.os_name} {self.os_version})",
            f"CPU: {self.cpu.model} ({self.cpu.cores_physical}P/{self.cpu.cores_logical}L cores, {self.cpu.usage_percent:.0f}% load)",
            f"RAM: {self.ram.available_gb:.1f}/{self.ram.total_gb:.1f} GB available ({self.ram.usage_percent:.0f}% used)",
            f"GPU: {self.gpu.model} ({self.gpu.cores} cores, {self.gpu.vram_gb:.0f} GB VRAM)",
        ]
        for s in self.storage:
            lines.append(f"Storage [{s.mount_point}]: {s.free_gb:.0f}/{s.total_gb:.0f} GB free ({s.usage_percent:.0f}% used)")
        for n in self.network:
            if n.is_up and n.ip:
                lines.append(f"Network [{n.name}]: {n.ip}")
        if self.displays:
            lines.append(f"Displays: {len(self.displays)} — " +
                         ", ".join(f"{d.name} ({d.resolution})" for d in self.displays))
        if self.peripherals:
            lines.append(f"Peripherals: {len(self.peripherals)} — " +
                         ", ".join(p.name for p in self.peripherals[:5]))
        return "\n".join(lines)


class HardwareSense:
    """Scans and reports the machine's hardware state."""

    def scan(self) -> HardwareState:
        """Full hardware scan. Returns a complete snapshot."""
        state = HardwareState()
        state.hostname = platform.node()
        state.os_name = platform.system()
        state.os_version = platform.release()

        self._scan_cpu(state)
        self._scan_ram(state)
        self._scan_gpu(state)
        self._scan_storage(state)
        self._scan_network(state)
        self._scan_displays(state)
        self._scan_peripherals(state)

        return state

    def _scan_cpu(self, state: HardwareState) -> None:
        import psutil
        state.cpu.cores_physical = psutil.cpu_count(logical=False) or 0
        state.cpu.cores_logical = psutil.cpu_count(logical=True) or 0
        state.cpu.usage_percent = psutil.cpu_percent(interval=0.1)
        state.cpu.architecture = platform.machine()

        # macOS: get chip name
        if platform.system() == "Darwin":
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    text=True, timeout=5).strip()
                state.cpu.model = out
            except Exception:
                state.cpu.model = platform.processor() or "unknown"
        else:
            state.cpu.model = platform.processor() or "unknown"

    def _scan_ram(self, state: HardwareState) -> None:
        import psutil
        mem = psutil.virtual_memory()
        state.ram.total_gb = mem.total / (1024 ** 3)
        state.ram.available_gb = mem.available / (1024 ** 3)
        state.ram.usage_percent = mem.percent

    def _scan_gpu(self, state: HardwareState) -> None:
        if platform.system() != "Darwin":
            return
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                text=True, timeout=10)
            import json
            data = json.loads(out)
            displays = data.get("SPDisplaysDataType", [])
            if displays:
                gpu = displays[0]
                state.gpu.model = gpu.get("sppci_model", "unknown")
                cores_str = gpu.get("sppci_cores", "0")
                # Handle "30-core" style strings
                if isinstance(cores_str, str):
                    cores_str = cores_str.split("-")[0].split()[0]
                state.gpu.cores = int(cores_str) if cores_str.isdigit() else 0
                state.gpu.metal_support = "spdisplays_metal" in gpu
                # Unified memory on Apple Silicon — VRAM = shared with RAM
                if "apple" in state.cpu.model.lower() or "m1" in state.cpu.model.lower() or "m2" in state.cpu.model.lower():
                    state.gpu.vram_gb = state.ram.total_gb
        except Exception:
            pass

    def _scan_storage(self, state: HardwareState) -> None:
        import psutil
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                state.storage.append(StorageInfo(
                    total_gb=usage.total / (1024 ** 3),
                    free_gb=usage.free / (1024 ** 3),
                    usage_percent=usage.percent,
                    mount_point=part.mountpoint,
                ))
            except Exception:
                continue

    def _scan_network(self, state: HardwareState) -> None:
        import psutil
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        for iface, addr_list in addrs.items():
            if iface.startswith("lo"):
                continue
            ni = NetworkInterface(name=iface)
            ni.is_up = stats.get(iface, type("", (), {"isup": False})).isup
            for addr in addr_list:
                if addr.family.name == "AF_INET":
                    ni.ip = addr.address
                elif addr.family.name == "AF_LINK":
                    ni.mac = addr.address
            if ni.ip or ni.is_up:
                state.network.append(ni)

    def _scan_displays(self, state: HardwareState) -> None:
        if platform.system() != "Darwin":
            return
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                text=True, timeout=10)
            import json
            data = json.loads(out)
            for gpu_data in data.get("SPDisplaysDataType", []):
                for display in gpu_data.get("spdisplays_ndrvs", []):
                    name = display.get("_name", "unknown")
                    res = display.get("_spdisplays_resolution", "unknown")
                    state.displays.append(DisplayInfo(name=name, resolution=res))
        except Exception:
            pass

    def _scan_peripherals(self, state: HardwareState) -> None:
        if platform.system() != "Darwin":
            return
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPUSBDataType", "-json"],
                text=True, timeout=10)
            import json
            data = json.loads(out)
            self._extract_usb_devices(data.get("SPUSBDataType", []), state)
        except Exception:
            pass

    def _extract_usb_devices(self, items: list, state: HardwareState) -> None:
        for item in items:
            name = item.get("_name", "")
            vendor = item.get("manufacturer", "")
            if name and not name.startswith("USB") and vendor:
                state.peripherals.append(PeripheralInfo(
                    name=name, vendor=vendor, bus="USB"))
            # Recurse into hubs
            for sub in item.get("_items", []):
                self._extract_usb_devices([sub], state)
