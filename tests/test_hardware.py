"""Tests for the hardware awareness sense."""

from unittest.mock import patch, MagicMock
from klomboagi.senses.hardware import (
    HardwareSense, HardwareState, CPUInfo, RAMInfo, GPUInfo,
    StorageInfo, NetworkInterface, DisplayInfo, PeripheralInfo,
)


def test_hardware_state_summary():
    """Summary renders all fields correctly."""
    state = HardwareState(
        hostname="test-machine",
        os_name="Darwin",
        os_version="25.0.0",
        cpu=CPUInfo(model="Apple M1 Max", cores_physical=10, cores_logical=10,
                    architecture="arm64", usage_percent=15.0),
        ram=RAMInfo(total_gb=32.0, available_gb=16.0, usage_percent=50.0),
        gpu=GPUInfo(model="Apple M1 Max", cores=24, vram_gb=32.0, metal_support=True),
        storage=[StorageInfo(total_gb=460.0, free_gb=300.0, usage_percent=35.0, mount_point="/")],
        network=[NetworkInterface(name="en0", ip="192.168.1.1", is_up=True)],
        displays=[DisplayInfo(name="DELL", resolution="1920x1080")],
        peripherals=[PeripheralInfo(name="Keyboard", vendor="Apple", bus="USB")],
    )
    summary = state.summary()
    assert "test-machine" in summary
    assert "Apple M1 Max" in summary
    assert "32.0 GB" in summary
    assert "24 cores" in summary
    assert "en0" in summary
    assert "DELL" in summary
    assert "Keyboard" in summary


def test_scan_returns_hardware_state():
    """scan() returns a populated HardwareState."""
    hw = HardwareSense()
    state = hw.scan()
    assert isinstance(state, HardwareState)
    assert state.hostname != "unknown"
    assert state.cpu.cores_logical > 0
    assert state.ram.total_gb > 0
    assert len(state.storage) > 0


def test_empty_state_summary():
    """Empty state renders without crashing."""
    state = HardwareState()
    summary = state.summary()
    assert "unknown" in summary
