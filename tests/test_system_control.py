"""Tests for system control."""

from klomboagi.senses.system_control import SystemControl


def test_safe_command():
    ctrl = SystemControl()
    result = ctrl.execute("echo hello")
    assert result.allowed
    assert "hello" in result.stdout


def test_blocked_sudo():
    ctrl = SystemControl()
    result = ctrl.execute("sudo rm -rf /")
    assert not result.allowed
    assert "Blocked" in result.blocked_reason


def test_blocked_rm_rf():
    ctrl = SystemControl()
    result = ctrl.execute("rm -rf /tmp/something")
    assert not result.allowed


def test_blocked_unknown_command():
    ctrl = SystemControl()
    result = ctrl.execute("malware --install")
    assert not result.allowed
    assert "not in allowlist" in result.blocked_reason


def test_list_processes():
    ctrl = SystemControl()
    procs = ctrl.list_processes()
    assert len(procs) > 0
    assert "pid" in procs[0]
    assert "name" in procs[0]


def test_network_status():
    ctrl = SystemControl()
    status = ctrl.network_status()
    assert "connected" in status
    assert "interfaces" in status
