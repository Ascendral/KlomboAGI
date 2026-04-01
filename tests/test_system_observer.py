"""Tests for system observer."""

from klomboagi.senses.system_observer import SystemObserver, SystemSnapshot, Anomaly


def test_take_snapshot():
    obs = SystemObserver()
    snap = obs.take_snapshot()
    assert isinstance(snap, SystemSnapshot)
    assert snap.cpu_percent >= 0
    assert snap.ram_percent > 0
    assert snap.process_count > 0


def test_check_anomalies_normal():
    obs = SystemObserver()
    snap = SystemSnapshot(
        timestamp=0, cpu_percent=20, ram_percent=40,
        ram_available_gb=16, disk_percent=50, disk_free_gb=200,
        net_bytes_sent=0, net_bytes_recv=0, process_count=100)
    anomalies = obs.check_anomalies(snap)
    assert len(anomalies) == 0


def test_check_anomalies_high_cpu():
    obs = SystemObserver()
    snap = SystemSnapshot(
        timestamp=0, cpu_percent=96, ram_percent=40,
        ram_available_gb=16, disk_percent=50, disk_free_gb=200,
        net_bytes_sent=0, net_bytes_recv=0, process_count=100,
        top_process="heavy_app", top_process_cpu=80)
    anomalies = obs.check_anomalies(snap)
    assert any(a.category == "cpu" and a.severity == "critical" for a in anomalies)


def test_get_summary():
    obs = SystemObserver()
    summary = obs.get_summary()
    assert "current" in summary
    assert "averages" in summary
    assert summary["current"]["cpu_percent"] >= 0
