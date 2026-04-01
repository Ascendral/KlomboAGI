"""
KlomboAGI Menu Bar App — always visible, always accessible.

Lives in the macOS menu bar. Click to chat, see status, get alerts.
System notifications for anomalies. Brain icon shows it's alive.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
import urllib.error


# Use PyObjC for native macOS menu bar (no external deps needed on macOS)
try:
    import AppKit
    import Foundation
    import objc
    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False


PORT = 3141
BASE = f"http://localhost:{PORT}"


def _api_get(path: str) -> dict | None:
    try:
        r = urllib.request.urlopen(f"{BASE}{path}", timeout=3)
        return json.loads(r.read())
    except Exception:
        return None


def _api_post(path: str, data: dict) -> dict | None:
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{BASE}{path}", data=body,
            headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=15)
        return json.loads(r.read())
    except Exception:
        return None


def _notify(title: str, message: str) -> None:
    """Send a macOS notification."""
    try:
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "{title}"'
        ], capture_output=True, timeout=5)
    except Exception:
        pass


if HAS_APPKIT:
    class KlomboMenuBar(AppKit.NSObject):
        """Native macOS menu bar app using PyObjC."""

        def init(self):
            self = objc.super(KlomboMenuBar, self).init()
            if self is None:
                return None

            # Create status bar item
            self.statusbar = AppKit.NSStatusBar.systemStatusBar()
            self.item = self.statusbar.statusItemWithLength_(
                AppKit.NSVariableStatusItemLength)
            self.item.setTitle_("🧠")

            # Create menu
            self.menu = AppKit.NSMenu.alloc().init()
            self._build_menu()
            self.item.setMenu_(self.menu)

            # Start background monitor
            self._last_anomaly_count = 0
            self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
            self._monitor_thread.start()

            return self

        def _build_menu(self):
            self.menu.removeAllItems()

            # Status line
            self.status_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "KlomboAGI — Checking...", None, "")
            self.status_item.setEnabled_(False)
            self.menu.addItem_(self.status_item)

            self.menu.addItem_(AppKit.NSMenuItem.separatorItem())

            # Chat
            chat_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Open Chat...", "openChat:", "c")
            chat_item.setTarget_(self)
            self.menu.addItem_(chat_item)

            # Dashboard
            dash_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Open Dashboard", "openDashboard:", "d")
            dash_item.setTarget_(self)
            self.menu.addItem_(dash_item)

            self.menu.addItem_(AppKit.NSMenuItem.separatorItem())

            # System info
            self.cpu_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "CPU: --", None, "")
            self.cpu_item.setEnabled_(False)
            self.menu.addItem_(self.cpu_item)

            self.ram_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "RAM: --", None, "")
            self.ram_item.setEnabled_(False)
            self.menu.addItem_(self.ram_item)

            self.beliefs_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Beliefs: --", None, "")
            self.beliefs_item.setEnabled_(False)
            self.menu.addItem_(self.beliefs_item)

            self.turns_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Turns: --", None, "")
            self.turns_item.setEnabled_(False)
            self.menu.addItem_(self.turns_item)

            self.menu.addItem_(AppKit.NSMenuItem.separatorItem())

            # Quit
            quit_item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Quit Menu Bar", "quitApp:", "q")
            quit_item.setTarget_(self)
            self.menu.addItem_(quit_item)

        @objc.python_method
        def _update_status(self):
            health = _api_get("/health")
            if health and health.get("status") == "alive":
                cpu = health.get("cpu_percent", "?")
                ram = health.get("ram_percent", "?")
                turns = health.get("total_turns", 0)

                self.status_item.setTitle_(f"KlomboAGI — Alive")
                self.cpu_item.setTitle_(f"CPU: {cpu}%")
                self.ram_item.setTitle_(f"RAM: {ram}%")
                self.turns_item.setTitle_(f"Turns: {turns}")
                self.item.setTitle_("🧠")

                beliefs = _api_get("/beliefs")
                if beliefs:
                    self.beliefs_item.setTitle_(f"Beliefs: {beliefs.get('count', '?')}")

                # Check anomalies
                obs = _api_get("/observe")
                if obs:
                    anomalies = obs.get("anomalies", [])
                    if len(anomalies) > self._last_anomaly_count:
                        for a in anomalies[self._last_anomaly_count:]:
                            _notify("KlomboAGI", a.get("message", "Anomaly detected"))
                        self._last_anomaly_count = len(anomalies)
                        self.item.setTitle_("🧠⚠️")
            else:
                self.status_item.setTitle_("KlomboAGI — Offline")
                self.item.setTitle_("🧠💤")

        @objc.python_method
        def _monitor(self):
            while True:
                try:
                    self.performSelectorOnMainThread_withObject_waitUntilDone_(
                        "_updateStatusObjC:", None, False)
                except Exception:
                    pass
                time.sleep(10)

        def _updateStatusObjC_(self, _):
            self._update_status()

        def openChat_(self, sender):
            # Open terminal with chat
            subprocess.Popen([
                "osascript", "-e",
                'tell application "Terminal" to do script '
                '"cd /opt/klomboagi && python3 -m klomboagi chat"'
            ])

        def openDashboard_(self, sender):
            subprocess.Popen(["open", f"http://localhost:{PORT}"])

        def quitApp_(self, sender):
            AppKit.NSApplication.sharedApplication().terminate_(self)


def run_menubar():
    """Start the menu bar app."""
    if not HAS_APPKIT:
        print("PyObjC not available. Install with: pip3 install pyobjc-framework-Cocoa")
        print("Falling back to notification-only mode...")
        _run_notification_only()
        return

    app = AppKit.NSApplication.sharedApplication()
    delegate = KlomboMenuBar.alloc().init()
    app.setDelegate_(delegate)

    # Don't show in dock
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    _notify("KlomboAGI", "Brain is active in the menu bar")
    app.run()


def _run_notification_only():
    """Fallback: just send notifications for anomalies."""
    last_count = 0
    _notify("KlomboAGI", "Monitoring started (notification-only mode)")

    while True:
        try:
            obs = _api_get("/observe")
            if obs:
                anomalies = obs.get("anomalies", [])
                if len(anomalies) > last_count:
                    for a in anomalies[last_count:]:
                        _notify("KlomboAGI", a.get("message", "Anomaly"))
                    last_count = len(anomalies)
        except Exception:
            pass
        time.sleep(30)


if __name__ == "__main__":
    run_menubar()
