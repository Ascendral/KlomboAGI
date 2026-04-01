"""
KlomboAGI Chat Shell — talk to the brain naturally.

Usage: python3 -m klomboagi chat [--port 3141]

Connects to the running KlomboAGI server and gives you a clean
conversation interface. Just type and talk.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error


def chat(port: int = 3141) -> None:
    base = f"http://localhost:{port}"

    # Check server is alive
    try:
        r = urllib.request.urlopen(f"{base}/health", timeout=3)
        data = json.loads(r.read())
        if data.get("status") != "alive":
            print("KlomboAGI is not responding.")
            sys.exit(1)
    except Exception:
        print(f"Can't reach KlomboAGI on port {port}.")
        print("Start it with: python3 -m klomboagi serve")
        sys.exit(1)

    # Get hardware info for greeting
    try:
        r = urllib.request.urlopen(f"{base}/hardware", timeout=3)
        hw = json.loads(r.read())
        cpu = hw.get("cpu", {}).get("model", "unknown")
        ram = hw.get("ram", {}).get("total_gb", 0)
        print(f"\n  KlomboAGI — running on {cpu}, {ram:.0f}GB RAM")
    except Exception:
        print("\n  KlomboAGI")

    print("  Type 'quit' to exit.\n")

    while True:
        try:
            msg = input("you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            break

        # Special commands
        if msg.lower() == "/status":
            try:
                r = urllib.request.urlopen(f"{base}/status", timeout=10)
                print(json.loads(r.read()).get("status", ""))
            except Exception as e:
                print(f"  Error: {e}")
            continue

        if msg.lower() == "/hardware":
            try:
                r = urllib.request.urlopen(f"{base}/hardware", timeout=10)
                print(json.loads(r.read()).get("summary", ""))
            except Exception as e:
                print(f"  Error: {e}")
            continue

        if msg.lower() == "/beliefs":
            try:
                r = urllib.request.urlopen(f"{base}/beliefs", timeout=10)
                data = json.loads(r.read())
                print(f"  {data['count']} beliefs")
                for b in data["beliefs"][:10]:
                    print(f"    {b['statement']}")
            except Exception as e:
                print(f"  Error: {e}")
            continue

        # Send message to brain
        try:
            body = json.dumps({"message": msg}).encode()
            req = urllib.request.Request(
                f"{base}/hear", data=body,
                headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=30)
            resp = json.loads(r.read())
            print(f"\nklombo: {resp['response']}\n")
        except urllib.error.HTTPError as e:
            err = json.loads(e.read())
            print(f"  Error: {err.get('error', 'unknown')}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    chat()
