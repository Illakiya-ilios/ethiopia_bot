"""Quick live smoke test for the chat API.

Start the server first (in another terminal):

    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

Then run:

    python test_api.py
    python test_api.py --base http://localhost:8000 --user-id 2
"""

from __future__ import annotations

import argparse
import json
import urllib.request


def call(base: str, message: str, user_id: int | None) -> dict:
    payload = {"message": message}
    if user_id is not None:
        payload["user_id"] = user_id

    req = urllib.request.Request(
        f"{base}/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--user-id", type=int, default=2)
    args = parser.parse_args()

    # 1) health
    with urllib.request.urlopen(f"{args.base}/health", timeout=30) as r:
        print("HEALTH:", r.read().decode())

    tests = [
        ("navigate", "take me to the registration page", args.user_id),
        ("discovery", "what are the top attractions in Lalibela?", None),
        ("bookings", "what packages have I booked?", args.user_id),
    ]

    for label, message, uid in tests:
        print(f"\n=== {label.upper()} ===")
        print("Q:", message)
        try:
            result = call(args.base, message, uid)
            print("route     :", result.get("route"))
            print("answer    :", result.get("answer"))
            print("navigation:", result.get("navigation"))
        except Exception as exc:  # noqa: BLE001
            print("ERROR:", exc)


if __name__ == "__main__":
    main()
