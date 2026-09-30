"""Fetch the React site and extract its client-side routes.

The site is a single-page app, so routes live inside the compiled JS bundle,
not the HTML. This script:

    1. Downloads the index HTML.
    2. Finds the referenced JS bundle(s).
    3. Downloads them and scans for route-like paths.

Run:
    python discover_routes.py
    python discover_routes.py --base http://4.240.116.35:3000
"""

from __future__ import annotations

import argparse
import re
import urllib.request


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "route-finder"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://4.240.116.35:3000")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    print(f"Fetching {base}/ ...")
    html = fetch(f"{base}/")

    # Find JS bundles referenced in the HTML.
    js_files = re.findall(r'src="([^"]+\.js)"', html)
    js_files += re.findall(r'"([^"]+\.js)"', html)
    js_files = sorted(set(js_files))
    print(f"Found {len(js_files)} JS reference(s).")

    routes: set[str] = set()

    for js in js_files:
        js_url = js if js.startswith("http") else f"{base}/{js.lstrip('/')}"
        try:
            print(f"  scanning {js_url}")
            code = fetch(js_url)
        except Exception as exc:  # noqa: BLE001
            print(f"    (skip: {exc})")
            continue

        # Common React Router patterns: path:"/foo", to:"/foo", "#/foo"
        for pat in [
            r'path\s*:\s*"([^"]+)"',
            r"path\s*:\s*'([^']+)'",
            r'to\s*:\s*"(/[^"]+)"',
            r'"(#/?[a-zA-Z0-9\-_/:]+)"',
            r'"(/[a-zA-Z0-9\-_/:]+)"',
        ]:
            for m in re.findall(pat, code):
                if 1 < len(m) < 60 and not m.endswith((".js", ".css", ".png",
                                                        ".svg", ".ico")):
                    routes.add(m)

    print("\n=== Candidate routes ===")
    for r in sorted(routes):
        print(" ", r)
    if not routes:
        print("  (none found — the bundle may be minified differently)")


if __name__ == "__main__":
    main()
