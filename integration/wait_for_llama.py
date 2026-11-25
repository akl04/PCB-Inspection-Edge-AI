import json
import time
import urllib.request

URL = "http://127.0.0.1:8091/health"

deadline = time.monotonic() + 180

print("Waiting for local Llama server...", flush=True)

while time.monotonic() < deadline:

    try:
        with urllib.request.urlopen(URL, timeout=3) as response:
            data = json.loads(response.read())

        if data.get("status") == "ok":
            print("Local Llama server is HEALTHY", flush=True)
            raise SystemExit(0)

    except Exception:
        pass

    time.sleep(2)

print("Llama server health check timed out.", flush=True)

raise SystemExit(1)
