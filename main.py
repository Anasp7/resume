import sys
import os

# Fix path BEFORE any other import
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Load .env file if present (so GROQ_API_KEY persists without re-setting each terminal)
env_file = os.path.join(ROOT, ".env")
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # Strip surrounding quotes: KEY="value" or KEY='value'
            if len(v) >= 2 and v[0] in ('"', "'") and v[-1] == v[0]:
                v = v[1:-1]
            os.environ.setdefault(k, v)
    # Confirm key loaded
    if os.environ.get("GROQ_API_KEY"):
        print(f"[main] GROQ_API_KEY loaded from .env ({len(os.environ['GROQ_API_KEY'])} chars)")
    else:
        print("[main] WARNING: GROQ_API_KEY not found in .env — check file exists and key is correct")

import uvicorn
from app import app  # import the actual object, not a string

if __name__ == "__main__":
    uvicorn.run(
        app,              # object reference — no subprocess, no path issues
        host="127.0.0.1",
        port=8001,
        reload=False,
    )