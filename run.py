"""
Smart Resume — Entry Point
===========================
Place this file at:  C:\\Users\\anasa\\smart_resume\\main.py

Run with:
    python main.py

This avoids all sys.path / subprocess issues because Python
runs from the project root where core/ api/ ui/ all live.
"""

import sys
import os

# Guarantee project root is always on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Now safe to import
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",          # points to app.py in this same folder
        host="127.0.0.1",
        port=8000,
        reload=False,       # reload=False avoids the subprocess path problem entirely
    )