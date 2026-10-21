"""
make_payload.py -- writes a valid example request to payload.json.

Why a file? Typing 30 numbers into a terminal is painful, and quote rules
are different in PowerShell and bash. With a file you can simply run:
    curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@payload.json"

Run it from the project root:
    python tests/make_payload.py
"""

import json
import sys
from pathlib import Path

# Make "import app" work when this file is run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.features import EXAMPLE_PAYLOAD  # noqa: E402

Path("payload.json").write_text(json.dumps(EXAMPLE_PAYLOAD), encoding="utf-8")
print("Wrote payload.json with", len(EXAMPLE_PAYLOAD), "features")
