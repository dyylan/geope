import sys
from pathlib import Path

# Ensure the src/ layout is importable without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
