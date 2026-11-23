"""
Streamlit Community Cloud entrypoint.
Loads and runs frontend/app.py with absolute path resolution.
"""
import sys
import runpy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

frontend_app = ROOT_DIR / "frontend" / "app.py"
runpy.run_path(str(frontend_app), run_name="__main__")
