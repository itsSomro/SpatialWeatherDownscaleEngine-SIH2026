"""
Root entry point redirecting to frontend/ui.py
Allows running either `streamlit run ui.py` or `streamlit run frontend/ui.py`.
"""
from pathlib import Path
import runpy

frontend_ui = Path(__file__).resolve().parent / "frontend" / "ui.py"
runpy.run_path(str(frontend_ui), run_name="__main__")
