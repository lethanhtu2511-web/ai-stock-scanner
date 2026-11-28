"""Standalone launcher for BOT 2 — Stock Behavior Research.

Run independently from the original AI Stock Scanner:
    streamlit run BOT_2.py

This launches the research UI only. It does not start the original scanner,
automatically scan at startup, or place orders.
"""
from pathlib import Path
import runpy

research_page = Path(__file__).resolve().parent / "pages" / "2_Nghien_cuu_quy_luat.py"
runpy.run_path(str(research_page), run_name="__main__")
