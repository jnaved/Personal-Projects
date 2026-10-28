"""
Root launcher for run_improved.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Face_Detection"))
from run_improved import main

if __name__ == "__main__":
    main()
