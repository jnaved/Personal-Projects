"""
Wrapper to run main_improved from the harr_cascade root directory.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Face_Detection"))
from main_improved import main

if __name__ == "__main__":
    main()
