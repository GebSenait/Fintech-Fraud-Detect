"""
Utility functions for path management and imports.
"""
from pathlib import Path
import sys


def setup_paths():
    """
    Add src directory to Python path for imports.
    Works correctly in both Jupyter notebooks and regular Python scripts.
    """
    # Get the project root (parent of src directory)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    # Add src to path if not already there
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    return project_root, src_path

