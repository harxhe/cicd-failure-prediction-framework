"""
Pytest configuration for M2 tests.

Adds ml/ directory to sys.path so that imports like
'from config.settings import ...' and 'from src.data_loader import ...'
work regardless of how pytest is invoked.
"""

import sys
from pathlib import Path

# Add ml/ root to sys.path
sys.path.insert(0, str(Path(__file__).parent))
