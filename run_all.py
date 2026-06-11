#!/usr/bin/env python3
"""Project-root wrapper for the full DeCal experiment suite."""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from decal.experiments.run_all import main


if __name__ == "__main__":
    main()
