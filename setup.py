from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"

setup(
    name="decal",
    version="0.1.0",
    description="DeCal: Decoupled Calibration for KV Cache Budgeting in Non-Stationary LLM Serving",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="DeCal Team",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20.0",
        "setproctitle>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    entry_points={
        "console_scripts": [
            "decal-run=decal.experiments.run_all:main",
        ],
    },
)
