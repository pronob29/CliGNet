"""
setup.py
--------
Package setup for CLiGNet.

Install in editable mode from repo root:
    pip install -e .

This allows importing from the src/ package tree directly in all scripts.
"""

from setuptools import setup, find_packages

setup(
    name="clignet",
    version="1.0.0",
    description=(
        "CLiGNet: Clinical Label-interaction Graph Network for "
        "Multi-class Medical Specialty Classification"
    ),
    author="Pronob",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "captum>=0.6.0",
        "scikit-multilearn>=0.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "interpretability": ["scispacy>=0.5.3"],
        "ablation":         ["imbalanced-learn>=0.11.0"],
        "notebooks":        ["jupyter>=1.0.0", "ipykernel>=6.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Intended Audience :: Science/Research",
    ],
)
