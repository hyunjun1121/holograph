"""
HOLOGRAPH: Active Causal Discovery via Continuous Sheaf Alignment and Natural Gradient Descent
"""

from setuptools import setup, find_packages

setup(
    name="holograph",
    version="0.1.0",
    description="Active Causal Discovery via Continuous Sheaf Alignment and Natural Gradient Descent",
    author="HOLOGRAPH Team",
    python_requires=">=3.8",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "holograph=holograph.run:main",
        ],
    },
)
