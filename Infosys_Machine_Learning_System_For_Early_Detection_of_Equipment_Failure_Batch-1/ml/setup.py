from setuptools import setup, find_packages

setup(
    name="flash-crash-watchdog",
    version="0.4.0",
    description="Real-time flash-crash detector on limit-order-book streams",
    author="Z.ai Quant Research",
    license="Apache-2.0",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "torch>=2.1",
        "websockets>=12.0",
        "pyarrow>=14.0",
        "pyyaml>=6.0",
        "joblib>=1.3",
    ],
    entry_points={
        "console_scripts": [
            "flash-crash-watchdog=flash_crash_watchdog.cli:main",
        ],
    },
)
