from setuptools import setup, find_packages

setup(
    name="PhysioSignal",
    version="1.1.0",
    description="Librería para procesamiento de señales EEG, ECG y EMG",
    author="Agustín Quintana",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "matplotlib>=3.5",
        "scipy>=1.9"
    ],
    python_requires=">=3.10"
)