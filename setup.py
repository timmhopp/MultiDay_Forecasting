"""
Setup configuration for FusedODModel package.
"""

from setuptools import setup, find_packages

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fused-od-model",
    version="1.0.0",
    author="Timm Hopp",
    author_email="",
    description="Multi-Scale Origin-Destination Flow Prediction using Graph Attention Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/FusedODModel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fusedod-train=train_main:main",
            "fusedod-eval=eval_main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)