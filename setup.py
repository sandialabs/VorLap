from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).with_name("README.md")
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="vorlap",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "h5py",
        "plotly",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov"],
        "docs": ["mkdocs", "mkdocs-material", "mkdocstrings[python]", "pymdown-extensions"],
        "gui": ["matplotlib", "kaleido", "pyinstaller"],
    },
    author="Sandia National Laboratories",
    author_email="",
    description="Generalized Vortex Overlap Fluid Structure Interaction Prediction Code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/VorLap",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "vorlap": [
            "airfoils/*.csv",
            "airfoils/*.h5",
            "componentsHAWT/*.csv",
            "componentsHVAWT/*.csv",
            "componentsVAWT/*.csv",
        ],
    },
)
