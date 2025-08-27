from setuptools import setup, find_packages

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
    author="Sandia National Laboratories",
    author_email="",
    description="Vortex Lattice Method for Wind Turbine Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sandialabs/VorLap",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
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
