from setuptools import setup, find_packages

setup(
    name="MolTools",
    version="0.0.1",
    author="David Meijer",
    author_email="david.meijer@wur.nl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "rdkit>=2022.03.5"
    ],
    python_requires=">=3.10"
)