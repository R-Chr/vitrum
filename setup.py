from setuptools import setup, find_packages

setup(
    name="vitrum",
    version="0.1.0",
    description="A brief description of your package",
    author="Rasmus Christensen",
    author_email="rasmusc@bio.aau.dk",
    url="https://github.com/R-Chr/vitrum",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy",
        "ase",
        "pandas",
        "scikit-learn",
        "scipy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
