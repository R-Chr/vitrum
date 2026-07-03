from setuptools import setup, find_packages

setup(
    name="vitrum",
    version="0.1.01",
    description="vitrum is a package for generating input data and analyzing simulation data of glass structures",
    author="Rasmus Christensen",
    author_email="rasmusc@bio.aau.dk",
    url="https://github.com/R-Chr/vitrum",
    license="MIT",
    packages=find_packages("src"),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={"": "src"},
    package_data={"vitrum": ["./scattering_lengths.csv", "./x_ray_scattering_factor_coefficients.csv"]},
    include_package_data=True,
    install_requires=[
        "numpy",
        "ase",
        "pandas",
        "scikit-learn",
        "scipy",
        "pymatgen",
        "ruamel-yaml==0.17.9",
        "numba",
        "matplotlib",
    ],
    extras_require={
        "workflows": ["fireworks", "jobflow", "atomate2"],
        "volume_estimation": ["atomate2", "mp_api"],
        # diode (used by persistent_homology.py) is not a normal PyPI package;
        # see docs/vitrum/install.md for its manual install instructions.
        "persistent_homology": ["dionysus"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
