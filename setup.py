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
        "mp_api",
        "fireworks",
        "jobflow",
        "atomate2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
