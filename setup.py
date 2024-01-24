import pathlib

from setuptools import find_packages, setup

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name="cpsplines",
    license="MIT",
    version="0.2.3",
    packages=find_packages(),
    description="Constrained P-splines",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Manuel Navarro García",
    author_email="manuelnavarrogithub@gmail.com",
    url="https://github.com/ManuelNavarroGarcia/cpsplines/",
    download_url="https://github.com/ManuelNavarroGarcia/cpsplines/archive/refs/tags/0.2.3.tar.gz",
    keywords=["P-splines", "MOSEK", "Python 3.9", "Constraints", "Optimization"],
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "joblib",
        "matplotlib",
        "mosek",
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "tensorly",
    ],
    extras_require={
        "dev": [
            "black[jupyter]",
            "ipykernel",
            "mypy",
            "pip-tools>=7.0.0",
            "pytest",
            "ruff",
        ]
    },
)
