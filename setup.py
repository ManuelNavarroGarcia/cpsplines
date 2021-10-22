from setuptools import find_packages, setup

setup(
    name="cpsplines",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "joblib",
        "matplotlib",
        "mosek",
        "numpy",
        "pandas",
        "scipy",
        "tensorly",
    ],
    extras_require={"dev": ["black", "ipykernel", "mypy", "pip-tools", "pytest"]},
)
