# cpsplines

A template for python repositories using `setup.py`, tests and usual workflows

## Components

The repository is structured into the following directories:

- `/cpsplines`: where the python code is.
- `/data`: where the data files are (or their DVC pointers).
- `/tests`: python code for testing via `pytest`.

Conveniently, a set of workflows via Github actions are already installed:

- `black`: code formatting
- `pytest`: automatically discover and runs tests in `tests/`
- `mypy`: automatically runs type checking

## Install dependencies

There are two options, depending on whether you use conda or not:

- Conda: 
  ```
  conda env create -f env.yml
  ```

- Pip: 
  ```
  pip install -r requirements.txt
  pip install -e .[dev]
  ```

The difference between conda and pip is that conda will create an isolated environment while pip will install all the dependencies in the current Python env. This might be a conda environment or any other Python env created by other tools. If you already have the dependencies installed, you can update it to reflect the last version of the packages in the `requirements.txt` with `pip-sync`. 

## Add dependencies

Add abstract dependency to `setup.py`. If neccessary, add version requirements but try to be as flexible as possible

- Update `requirements.txt`: `pip-compile --extra dev > requirements.txt`
- Update environment: `pip-sync`

## Run tests

- Run tests: `pytest`


