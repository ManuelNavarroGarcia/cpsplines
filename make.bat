@echo off

set ENV_NAME=POC_MyC
set PYTHON_VERSION=3.13

set valid_commands= sync activate-env upgrade hooks ruff test mypy audit complexipy

for %%C in (%valid_commands%) do (
    if "%1"=="%%C" goto %%C
)

echo Invalid command: %1
goto :eof

:sync
    uv sync
    goto:eof

:activate-env
    .venv\Scripts\activate
    goto:eof

:upgrade
    uv lock --upgrade
    goto:eof

:hooks
    pre-commit install
    pre-commit run --all-files
    goto:eof

:ruff
    ruff format .
    ruff check --fix --show-fixes .
    goto:eof

:test
    coverage erase
    coverage run --source=src/ -m pytest
    coverage report -m
    goto:eof

:mypy
    mypy . --check-untyped-defs --strict --explicit-package-bases
    goto:eof

:audit
    call .venv\Scripts\activate
    set PIPAPI_PYTHON_LOCATION=%VIRTUAL_ENV%\Scripts\python.exe
    pip-audit
    goto:eof

:complexipy
    complexipy . --max-complexity-allowed 15