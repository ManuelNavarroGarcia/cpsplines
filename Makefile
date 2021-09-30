env:
	conda env create -f env.yml

install:
	pip install -e .[dev]
	
compile:
	pip-compile --extra dev > requirements.txt
