# cpsplines


[TODO] Description of the project

## Project structure

The current version of the project is structured as follows:
* **cpsplines**: the main directory of the project, which consist of:
    * **fittings**: contains the smoothing algorithms.  
    * **graphics**: constituted by graphic methods to visualize the results.
    * **mosek_functions**: contains the functions used to define the optimization problems. 
    * **psplines**: composed by several methods that build the main objects of P-splines. 
    * **utils**: integrated by a miscellanea of files used for a variety of purposes (numerical computations, data processing, ...)
* **data**: a folder containing CSV files used in the real data numerical experiments.
* **examples**: a directory containing multiple numerical experiments, using both synthetic and real data sets. 
* **tests**: a folder including tests for the main methods of the project. 


## Package dependencies

`cpsplines` mainly depends on the following packages:

* [Joblib](https://joblib.readthedocs.io/).
* [Matplotlib](https://matplotlib.org/). 
* [MOSEK](https://www.mosek.com). **License Required**
* [Numpy](https://numpy.org/).
* [Pandas](https://pandas.pydata.org/).
* [Scipy](https://www.scipy.org/).
* [Tensorly](http://tensorly.org/).

MOSEK requires a license to be used. For research or educational purposes, a free yearly and renewable [academic license](https://www.mosek.com/products/academic-licenses/) is offered by the company. For other cases, a 30-day [trial license](https://www.mosek.com/try/) is available. According to MOSEK indications, the license file (`mosek.lic`) must be located at 
```
$HOME/mosek/mosek.lic                (Linux/OSX)
%USERPROFILE%\mosek\mosek.lic        (Windows)
```

## Installation

1. To clone the repository on your own device, use 

```
git clone https://github.com/ManuelNavarroGarcia/cpsplines.git
cd cpsplines
```

2. To install the dependencies, there are two options according to your installation preferences:

* Create and activate a virtual environment with `conda` (recommended)

```
conda env create -f env.yml
conda activate cpsplines
```

* Install the setuptools dependencies via `pip`

```
pip install -r requirements.txt
pip install -e .[dev]
```

3. If neccessary, add version requirements to existing dependencies or add new ones on `setup.py`. Then, update `requirements.txt` file using

```
pip-compile --extra dev > requirements.txt
```

and update the environment with `pip-sync`.

## Usage 

[TODO] How to use the repository


## Testing

The repository contains a folder with unit tests to guarantee the main methods meets their design and behave as intended. To launch the test suite, it is enough to enter `pytest`. If only one test file wants to be run, the syntax is given by 

```
pytest tests/test_<file_name>.py
```

Moreover, a GitHub Action runs automatically all the tests but `tests/test_solution.py` (which requires MOSEK license) when any commit is pushed on any Pull Request. 


## Contributing

Contributions to the repository are welcomed! Regardless of whether it is a small fix on the documentation or a notable feature to be included, I encourage you to develop your ideas and make this project greater. Even suggestions about the code structure are highly appreciated. Furthermore, users participating on these submissions will figure as contributors on this main page of the repository. 

There are many ways you can contribute on this repository:

* [Discussions](https://github.com/ManuelNavarroGarcia/cpsplines/discussions). To ask questions you are wondering about or share ideas, you can enter an existing discussion or open a new one. 

* [Issues](https://github.com/ManuelNavarroGarcia/cpsplines/issues). If you detect a bug or you want to propose an enhancement of the current version of the code, a issue with reproducible code and/or a detailed description is highly appreciated.

* [Pull Requests](https://github.com/ManuelNavarroGarcia/cpsplines/pulls). If you feel I am missing an important feature, either in the code or in the documentation, I encourage you to start a pull request developing this idea. Nevertheless, before starting any major new feature work, I suggest you to open an issue or start a discussion describing what you are planning to do. I note that, before starting a pull request, all unit test must pass on your local repository. 

## Contact Information and Citation

If you have encountered any problem or doubt while using `cpsplines`, please feel free to let me know by sending me an email:

* Name: Manuel Navarro García (he/his)
* Email: manuelnavarrogithub@gmail.com

The formal background of the models used in this project are either published in research paper or under current research. If these techniques are helpful to your own research, consider citing the related papers of the project:

```
@TECHREPORT{navarro2020,
  Author = {Navarro-Garc{\'ia}, M. and Guerrero, V. and Durban, M.},
  Title = {Constrained smoothing and out-of-range prediction using cubic {P}-splines: a semidefinite programming approach},
 Institution = {Universidad Carlos III de Madrid},
  Address ={\url{https://www.researchgate.net/publication/347836694_Constrained_smoothing_and_out-of-range_prediction_using_cubic_P_-splines_a_semidefinite_programming_approach4}},
  Year = {2020}
}
```

## Acknowledgements

[TODO] Include acknowledgements