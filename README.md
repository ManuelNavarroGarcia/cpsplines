# cpsplines

`cpsplines` is a Python module to perform constrained regression under shape constraints on the component functions of the dependent variable. It is assumed that the smooth hypersurface to be estimated is defined through a reduced-rank basis (B−splines) and fitted via a penalized splines approach (P−splines). To embed requirements about the sign of any order partial derivative, the constraints are included in the fitting process as hard constraints, yielding a semidefinite optimization model. In particular, the problem of estimating the component function using this approach is stated as a convex semidefinite optimization problem with a quadratic objective function, which can be easily reformulated as a conic optimization problem. 

Sign related constraints are imposed using a well-known result carried out by Bertsimas and Popescu, 2002. This enables to enforce non-negativity of a univariate polynomial over a finite interval, which can be straightforwardly extended to the sign of any higher order derivative. When only one covariate is related to the response variable, these constraints are successfully fulfilled over the whole domain of the regressor sample. However, when facing multiple regression, this equivalence does not hold, so alternative approaches must be developed. The proposed framework in this repository uses the equivalence relation for univariate polynomials by imposing the constraints over a finite set of curves which belong to the hypersurface. 

At present, `cpsplines` can handle constrained regression problems for data lying on large grids. In this setting, the smooth hypersurface is constructed from the tensor products of B-splines basis along each axis, which allows to develop efficient algorithms accelerating the computations (Currie, Durban and Eilers, 2006). On this repository, fitting procedure is performed using the method `GridCPsplines`, whose main features are the following:

* Arbitrary knot sequence length to construct the B-spline basis. 
* Arbitrary B-spline basis degrees. 
* Arbitrary difference orders on the penalty term.
* Out-of-range prediction (backwards and forward) along every dimension (Currie and Durban, 2004), and the constraints are enforced either on the fitting and the prediction region. 
* The smoothing parameters are selected as the minimizer of the Generalized Cross Validation criteria, but this routine can be done either by choosing the best candidate out of a set of candidates or by finding them using numerical methods. 
* Enforcing sign related constraints over the fitting and prediction range (if prediction is required). Arbitrary number of sign constraints can be imposed along each regressor. 
* Enforcing the hypersurface (or any partial derivative) attains a certain value at a certain point. 

For solving the optimization problems, [MOSEK](https://www.mosek.com) optimization software is used. 

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

```
np.random.seed(6)
x4 = np.linspace(0, 1, 51)
y4 =  (2 * x4 - 1) ** 3 + np.random.normal(0, 0.25, 51)

example4_1 = GridCPsplines(
    deg=(3,),
    ord_d=(2,),
    n_int=(10,),
    sp_args={"options": {"ftol": 1e-12}},
)
example4_1.fit(x=(x4,), y=y4)

example4_2 = GridCPsplines(
    deg=(3,),
    ord_d=(2,),
    n_int=(10,),
    sp_args={"options": {"ftol": 1e-12}},
    int_constraints={0: {1: {"+": 0}}}
)
example4_2.fit(x=(x4,), y=y4)

plot_4 = plot_curves(
    fittings=(example4_1, example4_2),
    col_curve=("g", "k"),
    knot_positions=True,
    constant_constraints=True,
    x=(x4,), 
    y=(y4,),
    col_pt=("b",),
    alpha=0.25
)
```

```
np.random.seed(5)
x7_0 = np.linspace(0, 3 * np.pi, 301)
x7_1 = np.linspace(0, 2 * np.pi, 201)
y7 = np.outer(np.sin(x7_0), np.sin(x7_1)) + np.random.normal(0, 1, (301, 201))
example7 = GridCPsplines(
    deg=(3, 3),
    ord_d=(2, 2),
    n_int=(30, 20),
    sp_args={"verbose": True, "options": {"ftol": 1e-12}},
    int_constraints={0: {0: {"+": 0}}, 1: {0: {"+": 0}}}
)
example7.fit(x=(x7_0, x7_1), y=y7)

plot7 = plot_surfaces(
    fittings=(example7,),
    col_surface=("gist_earth",),
    orientation=(45, 45),
    figsize=(10, 6),
)
```

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