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

MOSEK requires a license to be used. For research or educational purposes, an free yearly and renewable [academic license](https://www.mosek.com/products/academic-licenses/) is offered by the company. For other cases, a 30-day [trial license](https://www.mosek.com/try/) is available. According to MOSEK indications, the license file (`mosek.lic`) must be located at 
```
$HOME/mosek/mosek.lic                (Linux/OSX)
%USERPROFILE%\mosek\mosek.lic        (Windows)
```

