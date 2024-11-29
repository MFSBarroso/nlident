# nlident
nlindent Package for Nonlinear Systems Identification
# nlindent Package for Nonlinear Systems Identification

## Overview

The `nlindent` package is a powerful tool for identifying nonlinear systems. It provides a comprehensive set of functions for generating candidate terms, building regressor matrices, estimating parameters, and analyzing model performance. This package is designed to facilitate the development and validation of nonlinear models, making it an invaluable resource for researchers and engineers working in the field of system identification.

## Installation

You can install the `nlindent` package directly from GitHub. First, clone the repository, then navigate to the directory and install the package using pip:

```bash
git clone https://github.com/MFSBarroso/nlident/nlindent.git
cd nlindent
pip install .

Features
The nlindent package includes the following functions:

genterms: Generate candidate terms for the model

build_pr: Build the process regressor matrix

build_no: Build the noise regressor matrix

get_info: Extract information from the model

mcand: Generate model candidates

sort_pr: Sort the process regressors

sort_no: Sort the noise regressors

simodeld: Simulate the model's output

ols: Ordinary Least Squares estimation

els: Extended Least Squares estimation

coefcorr: Calculate the correlation coefficient

rmse: Calculate the Root Mean Squared Error

aic: Calculate the Akaike Information Criterion

likelihood: Calculate the likelihood of the model

akaike: Calculate the Akaike Information Criterion for model comparison

funcorr: Calculate the autocorrelation function

plot_funcorr: Plot the autocorrelation function

funcorrcruz: Calculate the cross-correlation function

plot_funcorrcruz: Plot the cross-correlation function

narx: Identify a NARX model

narmax: Identify a NARMAX model



