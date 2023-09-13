# Code for generative modeling and Bayesian analysis of data for interferometric scattering microscopy

This repository contains Python scripts and Jupyter notebooks for analyzing data from interferometric scattering microscopy.  Xander de Wit is the primary author of the code.  Please see the following publication for additional details and cite it if you use the code in your work:

Xander M. de Wit, Amelia W. Paine, Caroline Martin, Aaron M. Goldfain, Rees F. Garmann, and Vinothan N. Manoharan, "Precise characterization of nanometer-scale systems using interferometric scattering microscopy and Bayesian analysis," _Applied Optics_ 62, 7205-7215 (2023) 

Please consult the main notebook `main.ipynb` for details and instructions.

All data required for the main notebook as well the auxiliary .py scripts are available for download in the [Harvard Dataverse repository](https://doi.org/10.7910/DVN/N7GJYC). Please download all files and place them in the `data/` directory.

The `environment.yml` file can be used to set up a python environment to run the analysis code.  To install this environment with the `conda` package manager, use
```
conda env create -f environment.yml
```
