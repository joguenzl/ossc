This repository contains the code for the supplementary information of the article "Reaction monitoring including unknown chemicals using in situ infrared spectroscopy". It provides an implementation of the algorithms introduced in the article and two interactive notebooks:

- A Python script that produces all of the graphics used in the article and performs all the data analysis
- An interactive Python notebook to test the algorithm on experimental and simulated data


The scripts require a Python package that can be installed locally from this repository. To install it, navigate to the subdirectory `package` in your command line and run `python -m pip install .` or the equivalent of your package manager (preferably in a virtual environment). We use [Marimo](https://marimo.io) for the notebooks, a tool focused on reproducible notebook outputs. It should be automatically installed and added to the path; if the below commands do not work, please check the installation instructions for Marimo itself.

To run the evaluation script, execute `marimo run figures.py` from your command line from the repository's root directory.

To run the interactive notebook, execute `marimo run gui.py` from your command line from the repository's root directory.
