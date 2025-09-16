[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14525443.svg)](https://doi.org/10.5281/zenodo.14525443)

# Telomeres

## Overview

Code associated with "Mathematical model linking telomeres to senescence
in _Saccharomyces cerevisiae_ reveals cell lineage versus population
dynamics".

- Preprint version: [biorxiv.org](https://doi.org/10.1101/2023.11.22.568287)
- See also Chapter 3 of the PhD thesis: [HAL theses](https://theses.hal.science/tel-04250492)

The `telomeres` package contains all the necessary auxiliary code.
This is where the mathematical model is encoded, with its
default parameters (`parameters.py`). More generally, it contains all
the functions allowing to

- Posttreat the raw data (`make_*.py`)
- Simulate the model (`simulation.py`)
- Plot the simulated and experimental data, the laws of the model... (`plot.py`)

The scripts in this folder are not intended to be modified (unless
you find errors, in which case please let me know) or used directly to
run simulations.

The `makeFiles` folder contains scripts to run to generate the
`data/processed` directory, that contains the posstreated data.

The `main` folder contains the scripts that should be run to perform
the simulations and plot their results.

## Contents

1. [Software Requirements](#software-requirements)
2. [Dependencies](#dependencies)
   - [Python Versions and Packages](#python-versions-and-packages)
   - [Additional Requirements for Ubuntu Users](#additional-requirements-for-ubuntu-users)
3. [Dataset](#dataset)
4. [Code](#code)
   - [Result Reproducibility](#result-reproducibility)

## Software Requirements

The code has been tested on the following systems:

- **Linux**: Ubuntu 22.04.5 LTS, 24.04 LTS
- **Mac**: MacOS Big Sur 11.6.2
- **Windows**:

## Dependencies

### Python Versions and Packages

The code has been developed in _Python 3.8_ and maintained with _Python 3.11.7_.

Errors might occur on a machine or cluster that is missing some Python packages, like `cma`, `latex` and `mpl-axes-aligner`. They can be installed via Python console with:

```bash
pip install cma
pip install latex
pip install mpl-axes-aligner
```

### Additional Requirements for Ubuntu Users

Ubuntu users might need to install texlive packages via terminal:

```bash
sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super texlive-fonts-extra
```

or

```bash
sudo apt install texlive-full
```

## Dataset

Available in the `data` folder and gathered the _Source Data.xlsx_ file provided with the article.

## Code

### Result Reproducibility

Reproducing our results is costly in terms of time and memory.
We therefore recommend simulating with "small" parameters first.
For example, compute averages on $k = 2$ simulations:

- In _main/lineage/compute.py_ taking `SIMU_COUNT = 2` instead of `1000`
- In _main/population/compute.py_ taking `SIMU_COUNT = 3` instead of `30`, and start with $N_{init} = 5$ cells rather than $300$ or $1000$ by setting `C_EXP = np.array([5])`

For "larger" parameters, parallel computing on a cluster is strongly recommended.
We used the [CLEPS cluster](https://paris-cluster-2019.gitlabpages.inria.fr/cleps/cleps-userguide/index.html) from Inria Paris. Our Slurm submission scripts are the `.batch` files contained in the _main_ directory.

Please contact me directly if you need some of our raw simulated data (250 Go total, including less than 5 Go for lineage simulations).
