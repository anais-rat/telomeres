# Telomeres

## Overview

Code for "Individual cell fate and population dynamics revealed by a mathematical model linking telomere length and replicative senescence". 

- Preprint version: [biorxiv.org](https://doi.org/10.1101/2023.11.22.568287)
- See also Chapter 3 of the PhD thesis: [HAL theses](https://theses.hal.science/tel-04250492)

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

- **Linux**: Ubuntu 24.04 LTS
- **Mac**:
- **Windows**:

## Dependencies

### Python Versions and Packages

The code has been developed in *Python 3.8* and maintained with *Python 3.11.7*.

Errors might occur on a machine or cluster that is missing some Python packages, like `latex` and `mpl-axes-aligner`. They can be installed via Python console with:
```bash
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

Please contact us.

## Code 

### Result Reproducibility 

Reproducing our results is costly in time and memory. 
We therefore recommend simulating with "small" parameters first (e.g., average on $k=2$ simulations instead of $30$, for initially $N_{init} = 5$ cells rather than $300$ or $1000$).

For "larger" parameters, parallel computing is strongly recommended. 
We used the [CLEPS cluster](https://paris-cluster-2019.gitlabpages.inria.fr/cleps/cleps-userguide/index.html) from Inria Paris. 
Our *Slurm* submission scripts are the `.batch` files.

Please contact me directly if you need some of our raw simulated data (250 Go total).

