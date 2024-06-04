# Telomeres

## Overview

Codes for "Individual cell fate and population dynamics revealed by a mathematical model linking telomere length and replicative senescence". 

- Preprint version: [biorxiv.org](https://www.biorxiv.org/content/10.1101/2023.11.22.568287v1.full.pdf)
- Chapter 3 of the PhD thesis: [HAL theses](https://theses.hal.science/tel-04250492)

## Software Requirements

The code has been tested on the following systems:

- **Linux**: Ubuntu 24.04 LTS
- **Mac**
- **Windows**

## Dependencies

### Python versions and packages

Developed in Python 3.8 and maintained with Python 3.11.7.

Errors might occur on a machine or cluster that is missing some Python
packages. Among them **latex** and **mpl-axes-aligner**, that can be
installed from the Python console running:

```bash
pip install latex
pip install mpl-axes-aligner
```

### Additional Requirements for Ubuntu Users

Ubuntu users might need to install the following texlive packages:

```bash
sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super texlive-fonts-extra
```

or 

```bash
sudo apt install texlive-full
```
