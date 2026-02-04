[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14525443.svg)](https://doi.org/10.5281/zenodo.14525443) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18454274.svg)](https://doi.org/10.5281/zenodo.18454274)

# Telomeres

## Overview

Code associated with

<a id="ref-1"></a>

- [1]: "_Both Genome Instability and Replicative Senescence Stem from the Shortest Telomere in Telomerase-Negative Cells_": [BioRxiv↗](https://www.biorxiv.org/content/10.1101/2025.01.27.635053v2)
  <a id="ref-2"></a>
- [2]: "_Mathematical model linking telomeres to senescence
  in *Saccharomyces cerevisiae* reveals cell lineage versus population
  dynamics_": [Nat Commun 16, 1024 (2025)↗](https://www.nature.com/articles/s41467-025-56196-z), [BioRxiv↗](https://doi.org/10.1101/2023.11.22.568287)
  <a id="ref-2"></a>
- [3]: Chapter 3 of the PhD thesis: [HAL theses↗](https://theses.hal.science/tel-04250492)

The `telomeres` package contains all the necessary auxiliary code.
This is where the mathematical model is encoded, with its
default parameters (`parameters.py`). More generally, it contains all
the functions allowing to

- Post-treat the raw data (`make_*.py`)
- Simulate the model (`simulation.py`)
- Plot the simulated and experimental data, the laws of the model... (`plot.py`)

The scripts in this folder are not intended to be modified (unless
you find errors, in which case please let me know) or used directly to
run simulations.

The [makeFile](./makeFile/) folder contains scripts to run to generate the
[data/processed](./data/processed/) directory, that contains the post-treated data.
The [main](./main/) folder contains the scripts that should be run to perform
the simulations and plot their results.

## Contents

1. [Setup](#setup)
   - [Requirements](#requirements)
   - [Download](#download)
   - [Installation](#installation)
   - [Development and Contributions](#development-and-contributions)
2. [Dataset](#dataset)
3. [Reproducibility](#reproducibility)
   - [Environment](#environment)
   - [Simulated Data](#simulated-data)
4. [Acknowledgements](#acknowledgements)

## Setup

### Requirements

The code has been tested on the following systems:

- **Linux**: Ubuntu 22.04.5 LTS, 24.04 LTS
- **Mac**: MacOS Big Sur 11.6.2
- **Windows**: not tested; users may consider running the code via **WSL2 (Linux)**

The project requires several Python packages, all listed in the `[project.dependencies]` field of the [pyproject.toml🡕](./pyproject.toml) file.
They are automatically installed with the [telomeres🡕](./telomeres/) package when following the installation instructions below.

### Download

Get the latest development version from **GitHub**:

```bash
git clone https://github.com/anais-rat/telomeres.git
cd telomeres
```

Or download a archived version either from Zenodo or the corresponding tagged GitHub release:
| Version | Zenodo DOI | GitHub Release | Related work |
| ---------| -------------------------------------------------------------------------------------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------------| -------------------|
| v1 | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14525443.svg)](https://doi.org/10.5281/zenodo.14525443) | [![GitHub Release](https://img.shields.io/badge/GitHub-v1_release-blue)](https://github.com/anais-rat/telomeres/releases/tag/v1-publication) | [\[2,3\]](#ref-2) |
| v2 | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18454274.svg)](https://doi.org/10.5281/zenodo.18454274) | [![GitHub Release](https://img.shields.io/badge/GitHub-v2_release-blue)](https://github.com/anais-rat/telomeres/releases/tag/v2-publication) | [\[1\]](#ref-1) |

> _Note._ We recommend using the latest version, including for simulations related to references [\[2,3\]](#ref-2), as it provides better performances.

### Installation

#### Pixi installer _(Recommended)_

For an easy and reproducible installation, we recommend using [Pixi↗](https://pixi.sh/latest).

1. Install Pixi following [Pixi's official guide↗](https://pixi.sh/latest/installation/)

2. From the project root, simply run:

   ```bash
   # Create an environment (.pixi/envs/default) and install required dependencies
   pixi install
   ```

   or, to additionally activate the environment:

   ```bash
   pixi shell
   exit  # To deactivate the environment
   ```

   This creates a dedicated environment named `default`, with all project dependencies installed.<br> Other environments are also available:

   - For **using notebooks or the VSCode Interactive Window**, use the `interactive` environment instead, which also includes `jupyterlab`, `pip` and `ipykernel`:

     ```bash
     # Create the environment (.pixi/envs/interactive)
     pixi install -e interactive  # without activating it
     pixi shell -e interactive  # or activating it
     ```

   - For development setup, use the `dev` environment (see [Development and Contributions](#development-and-contributions)).

   - To combine both previous, use `full`.

#### _Alternative installation_

For users who prefer not to use **Pixi**, installation can be done using the standard **pip** package manager.
We highly recommend creating an isolated environment first, for example:

```bash
# (Recommended) Create and activate a conda environment
conda create -n yourenvname python=3.12
conda activate yourenvname
```

Once the environment is active, install the package and its dependencies:

```bash
pip install .
```

### Development and Contributions

To contribute to this project by submitting a pull request, please enable the development environment during installation

```bash
# (Recommended) Using Pixi
pixi install -e dev

# (Alternative) After environment creation, using pip
pip install -e .[dev]
```

This installs [pre-commit↗](https://pre-commit.com/) alongside the regular setup.
After activating your environment, run once:

```bash
pre-commit install
```

to enable automatic code checks and formatting at every commit, as configured in [.pre-commit-config.yaml](./.pre-commit-config.yaml).

## Dataset

Available in the [data](./data/) folder. In particular, see the [_Source Data Fig\*.xlsx_]() and [_Source Data SFig\*.xlsx_]() files in the [data/processed](./data/processed/) subfolders for mapping the data to the figures and supplementary figures in the article.

## Reproducibility

### Environment

Pixi ensures a fully reproducible environment using the dependencies and versions pinned in the [pixi.lock](./pixi.lock) file.
To exactly reproduce simulations and figures, use the same environment that we used:

```bash
pixi install -e full --frozen
```

> _Note.&nbsp;_ The `--frozen` flag ensures that all package versions are installed exactly as recorded in the [pixi.lock](./pixi.lock), without re-resolving or updating dependencies.
> This lockfile was generated on **Linux 64-bit**. Although it includes resolutions for macOS and Windows, _minor differences in dependency versions or native builds_ may occur on these platforms.

### Simulated Data

Reproducing our results is costly in terms of time and memory.
We therefore recommend simulating with "small" parameters first.
For example, compute averages on $k = 2$ simulations:

- In [main/lineage/compute.py](./main/lineage/compute.py) taking `SIMU_COUNT = 2` instead of `1000`
- In [main/population/compute.py](./main/population/compute.py) taking `SIMU_COUNT = 3` instead of `30`, and start with $N_{init} = 5$ cells rather than $300$ or $1000$ by setting `C_EXP = np.array([5])`

For "larger" parameters, parallel computing on a cluster is strongly recommended.
We used the [CLEPS cluster↗](https://paris-cluster-2019.gitlabpages.inria.fr/cleps/cleps-userguide/index.html) from Inria Paris. Our Slurm submission scripts are the `.batch` files contained in the [main](./main/) directory.

For access to our raw simulated data (250 GB total, including less than 5 GB for lineage simulations), please contact [Anaïs Rat↗](https://github.com/anais-rat) directly.

## Acknowledgements

We are very grateful to [Jules Olayé↗](https://julesolaye.github.io/) and [Virgile Andreani↗](https://www.normalesup.org/~andreani/) for their valuable feedback, suggestions and contributions to improve code performance.

This project was supported by the ERC Starting Grant SKIPPERAD 306321.
