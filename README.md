# gmxtop 

<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

GROMACS topology files for python

## Description
The `gmxtop` project provides a python interface to
* read and write toplogy and force field information from GROMACS-type top-files
* alter force field parameters 

The `gmxtop` project derives this functionality from the [kimmdy](https://github.com/graeter-group/kimmdy) project, originally developed by the graeter-group, and includes only minor modifications to parts of the original code to operate independently.
The `kimmdy` project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) and, as a derivative work, the `gmxtop` project is distributed under the same license. See [LICENSE](./LICENSE) for details.

### Limitations
* CMAP corrections (`[ cmaptypes ]`, `[ cmap ]`) are only passed through: they are read and written back unchanged, but not parsed into objects. Hence, CMAP terms are not generated for new residues and their atom numbers are not updated when atoms are renumbered.
* CHARMM force field files use `#ifdef`/`#ifndef` statements within sections (e.g. `#ifdef HEAVY_H` in `[ atomtypes ]`). This is not supported yet. Thus, CHARMM topologies can not be read directly. As a workaround, preprocess the topology with `grompp` first and use the resulting standalone topology:
```bash
gmx grompp -f empty.mdp -c conf.gro -p topol.top -pp topol_pp.top
```

## Tutorials
The following tutorials are avaiable as Google Colab notebooks and hence do not require a local installation:
* [Basic tutorial](https://colab.research.google.com/drive/19zjIw2H5O6InLDQWLXEvrwrZ5w0AL7i1?usp=sharing) - Basic accessing of topology information with gmxtop
* [Advanced tutorial](https://colab.research.google.com/drive/1uWgji2O1N4PnxnYpGc1yV68Qzm-PfVX1?usp=sharing) - Advanded accessing of topology information with gmxtop
* [Force field parameter tutorial](https://colab.research.google.com/drive/1YOPPkCDudqSwd2QV0P7qEcd-1bogrvtY?usp=sharing) - Adapting force field parameters with gmxtop

## Installation
### From PyPI
```bash
pip install gmxtop
```

### From source via uv
Clone repository and move into
```bash
git clone git@github.com:graeter-group/gmxtop.git
cd gmxtop
```
Install repository
```bash
uv sync
```
Activate virtual environment
```bash
source .venv/bin/activate
```
Verify install by running the tests
```bash
pytest tests
```

### From source via conda and pip
Clone repository and move into
```bash
git clone git@github.com:graeter-group/gmxtop.git
cd gmxtop
```
Install repository into conda environment
```bash
conda create -n gmxtop python -y
conda activate gmxtop
pip install -e '.[dev]' # install with dev dependencies
```
Verify install by running the tests
```bash
pytest tests
```