# Pressure-matching IBI
tools for performing pressure-matching iterative Boltzmann inversion (IBI) for coarse-grained molecular dynamics simulation

## Introduction

Bottom-up coarse-grained (CG) methods enable efficient simulations of complex molecular systems at mesoscopic scales. Among them, iterative Boltzmann inversion (IBI) is one of the simplest yet widely used approaches, optimizing pairwise potentials to reproduce radial distribution functions (RDF). While IBI effectively captures structural correlations, it often suffers from thermodynamic inconsistencies, particularly in reproducing the correct pressure of the all-atom reference system.

In this repository, we present simple yet effective extensions to IBI—iLS (iterative length-scale rescaling) and iLC (iterative linear correction)—which incorporate pressure corrections directly into the IBI framework.  This repository contains the core code for implementing and testing these methods.

## Installation and Dependencies

Clone the repository and set up the required environment:
```
git clone https://github.com/zyumse/IBI_PM.git
cd gECG_thiophene/

# create virtual environment 
conda create --name your_env_name python=3.11
conda activate your_env_name

# Install required Python packages
pip install -r requirements.txt
pip install -e .
```

## Usage
### CG mapping of all-atom simulation
1. Perform MD simulation at the all-atom level

    Here, we employe [Lammps](https://github.com/lammps/lammps) as the MD engine, but others work too 

2. Perform CG mapping

3. Extract structure properties and density

### Pressure-matching IBI

1. IBI-iLS

2. IBI-iLC

3. other variants of IBI

## Example (test)

