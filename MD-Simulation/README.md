# MD-Simulation

This folder contains the OpenMM/Open3SPN2 simulation driver in [run.py](run.py).

## Installation

Create and activate a conda environment, then install the required packages:

```bash
conda create -n md-simulation python=3.10 -y
conda activate md-simulation
conda install -c conda-forge open3spn2
conda install -c conda-forge openmm matplotlib pandas -y
```

Optional: install the OpenMM-PLUMED plugin if you want to enable the `PLUMED` restraints used by `run.py`.

```bash
pip install openmm-plumed
```

## Input Files

`run.py` expects an `INPUT/` folder next to the script. It scans that folder for `.pdb` files and processes each one.

For each input PDB file, the script also expects a matching dot-bracket text file with the same stem:

- `INPUT/example.pdb`
- `INPUT/example.txt`

### `INPUT/*.pdb`

The input structure should be a coarse-grained DNA PDB file compatible with `open3SPN2.DNA.fromCoarsePDB(...)`.

The script looks for residues/atoms named like the 3SPN2 coarse-grained DNA representation, including:

- nucleotide/base atoms: `A`, `C`, `G`, `T`
- backbone atoms used in this script: `P`, `S`

### `INPUT/*.txt`

The matching text file contains one dot-bracket character per nucleotide, read as a single line.

Supported characters used by the current script:

- `+` for residues that should be included in the special PLUMED mesh restraints
- `(` and `)` for restrained bases in the backbone restraint logic
- any other character is treated as unrestrained by those helpers

Keep the `.txt` file length aligned with the number of nucleotide residues in the corresponding PDB.

## Output Files

`run.py` writes results into an `OUTPUT/` folder next to the script. The file names are based on the input stem and simulation parameters.

Typical outputs include:

- `*_sim_k*_tol*.pdb` - trajectory saved as multi-model PDB
- `*_sim_k*_tol*.csv` - energy and temperature report
- `*_sim_k*_tol*.chk` - OpenMM checkpoint for resuming
- `*_sim_k*_tol*_progress.txt` - step counter used for resume logic
- `*_sim_k*_tol*.xml` - serialized system definition
- `*_sim_k*_tol*.plumed.dat` - generated PLUMED input, when enabled
- `*_sim_k*_tol*.mesh.txt` - summary of PLUMED mesh restraints, when enabled
- `*_sim_k*_tol*_lowest5.pdb` - the five lowest-energy frames extracted from the trajectory
- `*_sim_k*_tol*_energy_plot.png` - potential energy and temperature plot
- `console.log` - combined terminal log

## References

### Open3SPN2

Lu, W., Bueno, C., Schafer, N. P., Moller, J., Jin, S., Chen, X., ... & Wolynes, P. G. (2021). OpenAWSEM with Open3SPN2: A fast, flexible, and accessible framework for large-scale coarse-grained biomolecular simulations. *PLoS Computational Biology*, 17(2), e1008308. [https://doi.org/10.1371/journal.pcbi.1008308](https://doi.org/10.1371/journal.pcbi.1008308)

### 3SPN.2C

Freeman, G. S., Hinckley, D. M., Lequieu, J. P., Whitmer, J. K., & De Pablo, J. J. (2014). Coarse-grained modeling of DNA curvature. *Journal of Chemical Physics*, 141(16). [https://doi.org/10.1063/1.4897649](https://doi.org/10.1063/1.4897649)

### 3SPN.2

Hinckley, D. M., Freeman, G. S., Whitmer, J. K., & De Pablo, J. J. (2013). An experimentally-informed coarse-grained 3-site-per-nucleotide model of DNA: Structure, thermodynamics, and dynamics of hybridization. *Journal of Chemical Physics*, 139(14). [https://doi.org/10.1063/1.4822042](https://doi.org/10.1063/1.4822042)
