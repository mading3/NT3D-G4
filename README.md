
# NT3D-G4s:  An integrated deep learning and simulation-enabled pipeline to predict key structural features of DNA G-quadruplexes

This repository contains the fundamental scripts used in the development of the **NT3D-G4s** pipeline, a combined deep learning and coarse-grained molecular dynamics pipeline for identifying detailed structural characteristics of DNA G-quadruplex (G4) oligomers. **NT3D-G4s** utilizes a custom encoder-only Transformer neural network for initial generation of a DNA G4 topology in coarse-grain format, followed by a OpenMM/Open3SPN2-based molecular dynamics simulation pipeline with custom restraints to fine-tune generated predictions for effective structural analysis. 

## Citation

If you utilize this pipeline in your research, please cite:

TBA

**DOI:** TBA

## Repository Features

- **Custom Dataloader for 3SPN-type DNA structure files**
- **Encoder-only Transformer with training, model loading, and prediction scripts**
- **Scripts for generating weight matrices for DNA oligonucleotides with Watson-Crick and G-tetrad Hoogsteen base pairing for the weighted MSELoss function**
- **Script for OpenMM/Open3SPN.2-based CG-MD simulation**

## Content

- `machine-learning`: Directory containing machine learning pipeline documents
- `MD-simulation`: Directory containing CG-MD simulation documents
- `NT3D-G4s.ipynb`: Jupyter notebook housing model architecture, data loading and weight matrix use cases, training and prediction pipelines
- `G4_encoder.py`: Generation of weight matrices for G4 structures, requires an input of secondary structure (ViennaRNA dot bracket string) and a list of loop orders, factors all forms of "canonical" 4-tract G4 structures and several examples of non-canonical topologies
- `vRNA_encoder.py`: Generation of weight matrices for DNA structures with Watson-Crick base pairing, predicts secondary structure directly using pre-built computational algorithm (requires only sequence input)
- `run.py`: Main CG-MD simulation pipeline to perform energy minimization + production simulation on machine learning outputs
- `open3spn2_cg_converter_tool.py`: Script to convert coarse-grained PDB outputs to an Open3SPN.2-compatible format
- `all_dataset_g4_structures.xlsx`: Excel spreadsheet containing every DNA G4 structure included in the training/validation step with annotated information regarding its global topology, loop order, and secondary structure
- `README.md`: Project documentation.

## Requirements for Machine Learning portion of the NT3D-G4s pipeline

- Python v3.10
- Pytorch v2.5.1
- Numpy v2.2.6
- Pandas v2.2.3
- Matplotlib v3.10.0
- ViennaRNA v2.7.0

## Requirements for CG-MD portion of the NT3D-G4s pipeline

- TBA
