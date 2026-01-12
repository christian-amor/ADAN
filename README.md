# Autoregressive Dynamics-Aware Network (ADAN)

## Overview

This repository contains a demo of the Autoregressive Dynamics-Aware Network (ADAN). ADAN is a hybrid model that combines modal decompositions, specifically the dynamic mode decomposition (DMD, Schmid, 2010), with a neural network trained to predict flow quantities in physical space. The neural network is based on a sequence-to-sequence model (Sutskever et al., 2014), and it implements attention mechanisms following Bahdanau et al., 2014 and Luong et al., 2015 formulations, respectively.

## Requirements

- Python
- NumPy
- pandas
- SciPy
- h5py
- Matplotlib
- seaborn
- PyTorch
- path
- tqdm

## Example dataset

This demo showcases the prediction of the turbulent kinetic energy in the flow behind a cylinder at Reynolds number Re = 280. The `data/` repository contains the files `DeltasOmegasAmpl_Re280.mat` and `TKE_Re280.txt`. Trained models are saved in the `model/` directory.

Two pretrained models are included in the `model/` directory. The parameters of the models are already set in the `config` class (for using the model with Bahdanau attention, change `attn_model` to `'BAH'`).
