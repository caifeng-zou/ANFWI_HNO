# ANFWI_HNO: Ambient Noise Full Waveform Inversion with Helmholtz Neural Operator
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15061657.svg)](https://doi.org/10.5281/zenodo.15061657)

## Introduction
This repository provides code for the paper [Ambient Noise Full Waveform Inversion with Neural Operators](https://doi.org/10.1029/2025JB031624).

## File Description
- **code**: 
    - **util**: Classes and functions involved
    - FWI_real_SB1.ipynb: Workflow for full waveform inversion with [real ambient noise data](https://drive.google.com/drive/folders/1Qw9t7w6iu753IJSoH95ZUVt8wjHD9_eA?usp=sharing)
    - FWI_synthetic.ipynb: Workflow for full waveform inversion with synthetic data
- **data**: For direct use with the code, put [**CCSB1ZZ.npy**](https://drive.google.com/drive/folders/1Qw9t7w6iu753IJSoH95ZUVt8wjHD9_eA?usp=sharing) and [**CCSB1ZR.npy**](https://drive.google.com/drive/folders/1Qw9t7w6iu753IJSoH95ZUVt8wjHD9_eA?usp=sharing) under this folder.
- **data_generation**: Code for generating training data with [Salvus](https://mondaic.com/docs/2024.1.2/getting_started)
- **model**: Velocity models and normalizers for data processing. For direct use with the code, put the trained [**HNO.pth**](https://drive.google.com/drive/folders/1Qw9t7w6iu753IJSoH95ZUVt8wjHD9_eA?usp=sharing) under this folder.

## Dependencies
```
environment.yml
```

## Citations
We welcome any comments or questions regarding this work. To cite this work:
```
Zou, C., Ross, Z. E., Clayton, R. W., Lin, F.-C., & Azizzadenesheli, K. (2025). Ambient noise full waveform inversion with neural operators. Journal of Geophysical Research: Solid Earth, 130, e2025JB031624. https://doi.org/10.1029/2025JB031624
```
For more method details, we refer to a closely relevant paper [Deep Neural Helmholtz Operators for 3D Elastic Wave Propagation and Inversion](https://academic.oup.com/gji/article/239/3/1469/7760394) and the [repository](https://github.com/caifeng-zou/HelmholtzNO).


Caifeng Zou\
czou@caltech.edu

