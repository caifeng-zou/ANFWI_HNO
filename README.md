# ANFWI_HNO: Ambient Noise Full Waveform Inversion with Helmholtz Neural Operator

## Introduction
This repository provides code for the paper [Ambient Noise Full Waveform Inversion with Neural Operators](https://doi.org/10.48550/arXiv.2503.15013).

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
@misc{zou2025ambientnoisewaveforminversion,
      title={Ambient Noise Full Waveform Inversion with Neural Operators}, 
      author={Caifeng Zou and Zachary E. Ross and Robert W. Clayton and Fan-Chi Lin and Kamyar Azizzadenesheli},
      year={2025},
      eprint={2503.15013},
      archivePrefix={arXiv},
      primaryClass={physics.geo-ph},
      url={https://arxiv.org/abs/2503.15013}, 
}
```
For more method details, we refer to a closely relevant paper: [Deep Neural Helmholtz Operators for 3D Elastic Wave Propagation and Inversion](https://academic.oup.com/gji/article/239/3/1469/7760394)


Caifeng Zou\
czou@caltech.edu

