# 3D GAN Inversion with Pose Optimization     (WACV 2023)
Jaehoon Ko*, Kyusun Cho*, Daewon Choi, Kwangrok Ryoo, Seungryong Kim

<a href="https://arxiv.org/abs/2210.07301"><img src="https://img.shields.io/badge/arXiv-2210.07301-b31b1b.svg"></a>

Inference Notebook: <a href="https://colab.research.google.com/drive/1HY8g_HR26YHsYmzrjC6K3gIaIK09bWD7?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>  
 ## [[Project Page]](https://3dgan-inversion.github.io./)

![overview](docs/chris_hemsworth_inv.gif)

## Description   
Official Implementation of our WACV 2023 paper "3D GAN Inversion with Pose Optimization" by Jaehoon Ko, Kyusun Cho, Daewon Choi, Kwangrok Ryoo and Seungryong Kim.

This is an early release of our implementation, and is currently unfinished with several bugs and disordered organization. We shall update this repository with finalalized codes and trained models in a few weeks time. 


## Getting Started
### Prerequisites
We utilize the Python conda environment used in [EG3D](https://github.com/NVlabs/eg3d).

```.bash
conda env create -f environment.yml
conda activate eg3d
```


## Pretrained Models
Please execute the download script file for pre-trained networks for EG3D and our pre-trained networks. 
```.bash
sh download_networks.sh
```


### Training Initialization Networks
Our training codes for the camera viewpoint estimator and latent representation encoder are coming soon.
Meanwhile, you can download and use pre-trained networks.


### Running Inversion
The main training script is `inversion.py`. The script receives aligned and cropped images from paths configured in the "Input info" subscetion in`configs/paths_config.py`. 
Results are saved to directories found at "Dirs for output files" under `configs/paths_config.py`. This includes inversion latent codes and tuned generators. 
The hyperparametrs for the inversion task can be found at  `configs/hyperparameters.py`. They are intilized to the default values used in the paper. 

## Editing
We mainly use [GANSPACE](https://github.com/harskish/ganspace) for the latent-based manipulation of 3D-aware inverted images. The code and latent directions for editing are coming soon. 

## Inference Notebooks
To help visualize the results of our method we provide a Colab notebook linked above.   




## Credits

**StyleCLIP model and implementation:**   
https://github.com/NVlabs/eg3d
Copyright (c) 2021-2022, NVIDIA Corporation & affiliates. 
License (NVIDIA) https://github.com/NVlabs/eg3d/blob/main/LICENSE.txt

**PTI implementation:**   
https://github.com/danielroich/PTI
Copyright (c) 2021 Daniel Roich  
License (MIT) https://github.com/danielroich/PTI/blob/main/LICENSE

**GANSPACE implementation:**   
https://github.com/harskish/ganspace
Copyright (c) 2020 harkish  
License (Apache License 2.0) https://github.com/harskish/ganspace/blob/master/LICENSE


## Acknowledgments
This repository structure is heavily based on [EG3D](https://github.com/NVlabs/eg3d) and [PTI](https://github.com/danielroich/PTI) repositories

## Contact
For any inquiry please contact us at our email addresses: kjh9604@korea.ac.kr or kyustorm7@korea.ac.kr


## Citation
If you use this code for your research, please cite:
```
@article{ko20233d,
  author    = {Ko, Jaehoon and Cho, Kyusun and Choi, Daewon and Ryoo, Kwangrok and Kim, Seungryong},
  title     = {3D GAN Inversion with Pose Optimization},
  journal   = {WACV},
  year      = {2023},
}
```
