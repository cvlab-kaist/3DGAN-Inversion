# 3D GAN Inversion with Pose Optimization
<a href="https://arxiv.org/abs/2210.07301"><img src="https://img.shields.io/badge/arXiv-2210.07301-b31b1b.svg"></a>

<!-- Inference Notebook: <a href="https://colab.research.google.com/drive/1HY8g_HR26YHsYmzrjC6K3gIaIK09bWD7?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" height=20></a>   -->
 <!-- ## [[Project Page]](https://3dgan-inversion.github.io./) -->
### Official PyTorch implementation of the WACV 2023 paper

![1](https://user-images.githubusercontent.com/78152231/204740257-1996faa1-11ff-4710-8224-1cf340be7d29.png)
![2](https://user-images.githubusercontent.com/78152231/204739677-2580175e-37ee-403e-8159-8a37b71f0207.png)
![3](https://user-images.githubusercontent.com/78152231/204739594-110e6928-3ebe-4663-800d-4b37dbfdae88.png)
![0](https://user-images.githubusercontent.com/78152231/204739664-4df84a8a-28b5-4a36-8705-93e057e576c4.png)


#### Jaehoon Ko*, Kyusun Cho*, Daewon Choi, Kwangrok Ryoo, Seungryong Kim,

  **equal contribution*
 
For more information, check out the paper on [Arxiv](https://arxiv.org/abs/2210.07301) or [Project page](https://3dgan-inversion.github.io/)




# Requirements
NVIDIA GPUs. We have done all testings on RTX 3090 GPU.

64-bit Python 3.9, PyTorch 1.11.0 + CUDA toolkit 11.3

```
conda env create -f environment.yml
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda activate 3dganinv
```

# Pre-trained Networks
Download pre-trained weights on this google drive [Links](https://drive.google.com/drive/folders/1t7uD8ng-r2-3xaTpfY12Y7_ah-gLHz0c?usp=sharing)

Put weight of initializers and generators as followings:


    └── root

        └── initializer
    
            └── pose_estimator.pt
        
            └── pose_estimator_quat.pt
        
            └── pose_estimator_afhq.pt
        
            └── e4e_ffhq.pt
        
            └── e4e_afhq.pt
        
        └── pretrained_models
    
            └── afhqcats512-128.pkl
        
            └── ffhqrebalanced512-128.pkl
        
# Image Preprocessing
Follow the preprocessing code from EG3D repository. 

(Link: https://github.com/NVlabs/eg3d/blob/main/dataset_preprocessing/ffhq/crop_images_in_the_wild.py)

For cat face datasets, we just manually cropped the facial areas.

# Inversion
Run inversion process
```
python scripts/run_pti.py
```

You can edit the input & output directories, or GPU number on configs/paths_config.py


# Credits

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




# Acknowledgement
This code implementation is heavily borrowed from the official implementation of [EG3D](https://github.com/NVlabs/eg3d) and [PTI](https://github.com/danielroich/PTI). We really appreciate for all the projects.

### Bibtex
```
@article{ko20233d,
  author    = {Ko, Jaehoon and Cho, Kyusun and Choi, Daewon and Ryoo, Kwangrok and Kim, Seungryong},
  title     = {3D GAN Inversion with Pose Optimization},
  journal   = {WACV},
  year      = {2023},
}
```
