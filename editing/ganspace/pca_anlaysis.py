from utils.models_utils import toogle_grad, load_old_G
import torch
import numpy as np
from configs import global_config
from editings.ganspace.estimator import PCAEstimator

if __name__=="__main__":

    generator = load_old_G()
    toogle_grad(generator, True)
    n_samples = 10**5 # number of samples for PCA 
    z = torch.randn(n_samples, 512).to(global_config.device)
    front_cam = torch.tensor([0.9966070652008057,
                 0.003541737562045455,
                -0.08222994953393936,
                 0.20670529656089412,
                -0.009605886414647102,
                -0.9872410893440247,
                -0.15894262492656708,
                0.4137044218920643,
                -0.08174371719360352,
                0.1591932326555252,
                -0.9838574528694153,
               2.660098037982929,  0.0000,  0.0000,  0.0000,  1.0000, 
             4.2647, 0.0,0.5,0.0,4.2647,0.5,0.0,0.0,1.0]).repeat(n_samples, 1).to(global_config.device) #this is canonical direction's camera parameter. this is must be fixed!

    w = generator.mapping(z, front_cam) #[n_samples, 14, 512]
    w = w[:, 0, :] 
    K = 512 # number of components. 512 is max.
    pca = PCAEstimator(n_components=K) 
    pca.fit(w.cpu().detach().numpy())
    pca_comp, _, _ = pca.get_components() 
    np.save('editings/ganspace/pca_comp/pca_ffhqrebalanced_10_5_frontcam.npy', pca_comp)

    