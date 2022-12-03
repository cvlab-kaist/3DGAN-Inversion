import torch
import numpy as np
import PIL
import os
from torchvision.utils import make_grid 
from torchvision.transforms import ToPILImage
from configs import global_config


# idx_comp: determines which index of pca_component direction for editing direction(0~511, 0: first pca_component) 
# start_layer: first layer's index applying a editing direction. (0: first layer)
# layer_num: from start_layer to how many layers applying a editing direction.
# edit_power: -edit_power~edit_power
# num_imgs: # of editing imgs
# save_dir: path for saving editing imgs
# file_name: file name for saving editing imgs
# save_inter_images: determines whether imgs are stored individually. (True/False)
# save grid_images: determines whether grid img is stored. (True/False)

# W: [1, 14, 512](for EG3D). freeze_cam: [1, 512](this is input camera parameter)
def edit_ganspace(generator, pca_comp, w, freeze_cam, idx_comp, start_layer = 0, layer_num = 12, edit_power = 1, num_imgs = 5, save_dir=None, file_name=None, save_inter_images=False, save_grid_images=True): 
    print("editing start")
    if start_layer + layer_num > 14:
        print("layer_num exceed!")
        return

    V = torch.tensor(pca_comp).transpose(0, 1) #[512, K]
    K = V.shape[1]
    direction_list = []

    for i in range(1, num_imgs + 1):
        control_params = torch.zeros(K) # control parameter, [K]
        control_params[idx_comp] = -edit_power + ((2 * edit_power)/(num_imgs -1)) * (i-1) 
        direction = torch.matmul(V, control_params).reshape(1, -1).unsqueeze(0).expand(-1, layer_num, -1)  #[1, layer_num, 512]
        direction_matrix = torch.zeros(1, 14, 512) #[1, 14, 512]
        direction_matrix[0, start_layer:start_layer+layer_num, :] = direction
        direction_matrix = direction_matrix.to(global_config.device) 
        direction_list.append(direction_matrix) #  save edit direction 

        final_w = (w + direction_matrix).to(global_config.device) # [1, 14, 512], you can simply add direction matrix to input w vector for editing.
        syn_img = generator.synthesis(final_w, freeze_cam)['image'] #[1, 3, 512, 512]
        if save_inter_images:
            make_dir(save_dir)
            intimg = (syn_img[0].permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8) #[512, 512, 3]
            PIL.Image.fromarray(intimg.cpu().numpy(), 'RGB').save(os.path.join(save_dir, file_name + '_inter_' +str(i)+'.png'))
        if save_grid_images:
            syn_img = (syn_img * 127.5 + 128).clamp(0, 255).to(torch.uint8) #[512, 512, 3]
            if i == 1:
                    inter_imgs = syn_img
            else:
                    inter_imgs = torch.cat([inter_imgs, syn_img], dim=0)
    if save_grid_images:
        make_dir(save_dir)
        result_image = ToPILImage()(make_grid(inter_imgs.detach().cpu()))
        result_image.save(os.path.join(save_dir, file_name + '_grid' +'.png'))
    print("editing end")
    return direction_list

def make_dir(save_dir):
    if os.path.isdir(save_dir) == 0:
        os.mkdir(save_dir)


   
if __name__=="__main__":
    pca_comp = np.load('ganspace/pca_comp/pca_ffhqrebalanced_10_5_frontcam.npy') 
    V = torch.tensor(pca_comp).transpose(0, 1) #[512, K]
    
    # idx_comp, start_layer, layer_num, edit_power 
    # Below values are ONLY sample parameters. So, If this parameter don't be operated, you need to find your own best parameters for editing.  
    ganspace_directions = {
            'bright hair': (2, 7, 7, 4), #positive (direction)
            'smile': (12, 0, 5, 2), #positive 
            'age' : (5, 0, 5, 3.5), #negative: young
            'short hair': (2, 0, 5, 4), #negative
            'glass': (4, 0, 5, 4), #negative
            'gender': (0, 0, 5, 4) #negative(female -> male)
    }

 
    

