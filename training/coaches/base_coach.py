import abc
from math import radians
import os
import pickle
from argparse import Namespace
import wandb
import os.path
from criteria.localitly_regulizer import Space_Regulizer
import torch
from torchvision import transforms
from lpips import LPIPS
from training.projectors import w_projector
from configs import global_config, paths_config, hyperparameters
from criteria import l2_loss
from models.e4e.psp import pSp2
from utils.log_utils import log_image_from_w
from utils.models_utils import toogle_grad, load_old_G
import math
from torchvision.utils import make_grid
from gen_videos import create_samples
from tqdm import tqdm
from criteria import id_loss
import torch.nn.functional as F

import numpy as np
import mrcfile
import PIL

class BaseCoach:
    def __init__(self, data_loader, use_wandb):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0

        if hyperparameters.first_inv_type == 'w+':
            self.initilize_e4e()
        self.id_loss = id_loss.IDLoss().cuda().eval()

        self.e4e_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G()
        toogle_grad(self.G, True)

        self.original_G = load_old_G()

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot 

    def load_inversions(self, w_path_dir, image_name):
        w = torch.load(w_path_dir + f'{image_name}_ws.npy').to(global_config.device)
        cam = torch.load(w_path_dir + f'{image_name}_cam.npy').to(global_config.device)
        return w, cam

    def calc_inversions(self, image, image_name, cam_encoder, e4e_encoder, outdir):
        id_image = torch.squeeze(image.to(global_config.device))
        ws, cam = w_projector.project(self.G, id_image, device=torch.device(global_config.device), w_avg_samples=5000,
                                num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                                use_wandb=self.use_wandb, cam_encoder=cam_encoder, e4e_encoder=e4e_encoder, outdir=outdir)

        return ws, cam

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0
        real_images_128 = F.interpolate(real_images, size=(128,128), mode = 'area')
        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images['image'], real_images)
            l2_loss_val += l2_loss.l2_loss(generated_images['image_raw'], real_images_128)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images['image'], real_images)
            loss_lpips += self.lpips_loss(generated_images['image_raw'], real_images_128)
            loss_lpips = torch.squeeze(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        # Calculate depth regularization loss
        loss_depth = compute_tv_norm(generated_images['image_depth'].squeeze(0))
        loss += loss_depth

        return loss, l2_loss_val, loss_lpips#, loss_depth

    def forward(self, ws, freezed_cam=None, needs_img_grid=False, grid_num=5, need_shape=False, need_gt_ingrid=None):
        if needs_img_grid:
            
            extrinsics = self.look_at(grid_num=grid_num, num=needs_img_grid)
            intrinsic = torch.Tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1])
            
            images = []
            for i in range(extrinsics.shape[0]):
                cam = torch.cat([extrinsics[i], intrinsic], dim=0).unsqueeze(0).cuda()
                generated_image = self.G.synthesis(ws, cam)['image']
                images.append(generated_image)
            if needs_img_grid == 'large' and need_gt_ingrid != None:
                images.pop(1)
                images.pop(2)
                pred_cam_image = self.G.synthesis(ws, need_gt_ingrid[1])['image']
                images.insert(0, pred_cam_image)
                images.insert(0, need_gt_ingrid[0])

            elif needs_img_grid == 'small':
                pred_cam_image = self.G.synthesis(ws, need_gt_ingrid[1])['image']
                images.insert(0, pred_cam_image)
                images.insert(0, need_gt_ingrid[0])
                grud_num=1
                
            images = torch.cat(images, dim=0)
            image_grid = make_grid(images, nrow=grid_num)

            #extract mesh into mrc
            if need_shape:
                gen_shapes(ws, dir='./')

            return image_grid

        #opt process
        else: 
            generated_images = self.G.synthesis(ws[:, :14, :], freezed_cam[:, :25])
            return generated_images

    def initilize_e4e(self):
        self.e4e_inversion_net = pSp2()
        
        self.e4e_inversion_net.eval()
        self.e4e_inversion_net = self.e4e_inversion_net.to(global_config.device)
        for params in self.e4e_inversion_net.parameters():
            params.requires_grad = True

    def get_e4e_inversion(self, image):
        image = (image + 1) / 2
        new_image = self.e4e_image_transform(image[0]).to(global_config.device)
        print('check image type!!')
        ws = self.e4e_inversion_net(new_image.unsqueeze(0))
        if self.use_wandb:
            log_image_from_w(ws[:, :, :], self.G, 'First e4e inversion')
        return ws

    def gen_shapes(self, ws, dir):
        max_batch=1000000
        samples, _, _ = create_samples()
        samples = samples.to(ws.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=ws.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=ws.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = self.G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], ws.unsqueeze(0), noise_mode='const')['sigma']
                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((256, 256, 256)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)
        
        pad = int(30 * 256 / 256)
        pad_top = int(38 * 256 / 256)
        sigmas[:pad] = 0
        sigmas[-pad:] = 0
        sigmas[:, :pad] = 0
        sigmas[:, -pad_top:] = 0
        sigmas[:, :, :pad] = 0
        sigmas[:, :, -pad:] = 0

        with mrcfile.new_mmap(os.path.join(dir,'shape.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
            mrc.data[:] = sigmas

    def look_at(self, grid_num=5, radius=2.7, num='small'):
        '''
        num_grid should be odd number!
        '''
        at=torch.Tensor([0, 0, 0])
        up=torch.Tensor([0, 0, 1])
        eyes = self.gen_eyes(grid_num=grid_num, num='small')

        mats = []
        for eye in eyes:
            tmp = torch.Tensor([0,0,0,1])
            z_axis = eye - at
            x_axis = torch.cross(up, z_axis)
            x_axis = x_axis/torch.norm(x_axis)
            y_axis = torch.cross(z_axis, x_axis)
            y_axis = y_axis/torch.norm(y_axis)
            mat = torch.stack([x_axis, y_axis, z_axis], dim = -1)
            #x_axis, y_axis, z_axis/torch.norm(z_axis)

            ## For use of EG3D cam convention
            x, y, z = -mat[0], -mat[1], -mat[2]
            mat_revise = torch.stack([x,z,y], dim=0)

            mat_tmp = mat_revise.clone()

            loc = -mat_tmp[:, 2]*radius

            mat_revise = torch.cat([mat_revise, loc.unsqueeze(1)], dim=1)
            mat_revise = mat_revise.view(1,12).squeeze(0)
            mat_16 = torch.cat([mat_revise, tmp], dim=0)
            mats.append(mat_16)
        
        mats = torch.stack(mats, dim=0)
        return mats


    def gen_eyes(self, grid_num=5, coeff=8, num='small'):
        '''coeff decides sparsity of cameras'''
        all_xyz = []
        if num == 'small':
            num_images=2
            for i in range(num_images):
                if i == 0:
                    num = 1
                    x,y,z = 0,1,0
                    xyz = torch.Tensor([x,y,z])
                    all_xyz.append(xyz)
                else:
                    y = math.cos(math.pi/coeff)
                    x = math.sin(math.pi/coeff)
                    z = 0
                    x2 = -x
                    xyz_R = torch.Tensor([x,y,z])
                    xyz_L = torch.Tensor([x2,y,z])
                    all_xyz.insert(0, xyz_R)
                    all_xyz.append(xyz_L)
        elif num == 'large':
            half = int((grid_num)/2+1) 
            for i in range(half):
                if i == 0:
                    num = 1
                    x,y,z = 0,1,0
                    xyz = torch.Tensor([x,y,z])
                    all_xyz.append(xyz)
                else:
                    num = int(coeff*i)
                    y = math.cos(math.pi/coeff/(half-1)*i)
                    y_sin = math.sin(math.pi/coeff/(half-1)*i)
                    for i in range(num):
                        x = y_sin * math.cos(2*math.pi/num*(i+1))
                        z = y_sin * math.sin(2*math.pi/num*(i+1))
                        xyz = torch.Tensor([x,y,z])
                        all_xyz.append(xyz)

        all_xyz = torch.stack(all_xyz, dim=0)
        return all_xyz


def compute_tv_norm(values, losstype='l2', weighting=None):  # pylint: disable=g-doc-args
    """Returns TV norm for input values.
    Note: The weighting / masking term was necessary to avoid degenerate
    solutions on GPU; only observed on individual DTU scenes.
    """
    v00 = values[:, :-1, :-1]
    v01 = values[:, :-1, 1:]
    v10 = values[:, 1:, :-1]
    loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    loss = torch.mean(torch.mean(loss))
    
    return loss