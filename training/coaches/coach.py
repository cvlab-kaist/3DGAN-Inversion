import torch
import torch.nn.functional as F
from torchvision import transforms

import cv2
import numpy as np
import imageio

from doctest import TestResults
from gc import freeze
from locale import normalize
import os
import shutil
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config

from training.projectors import w_projector
from torchvision.utils import save_image
import PIL
import numpy as np
import mrcfile

from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from models.camera_estimator.resnet import resnet34
from models.latent_encoder.psp import pSp2
from criteria import l2_loss, id_loss
from pytorch_msssim import ms_ssim
from lpips import LPIPS

from criteria.localitly_regulizer import Space_Regulizer
from utils.models_utils import toogle_grad, load_old_G
from utils.grid_view import look_at
from training.projectors import w_projector

class Coach:

    def __init__(self, data_loader):
        # super().__init__(data_loader, use_wandb)
        self.device = global_config.device
        
        # Load camera viewpoint estimator and latent representation encoder
        pi_estimator= resnet34()
        w_encoder = torch.nn.DataParallel(pSp2())
        
        pi_estimator.load_state_dict(torch.load(paths_config.pi_estimator_path))
        w_encoder.load_state_dict(torch.load(paths_config.w_encoder_path, map_location='cpu'))
        self.pi_encoder = pi_estimator.to(self.device)
        self.w_encoder = w_encoder.to(self.device)
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0

        if hyperparameters.first_inv_type == 'w+':
            self.initilize_e4e()
        # self.id_loss = id_loss.IDLoss().cuda().eval()

        self.e4e_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G()
        toogle_grad(self.G, True)

        self.original_G = load_old_G()

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)
        
    def forward(self, ws, freezed_cam=None, need_shape=False):
        if need_shape:
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

            with mrcfile.new_mmap('./test.mrc', overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                mrc.data[:] = sigmas

        cam = freezed_cam.cuda()
        generated_images = self.G.synthesis(ws[:, :14, :], cam[:, :25])#['image']
        return generated_images
            
            
    def train(self):
        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        folder_dir = paths_config.output_data_path
        use_ball_holder = global_config.use_ball_holder
        
        iters=0
        w_pivot_init_save = None 
        for fname, image in tqdm(self.data_loader): # for each face samples
            os.makedirs("output", exist_ok=True)
            iters+=1
            image_name = fname[0]
    
            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None
            

            if os.path.isdir(folder_dir) == 0:
                os.mkdir(folder_dir)

            w_pivot, freezed_cam, w_pivot_init = self.calc_inversions(image, image_name, self.pi_encoder, self.w_encoder, folder_dir, w_pivot_init_save)
            
            freezed_cam.requires_grad=False # freeze when tuning
            w_pivot = w_pivot.to(global_config.device) ## pivot size edit[:, :14, :]

            torch.save(w_pivot, f'{embedding_dir}/0.pt')
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            ckpt_dir = os.path.join(paths_config.checkpoints_dir, folder_dir[2:])
            with torch.no_grad():
                generated_image = self.forward(w_pivot, freezed_cam)['image']
                generated_image = (generated_image + 1) / 2
                save_image(generated_image, os.path.join(paths_config.output_data_path, f'{image_name}.png'))
                self.make_video(w_pivot, grid_num = 9, video_name = f'{image_name}.mp4', show_depth=True)
                    
                
            for i in tqdm(range(hyperparameters.max_pti_steps)):
                generated_images = self.forward(w_pivot, freezed_cam)
                loss, l2_loss_val, loss_lpips, loss_depth = self.calc_loss(generated_images, real_images_batch, image_name, self.G, use_ball_holder, w_pivot)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1
            with torch.no_grad():
                generated_image = self.forward(w_pivot, freezed_cam)['image']
                generated_image = (generated_image + 1) / 2
                save_image(generated_image, os.path.join(paths_config.output_data_path, f'{image_name}_optim.png'))
            
            
            self.make_video(w_pivot, grid_num = 9, video_name = f'{image_name}_optim.mp4', show_depth=True)


            if os.path.isdir(ckpt_dir) != 1:
                os.mkdir(ckpt_dir)
        
            cam_np = freezed_cam.clone().detach().cpu().numpy()
            np.save(os.path.join(ckpt_dir, f'{image_name}_cam.npy'), cam_np)
                
                
    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0
        real_images_128 = F.interpolate(real_images, size=(128,128), mode = 'area')
        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images['image'], real_images)
            l2_loss_val += l2_loss.l2_loss(generated_images['image_raw'], real_images_128)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images['image'], real_images)
            loss_lpips += self.lpips_loss(generated_images['image_raw'], real_images_128)
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch)
            loss += ball_holder_loss_val

        loss_depth = compute_tv_norm(generated_images['image_depth'].squeeze(0))
        loss += loss_depth

        return loss, l2_loss_val, loss_lpips, loss_depth
    
    def calc_metrics(self, synimg, realimg, image_name = ''):
        with torch.no_grad():
            synimg = (synimg+1) / 2
            image = (realimg.cuda() + 1) / 2
            m_mse = l2_loss.l2_loss(synimg, image).item()
            m_lpips = self.lpips_loss(synimg, image).item()
            m_msssim = ms_ssim(synimg, image, data_range=1, size_average=False ).item()
            synimg = synimg*2 - 1
            image = image*2 - 1
            m_identity = self.id_loss(synimg, image).item()
            
            # save metrics to txt:
            with open(os.path.join(paths_config.output_data_path, f"{image_name}metrics.txt"), "w") as f:
                f.write("mse: {}\n".format(m_mse))
                f.write("lpips: {}\n".format(m_lpips))
                f.write("msssim: {}\n".format(m_msssim))
                f.write("identity: {}\n".format(m_identity))
                
    
    
    def make_video(self, ws, grid_num=9, video_name = "temp.mp4",  show_depth = False):
        extrinsics = look_at(grid_num=grid_num)
        single_intrinsic = torch.Tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1])
        
        frames = []
        
        with torch.no_grad():
            for i in range(grid_num**2):
                #import pdb; pdb.set_trace()
                cam = torch.cat([extrinsics[i], single_intrinsic], dim=0).unsqueeze(0).cuda()
                generated = self.G.synthesis(ws, cam, noise_mode='const', force_fp32=True)
                ###################################################################
                generated_image = generated["image"]
                generated_depth = generated["image_depth"]
                
                gen_img = (generated_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                gen_img = gen_img[0].cpu().numpy()

                if show_depth:
                    # import pdb; pdb.set_trace()
                    gen_depth = F.interpolate(generated_depth, size=(512, 512), mode='bilinear', align_corners=False).detach().cpu()
                    gen_depth = np.array(transforms.ToPILImage(mode='L')(gen_depth[0]))
                    gen_depth = cv2.cvtColor(gen_depth, cv2.COLOR_GRAY2RGB)
                    
                    gen_img = cv2.hconcat([gen_img, gen_depth])
                frames.append(gen_img)
            
            imageio.mimwrite(os.path.join(paths_config.output_data_path, video_name), frames, fps=15)


    def calc_inversions(self, image, image_name, encoder, e4e_encoder, outdir, w_pivot_init):
        id_image = torch.squeeze(image.to(global_config.device))
        ws, cam, ws_init = w_projector.project(self.G, id_image, device=torch.device(global_config.device), w_avg_samples=5000,
                                num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                                encoder=encoder, e4e_encoder=e4e_encoder, w_pivot_init=w_pivot_init)
        return ws, cam, ws_init




def create_geometry(G, ws, outdir, fname, shape_res = 512, shape_format = '.mrc'):
    # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
    max_batch=1000000
    device = global_config.device
    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
    samples = samples.cuda()
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                # import pdb; pdb.set_trace()
                coordinates = samples[:, head:head+max_batch]
                directions = transformed_ray_directions_expanded[:, :samples.shape[1]-head]
                planes = G.backbone.synthesis(ws, update_emas = False, noise_mode='const')
                planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
                sigma = G.renderer.run_model(planes, G.decoder, coordinates, directions, G.rendering_kwargs)['sigma']
            
                sigmas[:, head:head+max_batch] = sigma
                head += max_batch
                pbar.update(max_batch)

    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # Trim the border of the extracted cube
    pad = int(30 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value

    if shape_format == '.ply':
        from shape_utils import convert_sdf_samples_to_ply
        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, str(fname) + '.ply'), level=10)
    elif shape_format == '.mrc': # output mrc
        with mrcfile.new_mmap(os.path.join(outdir, str(fname) + '.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
            mrc.data[:] = sigmas
            
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size  



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