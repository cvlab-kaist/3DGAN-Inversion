from doctest import TestResults
from gc import freeze
from locale import normalize
import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from torchvision.utils import save_image
import PIL
import numpy as np
import mrcfile

from criteria import l2_loss
from pytorch_msssim import ms_ssim
from gen_videos import gen_interp_video


class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self, cam_encoder, e4e_encoder):

        folder_dir = paths_config.output_data_path
        use_ball_holder = True
        iters=0
        for fname, image in tqdm(self.data_loader): # for each face samples
            iters+=1
            image_name = fname[0]
    
            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break
            ckpt_dir = os.path.join(paths_config.embedding_dir, folder_dir[2:])

            if os.path.isdir(folder_dir) == 0:
                os.mkdir(folder_dir)
            if os.path.isdir(folder_dir + '_pivot') == 0:
                os.mkdir(folder_dir + '_pivot')

            if hyperparameters.use_last_w_pivots:
                w_pivot, freezed_cam = self.load_inversions(ckpt_dir, image_name)
            else:
                w_pivot, freezed_cam = self.calc_inversions(image, image_name, cam_encoder, e4e_encoder, folder_dir)
            freezed_cam.requires_grad=False
            w_pivot = w_pivot.to(global_config.device)

            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            
            with torch.no_grad():
                generated_image_grid = self.forward(w_pivot, needs_img_grid='small', grid_num=5, need_gt_ingrid=(real_images_batch, freezed_cam))
                intimg = (generated_image_grid.permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(intimg.cpu().numpy(), 'RGB').save(folder_dir + f'_pivot/{image_name}.png')
                if global_config.gen_video:
                    gen_interp_video(self.G, w_pivot, folder_dir + f'_pivot/{image_name}_pivot.mp4')

            for i in tqdm(range(hyperparameters.max_pti_steps)):
                generated_images = self.forward(w_pivot, freezed_cam)
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                                               self.G, use_ball_holder, w_pivot)
                self.optimizer.zero_grad()
                if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                    break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0
                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1
            with torch.no_grad():
                generated_image_grid = self.forward(w_pivot, needs_img_grid='small', grid_num=5, need_gt_ingrid=(real_images_batch, freezed_cam))
                intimg = (generated_image_grid.permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(intimg.cpu().numpy(), 'RGB').save(folder_dir + f'/{image_name}.png')
                if global_config.gen_video:
                    gen_interp_video(self.G, w_pivot, folder_dir + f'/{image_name}.mp4')
                
            if global_config.do_evaluation:
                with torch.no_grad():
                    # save reconstruction
                    synimg = self.G.synthesis(w_pivot[:, :14, :], freezed_cam[:, :25], noise_mode='const', force_fp32=True)['image']
                    synimg = (synimg+1) / 2                
                    
                    image = (image.cuda() + 1) / 2
                    m_mse = l2_loss.l2_loss(synimg, image).item()
                    m_lpips = self.lpips_loss(synimg, image).item()
                    m_msssim = ms_ssim(synimg, image, data_range=1, size_average=False ).item()
                    synimg = synimg*2 - 1
                    image = image*2 - 1
                    m_identity = self.id_loss(synimg, image).item()
                    
                    # save metrics to txt:
                    with open(os.path.join(folder_dir, f"{image_name}metrics.txt"), "w") as f:
                        f.write("mse: {}\n".format(m_mse))
                        f.write("lpips: {}\n".format(m_lpips))
                        f.write("msssim: {}\n".format(m_msssim))
                        f.write("identity: {}\n".format(m_identity))
                    
                    # save mesh 
                    if global_config.gen_mesh:
                        create_geometry(self.G, w_pivot, outdir = folder_dir, fname = str(image_name)+"_pti")
                    #save pivots
                    if global_config.save_pivot:
                        cam_np = freezed_cam.clone().detach().cpu().numpy()
                        w_np = torch.from_numpy(w_pivot.clone().detach().cpu())
                        np.save(os.path.join(ckpt_dir, f'{image_name}_cam.npy'), cam_np)
                        np.save(os.path.join(ckpt_dir, f'{image_name}_ws.npy'), w_np)


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