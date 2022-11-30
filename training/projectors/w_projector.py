# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from configs import global_config, hyperparameters
from utils import log_utils
import dnnlib
import math
import PIL
import os
import torchvision.models as models
from ..volumetric_rendering.ray_sampler import RaySampler
from utils.camera_utils import compute_rotation_matrix_from_quaternion, euler2rot, rot6d_to_rotmat
from ..warping_loss import calc_warping_loss

def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        lr_rampdown_length=0.25,
        initial_noise_factor=0.05,
        noise_ramp_length=0.75,
        lr_rampup_length=0.05,
        regularize_noise_weight=1e5,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        cam_encoder = None,
        e4e_encoder = None,
        outdir=None,
        w_name: str
):

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Call other vgg for warping loss
    torch_vgg = models.vgg16(pretrained=True).features.eval().cuda()
    for param in torch_vgg.parameters():
        param.requires_grad_(False)
    layers = '14' #7, 14, 21 -> 128,128,128 / 256, 64, 64 / 512, 32, 32
    
    # Load networks
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore
    cam_predictor = copy.deepcopy(cam_encoder).eval().cuda()
    if global_config.use_quaternions:
        cam_lr = hyperparameters.cam_lr_quat
    elif global_config.use_6d:
        cam_lr = hyperparameters.cam_lr_6d
    else:
        cam_lr = hyperparameters.cam_lr_2d
    
    e4e_enc = copy.deepcopy(e4e_encoder).eval().cuda()
    ray_generator = RaySampler()

    target_e4e = (((target+ 1) / 2) * 255).unsqueeze(0).to(device).to(torch.float32)
    if target_e4e.shape[2] > 256:
        target_e4e = F.interpolate(target_e4e, size=(256, 256), mode='area')
    _ = vgg16(target_e4e, resize_images=False, return_lpips=True) # for normalizing the target image

    radius = 2.7
    init_ext = torch.Tensor([1,0,0,0,\
                        0,-1,0,0,\
                        0,0,-1,2.7,\
                        0,0,0,1]).reshape(-1,4,4).cuda()
    intrinsic = torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).unsqueeze(0).cuda()
    canonical_cam = torch.cat([init_ext.reshape(-1, 16), intrinsic], dim=-1)

    #Calculate mean w
    with torch.no_grad():
        pred_ext_init = euler2rot(torch.tensor([math.pi/2]), torch.tensor([math.pi/2]), torch.zeros(1, 1), batch_size=1)
        # init camera -> canonical pose
        cam_init = torch.cat([pred_ext_init, intrinsic], dim=-1)
    cam_avg_samples = cam_init.repeat(w_avg_samples, 1)
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), cam_avg_samples, truncation_cutoff=14, truncation_psi=0.7)
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32) 
    w_avg = np.mean(w_samples, axis=0, keepdims=True)
    w_avg_tensor = torch.from_numpy(w_avg).to(global_config.device)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    mean_w = initial_w if initial_w is not None else torch.from_numpy(w_avg).cuda()
    start_w = e4e_enc(target_e4e).unsqueeze(1)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}
    noise_bufs2 = {name: buf for (name, buf) in G.superresolution.named_buffers() if 'noise_const' in name}
    
    # Features for target image.
    target_images_contiguous = target.contiguous()
    target_images = (((target+ 1) / 2) * 255).unsqueeze(0).to(device).to(torch.float32)
    
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # Setup optimizer
    w_opt = torch.tensor(mean_w + start_w, dtype=torch.float32, device=device,
                         requires_grad=True)
    start_translation = torch.zeros(1, 3).cuda()
    translation_opt = torch.tensor(start_translation, dtype=torch.float32, device=device,
                         requires_grad=True)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()) + list(noise_bufs2.values()), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)
    cam_optimizer = torch.optim.Adam(cam_predictor.parameters(), lr=cam_lr, betas=(0.9, 0.999))
    translation_optimizer = torch.optim.Adam([translation_opt], lr=hyperparameters.translation_lr)
    
    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True
    for buf in noise_bufs2.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # update without pose encoder
    '''with torch.no_grad():
        theta = torch.zeros(1,1)
        phi = torch.zeros(1,1)
    theta = torch.tensor(theta, dtype=torch.float32, device=device,
                         requires_grad=True)
    phi = torch.tensor(phi, dtype=torch.float32, device=device,
                         requires_grad=True)
    theta_optimizer = torch.optim.Adam([theta], betas=(0.9, 0.999),
                                 lr=hyperparameters.cam_latent_lr)
    phi_optimizer = torch.optim.Adam([phi], betas=(0.9, 0.999),
                                 lr=hyperparameters.cam_latent_lr)'''
    for step in tqdm(range(num_steps)):
        # Calculate rotation matrix
        if global_config.use_quaternions:
            pred_quat = cam_predictor(target_images)
            pred_rotmat = compute_rotation_matrix_from_quaternion(pred_quat)
        elif global_config.use_6d: # this is only for afhq
            pred_6d = cam_predictor(target_images)
            pred_rotmat = rot6d_to_rotmat(pred_6d)
        else:
            pred_angles = cam_predictor(target_images)
            theta = math.pi/2 + pred_angles[:, 0]
            phi = math.pi/2 + pred_angles[:, 1]
            roll = torch.zeros(1, 1)
            pred_rotmat = euler2rot(theta, phi, roll, batch_size=1).reshape(-1, 4, 4)[:, :3, :3]

        # Additional optimizable translation
        pred_ext_tmp = torch.eye(4).unsqueeze(0).repeat(pred_rotmat.shape[0], 1, 1).cuda()
        pred_translation = -radius*pred_rotmat[:, :3, 2]
        pred_ext_tmp[:, :3, :3] = pred_rotmat
        translation_opt_world = -torch.bmm(pred_ext_tmp[:, :3, :3], translation_opt.unsqueeze(-1)) * 2.7
        tmp_translation = translation_opt_world.squeeze(-1) + pred_translation
        tmp_translation = tmp_translation / torch.norm(tmp_translation, dim=-1) * 2.7 # normalize radius to 2.7

        # Formulate extrinsic matrix and input cam
        pred_ext = torch.eye(4).unsqueeze(0).cuda()
        pred_ext[:, :3, 3] = tmp_translation
        pred_ext[:, :3, :3] = pred_ext_tmp[:, :3, :3]
        pred_cam = torch.cat([pred_ext.reshape(-1, 16), intrinsic], dim=-1)

        t = (step - hyperparameters.cam_preheat_steps) / (num_steps - hyperparameters.cam_preheat_steps)
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr 

        # Synth images from opt_w.
        if step<hyperparameters.cam_preheat_steps:
            ws_expand = w_opt.repeat(1,14,1)
        else:
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws_expand = (w_opt+w_noise).repeat(1,14,1)
        pred_dict = G.synthesis(ws_expand, pred_cam, noise_mode='const', force_fp32=True)
        pred_depths = pred_dict['image_depth']
        pred_images = pred_dict['image']* 127.5 + 128

        if global_config.visualize_opt_process:
            if os.path.isdir(outdir + f'_pivot/{w_name}') == 0:
                os.makedirs(outdir + f'_pivot/{w_name}')
            if step%10==0:
                with torch.no_grad():
                    intimg = (pred_images.squeeze(0).permute(1,2,0)).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(intimg.cpu().numpy(), 'RGB').save(outdir + f'_pivot/{w_name}/{step}.png')

        # Calculate warping loss
        warp_loss = None
        ws_clone, canonical_cam_clone = ws_expand.clone().detach(), canonical_cam.clone().detach()
        warp_loss, test_img = calc_warping_loss(ws_clone, canonical_cam_clone, pred_ext, init_ext, intrinsic, pred_depths, target_images_contiguous, \
                                                G, torch_vgg, ray_generator, layers = layers)
        
        if global_config.visualize_warp_process:
            if step%10==0:
                intwarp = (test_img.squeeze(0).permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                if os.path.isdir(f'./warp_image_test/warp_{w_name}') == 0:
                    os.makedirs(f'./warp_image_test/warp_{w_name}')
                PIL.Image.fromarray(intwarp.cpu().numpy(), 'RGB').save(f'./warp_image_test/warp_{w_name}/{step}.png')
            
        # Calculate reconstruction loss
        if pred_images.shape[2] > 256:
            pred_images = F.interpolate(pred_images, size=(256, 256), mode='area')
        synth_features = vgg16(pred_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()          
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        for v in noise_bufs2.values():
            noise = v[None, None, :, :]       
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        loss = dist + reg_loss * regularize_noise_weight
        if warp_loss != None:
            loss += warp_loss

        # Print loss
        # if step%10==0:
        #     print(f'loss: {dist}, warp_loss: {warp_loss}')
        
        # Step
        if step < hyperparameters.cam_preheat_steps:
            cam_optimizer.zero_grad()
            translation_optimizer.zero_grad()
            loss.backward()
            cam_optimizer.step()
            translation_optimizer.step()
        else:
            optimizer.zero_grad()
            cam_optimizer.zero_grad()
            translation_optimizer.zero_grad()
            loss.backward()
            cam_optimizer.step()
            optimizer.step()
            translation_optimizer.step()
        
        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
            for buf in noise_bufs2.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
                
    #freeze encoder for tuning step.
    with torch.no_grad():
        cam = pred_cam.clone().detach()
        ws_expand =ws_expand.clone().detach()
    del G
    del cam_predictor
    del e4e_enc
    torch.cuda.empty_cache()
    return ws_expand, cam
