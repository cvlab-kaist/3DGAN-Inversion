# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
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
import cv2
import torchvision.models as models
from ..volumetric_rendering.ray_sampler import RaySampler
import torch.nn as nn
from ..explainability_network.loss_functions import photometric_reconstruction_loss, explainability_loss
from ..explainability_network.PoseExpNet import PoseExpNet
from torchvision import transforms

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device,
        initial_w=None,
        encoder = None,
        e4e_encoder = None,
        w_pivot_init=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str
):
    #assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore
    cam_predictor = copy.deepcopy(encoder).eval().cuda()
    e4e_enc = copy.deepcopy(e4e_encoder).eval().cuda()
    ray_generator = RaySampler()

    target_e4e = (((target+ 1) / 2) * 255).unsqueeze(0).to(device).to(torch.float32)
    
    if target_e4e.shape[2] > 256:
        target_e4e = F.interpolate(target_e4e, size=(256, 256), mode='area')
    _ = vgg16(target_e4e, resize_images=False, return_lpips=True)


    


    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    
    intrinsic = torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1]).unsqueeze(0).cuda()
    with torch.no_grad():
        thetaphi_init = encoder(target.unsqueeze(0))
        thetaphi_init = thetaphi_init - 0.5

        theta_init = thetaphi_init[:, :1] * math.pi/4
        phi_init = thetaphi_init[:, 1:2] * math.pi/4*2 
        roll_init = thetaphi_init[:, 2:] * math.pi/2 
        pred_ext_init = euler2rot_roll(torch.tensor([math.pi/2]), torch.tensor([math.pi/2]), torch.zeros(phi_init.shape[0], 1), batch_size=phi_init.shape[0])
        #init camera should be at the canonical pose
        cam_init = torch.cat([pred_ext_init, intrinsic], dim=-1)
    cam_avg_samples = cam_init.repeat(w_avg_samples, 1)
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), cam_avg_samples, truncation_cutoff=14)  # [N, L, C] ###truncation
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C] #jaehoon edit -> N, 14, C
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_avg_tensor = torch.from_numpy(w_avg).to(global_config.device)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    mean_w = initial_w if initial_w is not None else torch.from_numpy(w_avg).cuda()#jaehoon edit
    start_w = e4e_enc(target_e4e).unsqueeze(1)
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}##jaehoon edit
    
    target_images = (((target+ 1) / 2) * 255).unsqueeze(0).to(device).to(torch.float32)
    
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_mean = torch.tensor(mean_w, dtype=torch.float32, device=device, 
                         requires_grad=True)  # pylint: disable=not-callable
    start_depthcam = (torch.zeros(1, 6)).cuda()
    start_translation = torch.zeros(1, 3).cuda()
    translation_opt = torch.tensor(start_translation, dtype=torch.float32, device=device,
                         requires_grad=True)
    

    w_opt = w_mean
    
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)
    translation_optimizer = torch.optim.SGD([translation_opt], lr=hyperparameters.translation_lr)


    
    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    torch_vgg = models.vgg16(pretrained=True).features.eval().cuda()
    for param in torch_vgg.parameters():
        param.requires_grad_(False)

    radius = 2.7
    depth_step_cfg = -1
    encoder_finetune_cfg = -1
    # depth_step_cfg = 3
    # encoder_finetune_cfg = 1
    #################################
    with torch.no_grad():
        thetaphi = cam_predictor(target_images)
        theta = torch.zeros_like(thetaphi[:, :1])
        phi = torch.zeros_like(thetaphi[:, 1:])
    theta = torch.tensor(theta, dtype=torch.float32, device=device,
                         requires_grad=True)
    phi = torch.tensor(phi, dtype=torch.float32, device=device,
                         requires_grad=True)
    theta_optimizer = torch.optim.Adam([theta], betas=(0.9, 0.999),
                                 lr=hyperparameters.cam_latent_lr)
    phi_optimizer = torch.optim.Adam([phi], betas=(0.9, 0.999),
                                 lr=hyperparameters.cam_latent_lr)
    for step in tqdm(range(num_steps)):
        
        pred_ext = euler2rot(math.pi/2 + theta, math.pi/2 + phi, batch_size=phi.shape[0])
        pred_ext_with_trans = torch.eye(4).unsqueeze(0).cuda()

        translation_opt_world = -torch.bmm(pred_ext[:, :3, :3], translation_opt.unsqueeze(-1)) * 2.7
        
        tmp_translation = translation_opt_world.squeeze(-1) + pred_ext[:, :3, 3]
        tmp_translation = tmp_translation / torch.norm(tmp_translation, dim=-1) * 2.7
        pred_ext_with_trans[:, :3, 3] = tmp_translation
        pred_ext_with_trans[:, :3, :3] = pred_ext[:, :3, :3]
        
        
        init_ext = torch.eye(4).unsqueeze(0).repeat(thetaphi.shape[0], 1, 1).cuda()
        
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        w_noise = torch.randn_like(w_opt) * w_noise_scale/100
        w_opt = w_opt# + w_noise
        ws_expand = w_opt.repeat(1,14,1)
        

        start_6d = torch.Tensor([[1,0,0,0,-1,0]]).cuda()
        start_rot = rot6d_to_rotmat(start_6d)
        
        start_translation = -radius*start_rot[:, :3, 2]
        
        init_ext[:, :3, :3] = start_rot
        init_ext[:, :3, 3] = start_translation
        canonical_cam = torch.cat([init_ext.reshape(-1, 16), intrinsic], dim=-1)
        warp_cam = torch.cat([pred_ext_with_trans.reshape(-1, 16), intrinsic], dim=-1)
        canonical_dict = G.synthesis(ws_expand, canonical_cam, noise_mode='const', force_fp32=True)
        warp_dict = G.synthesis(ws_expand, warp_cam, noise_mode='const', force_fp32=True)
        synth_depths = canonical_dict['image_depth']
        synth_images = canonical_dict['image']
        warp_images = warp_dict['image']   # 1, 3, 512, 512
        warp_images = (warp_images + 1) * (255 / 2)
        
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
        warp_images = F.interpolate(warp_images, size=(256, 256), mode='area')

        warp_loss = None
        if step>depth_step_cfg:
            depth_mean = torch.mean(synth_depths)
            depth_zeros = torch.zeros_like(depth_mean).cuda()
            depth_ones = torch.ones_like(depth_mean).cuda()
            masked_depths = torch.where(synth_depths<depth_mean, depth_ones, depth_zeros)
            
            ray_origins2, ray_dirs2 = ray_generator(init_ext, intrinsic.reshape(3,3).unsqueeze(0), synth_depths.shape[-1])
            
            cam_xyz1 = ray_generator.calculate_xyz_of_depth(ray_origins2, ray_dirs2, synth_depths) 

            cam_xyz = cam_xyz1[:3].permute(1,0)
            pred_translation = pred_ext_with_trans[:, :3, 3]
            
            warpped_cam_origin = pred_translation.repeat(cam_xyz.shape[0], 1) #1-1
            warpped_vector = cam_xyz - warpped_cam_origin #2-1

            plane_norm_vector = -warpped_cam_origin #3-1
            plane_point = torch.bmm(pred_ext_with_trans.reshape(-1,4,4), torch.Tensor([[0,0,1,1]]).unsqueeze(-1).cuda()).squeeze(-1).repeat(cam_xyz.shape[0], 1)[:, :3] #4-1
            

            intersections = LinePlaneCollision(plane_norm_vector, plane_point, warpped_vector, warpped_cam_origin) #N, 3
            tmp_ones = torch.ones(intersections.shape[0], 1).cuda()
            intersections1 = torch.cat([intersections, tmp_ones], dim=-1).permute(1,0)
            try:
                pred_ext_pinv = torch.linalg.inv(pred_ext_with_trans.reshape(4,4))
            except:
                pred_ext_pinv = torch.linalg.inv(pred_ext_with_trans.reshape(4,4))
            
            
            layers = '7' #7, 14, 21 -> 128,128,128 / 256, 64, 64 / 512, 32, 32
            
            
            synth_images_clone = synth_images.clone().detach().cuda() # 카메라만 업데이트 할거니깐 pred_uv로만 업데이트

            torch_target_features = get_features(target_images, torch_vgg, layers)
            torch_synth_features = get_features(synth_images_clone, torch_vgg, layers)
            
            pred_uv = torch.mm(pred_ext_pinv, intersections1)[:3].permute(1,0)
            pred_uv = pred_uv/pred_uv[:, 2:]
            pred_uv = torch.mm(intrinsic.reshape(3,3), pred_uv.permute(1,0))[:2].permute(1,0)
            pred_uv = (pred_uv-0.5)*2
            
            res = int(pred_uv.shape[0]**(1/2))

            target_feature_res = torch_target_features.shape[-1]
            if res != target_feature_res:
                pred_uv_resized = F.interpolate(pred_uv.reshape(1, res,res,-1).permute(0,3,1,2), size=(target_feature_res, target_feature_res), mode='bilinear').permute(0,2,3,1)
                warpped_target_feature = F.grid_sample(torch_target_features, pred_uv_resized, mode='bilinear', align_corners=False)
                warpped_target_image = F.grid_sample(target_images, pred_uv_resized, mode='bilinear', align_corners=False)
                masked_depths = F.interpolate(masked_depths, size=(target_feature_res, target_feature_res), mode='bilinear')
            else:
                warpped_target_feature = F.grid_sample(torch_target_features, pred_uv.reshape(1, res,res,-1), mode='bilinear', align_corners=False)    
            warp_loss = photometric_reconstruction_loss(warpped_target_feature, torch_synth_features, masked_depths)/10
        
        
        synth_features = vgg16(warp_images, resize_images=False, return_lpips=True)
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
        loss = dist + reg_loss * regularize_noise_weight/10
        if warp_loss != None:
            loss += warp_loss

        if 0<=step<400:
            if (step//10)%2==1:
                optimizer.zero_grad()
                #translation_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #translation_optimizer.step()
            
            else:
                #optimizer.zero_grad()
                theta_optimizer.zero_grad()
                phi_optimizer.zero_grad()
                translation_optimizer.zero_grad()
                loss.backward()
                #optimizer.step()
                theta_optimizer.step()
                phi_optimizer.step()
                translation_optimizer.step()
        else:
            optimizer.zero_grad()
            theta_optimizer.zero_grad()
            phi_optimizer.zero_grad()
            translation_optimizer.zero_grad()
            loss.backward()
            theta_optimizer.step()
            phi_optimizer.step()
            optimizer.step()
            translation_optimizer.step()
        
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
        #import pdb; pdb.set_trace()
        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
                
        
    del G
    
    #freeze encoder for tuning step.
    with torch.no_grad():
        cam = warp_cam.clone().detach()
        ws_expand =ws_expand.clone().detach()
    del cam_predictor
    del e4e_enc
    w_init = (ws_expand[:, 0, :] - mean_w)
    w_init.requires_grad = False
    torch.cuda.empty_cache()
    return ws_expand, cam, w_init#.repeat([1, 18, 1]) #jaehoon edit W+ space인것 같은데..


def euler2rot(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=2.7, batch_size=32, device='cpu'):

    theta = horizontal_mean.cuda()
    phi = vertical_mean.cuda()

    camera_origins = torch.zeros((batch_size, 3), device=device).cuda()
    #import pdb; pdb.set_trace()
    try:
        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)
    except:
        import pdb; pdb.set_trace()

    forward_vectors = normalize_vecs(-camera_origins)
    
    return create_cam2world_matrix(forward_vectors, camera_origins)    

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    #import pdb; pdb.set_trace()
    return cam2world

def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def euler2rot_roll(horizontal_mean, vertical_mean, roll_mean, horizontal_stddev=0, vertical_stddev=0, radius=2.7, batch_size=32, device='cpu'):
    # h = horizontal_mean.cuda()
    # v = vertical_mean.cuda()
    # # h = torch.tensor(horizontal_mean)
    # # v = torch.tensor(vertical_mean)
    # v = torch.clamp(v, 1e-5, math.pi - 1e-5)

    # theta = h
    # v = v / math.pi
    # phi = torch.arccos(1 - 2*v)
    theta = horizontal_mean.cuda()
    phi = vertical_mean.cuda()
    roll = roll_mean.cuda()

    camera_origins = torch.zeros((batch_size, 3), device=device).cuda()
    #import pdb; pdb.set_trace()
    try:
        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)
    except:
        print('batch size does not fit! please check the number of samples and batch size')
        import pdb; pdb.set_trace()

    forward_vectors = normalize_vecs(-camera_origins)
    return create_cam2world_matrix_roll(forward_vectors, camera_origins, roll)   

def create_cam2world_matrix_roll(forward_vector, origin, roll):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotmat_tmp = torch.stack((right_vector, up_vector, forward_vector), axis=-1)
    roll_tensor = roll.cuda()
    
    r1 = torch.cat([torch.cos(roll_tensor), -torch.sin(roll_tensor), torch.zeros(roll_tensor.shape[0],1).cuda()], dim=1)
    r2 = torch.cat([torch.sin(roll_tensor), torch.cos(roll_tensor), torch.zeros(roll_tensor.shape[0],1).cuda()], dim=1)
    r3 = torch.cat([torch.zeros(roll_tensor.shape[0],1).cuda(),torch.zeros(roll_tensor.shape[0],1).cuda(),torch.ones(roll_tensor.shape[0],1).cuda()], dim=1)
    # roll_mat = torch.tensor([[torch.cos(roll_tensor).item(), -torch.sin(roll_tensor).item(), 0],\
    #                         [torch.sin(roll_tensor).item(), torch.cos(roll_tensor).item(), 0], \
    #                         [0,0,1]]).unsqueeze(0).cuda()
    #import pdb; pdb.set_trace()
    roll_mat = torch.stack([r1,r2,r3], dim=1)
    rotation_matrix[:, :3, :3] = torch.bmm(roll_mat, rotmat_tmp)
    origin = -rotation_matrix[:, :3, 2] * 2.7

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    cam2world = cam2world.reshape(-1, 16)
    return cam2world

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    #import pdb; pdb.set_trace()
    x = x.view(-1,2,3) + 1e-4
    a1 = x[:, 0, :]
    a2 = x[:, 1, :]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def rotmat_to_rot6d(x):
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]#canonical : [1,0,0,0,-1,0]
    # first one 
    return torch.cat([a1, a2], dim=-1)
    # second one
    # return x[:, :, :2].reshape(-1 ,6)

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    '''
    every input should be in (N, 3) shape
    return Psi is also B, N
    reference : https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72, 
                https://discuss.pytorch.org/t/dot-product-batch-wise/9746
    '''
    #import pdb; pdb.set_trace()
    #ns = rayDirection.shape[0]
    # planeNormal = planeNormal.repeat(ns, 1)
    # planePoint = planePoint.repeat(ns, 1)
    ndotu = torch.bmm(planeNormal.unsqueeze(1), rayDirection.unsqueeze(-1)).squeeze(-1) #B, 1
    #print(planeNormal.dot(rayDirection))
    if abs(torch.min(ndotu)) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w_vec = rayPoint - planePoint
    si = -torch.bmm(planeNormal.unsqueeze(1), w_vec.unsqueeze(-1)).squeeze(-1) / ndotu
    #si = -planeNormal.dot(w) / ndotu
    Psi = w_vec + si * rayDirection + planePoint
    return Psi


def get_features(x, model, layers):
    #reference : https://deep-learning-study.tistory.com/678
    layer_list = []
    
    for name, layer in enumerate(model.children()): # 0, conv
        layer_list.append(layer)
        #import pdb; pdb.set_trace()
        # if str(name) in layers:
        #     return x
    #import pdb; pdb.set_trace()
    x1 = layer_list[0](x)
    x2 = layer_list[1](x1)
    x3 = layer_list[2](x2)
    x4 = layer_list[3](x3)
    x5 = layer_list[4](x4)
    x6 = layer_list[5](x5)
    x7 = layer_list[6](x6)
    x8 = layer_list[7](x7)
    x9 = layer_list[8](x8)
    x10 = layer_list[9](x9)
    x11 = layer_list[10](x10)
    x12 = layer_list[11](x11)
    x13 = layer_list[12](x12)
    x14 = layer_list[13](x13)
    x15 = layer_list[14](x14)
    x16 = layer_list[15](x15)
    x17 = layer_list[16](x16)
    x18 = layer_list[17](x17)
    x19 = layer_list[18](x18)
    x20 = layer_list[19](x19)
    x21 = layer_list[20](x20)
    x22 = layer_list[21](x21)

    if layers=='7':
        return x8
    elif layers=='14':
        return x15
    elif layers=='21':
        return x22
    else:
        print('layers must be multipliers of 7')
        raise ValueError

    # 0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 1 ReLU(inplace=True)
    # 2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 3 ReLU(inplace=True)
    # 4 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # 5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 6 ReLU(inplace=True)
    ## 7 Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 8 ReLU(inplace=True)
    # 9 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # 10 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 11 ReLU(inplace=True)synth_images
    # 15 ReLU(inplace=True)
    # 16 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # 17 Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 18 ReLU(inplace=True)
    # 19 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 20 ReLU(inplace=True)
    ## 21 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 22 ReLU(inplace=True)
    # 23 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    # 24 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 25 ReLU(inplace=True)
    # 26 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 27 ReLU(inplace=True)
    # 28 Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # 29 ReLU(inplace=True)
    # 30 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    
def write_depth(path, depth, bits=1, absolute_depth=False):
    """Write depth map to pfm and png file.
    Args:
        path (str): filepath without extension
        depth (array): depth
    """

    if absolute_depth:
        out = depth
    else:
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2 ** (8 * bits)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)

    if bits == 1:
        cv2.imwrite(path + ".png", out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif bits == 2:
        cv2.imwrite(path + ".png", out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return