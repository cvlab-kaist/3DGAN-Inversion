import torch
import torch.nn.functional as F
from .explainability_network.loss_functions import photometric_reconstruction_loss


def calc_warping_loss(ws, canonical_cam, extrinsic, init_ext, intrinsic, depth, target_images, G, torch_vgg, ray_generator, layers = '14'):
    canonical_dict = G.synthesis(ws, canonical_cam, noise_mode='const', force_fp32=True)
    can_images = canonical_dict['image']
    if can_images.shape[2] > 256:
        can_images = F.interpolate(can_images, size=(256, 256), mode='area')

    # make loss only from the foreground area
    depth_mean = torch.mean(depth)
    depth_zeros = torch.zeros_like(depth_mean).cuda()
    depth_ones = torch.ones_like(depth_mean).cuda()
    masked_depths = torch.where(depth<depth_mean, depth_ones, depth_zeros)
    
    ray_origins2, ray_dirs2 = ray_generator(extrinsic, intrinsic.reshape(3,3).unsqueeze(0), depth.shape[-1]) #world space
    
    # Calculate the surface points
    cam_xyz1 = ray_generator.calculate_xyz_of_depth(ray_origins2, ray_dirs2, depth) # world space
    cam_xyz = cam_xyz1[:3].permute(1,0)#grad only goes to extrinsic
    init_trans = init_ext[:, :3, 3]
    
    canonical_cam_origin = init_trans.repeat(cam_xyz.shape[0], 1) # Camera origin
    vectors = cam_xyz - canonical_cam_origin # Ray direction
    plane_norm_vector = -canonical_cam_origin # Norm vector orthogonal to image plane
    plane_point = torch.bmm(init_ext.reshape(-1,4,4), torch.Tensor([[0,0,1,1]]).unsqueeze(-1).cuda()).squeeze(-1).repeat(cam_xyz.shape[0], 1)[:, :3] # Select a point on the image plane

    #Calculate intersections 
    intersections = LinePlaneCollision(plane_norm_vector, plane_point, vectors, canonical_cam_origin) # N, 3
    tmp_ones = torch.ones(intersections.shape[0], 1).cuda()
    intersections1 = torch.cat([intersections, tmp_ones], dim=-1).permute(1,0) # N, 4
    
    torch_target_features = get_features(target_images, torch_vgg, layers)
    torch_synth_features = get_features(can_images, torch_vgg, layers) 
    
    # Normalize to uv coordinate
    w2c = torch.linalg.inv(init_ext.reshape(4,4)) 
    pred_uv = torch.mm(w2c, intersections1)[:3].permute(1,0)
    pred_uv = pred_uv/pred_uv[:, 2:]
    pred_uv = torch.mm(intrinsic.reshape(3,3), pred_uv.permute(1,0))[:2].permute(1,0)
    pred_uv = (pred_uv-0.5)*2 #128, 128
    
    res = int(pred_uv.shape[0]**(1/2))

    # Sample feature map by pred_uv
    target_feature_res = torch_target_features.shape[-1]
    pred_uv_resized = F.interpolate(pred_uv.reshape(1, res,res,-1).permute(0,3,1,2), size=(target_feature_res, target_feature_res), mode='bilinear').permute(0,2,3,1)
    warpped_feature = F.grid_sample(torch_synth_features, pred_uv_resized, mode='bilinear', align_corners=False)
    warpped_image = F.grid_sample(can_images, pred_uv.reshape(1, res,res,-1), mode='bilinear', align_corners=False)
    masked_depths = F.interpolate(masked_depths, size=(target_feature_res, target_feature_res), mode='bilinear') 
    
    loss = photometric_reconstruction_loss(warpped_feature, torch_target_features, masked_depths)
    
    return loss, warpped_image

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    '''
    every input should be in (N, 3) shape
    return Psi is also B, N
    reference : https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72, 
                https://discuss.pytorch.org/t/dot-product-batch-wise/9746
    '''
    ndotu = torch.bmm(planeNormal.unsqueeze(1), rayDirection.unsqueeze(-1)).squeeze(-1) #B, 1
    if abs(torch.min(ndotu)) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w_vec = rayPoint - planePoint
    si = -torch.bmm(planeNormal.unsqueeze(1), w_vec.unsqueeze(-1)).squeeze(-1) / ndotu
    Psi = w_vec + si * rayDirection + planePoint
    return Psi

def get_features(x, model, layers):
    layer_list = []
    
    for _, layer in enumerate(model.children()): # 0, conv
        layer_list.append(layer)
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